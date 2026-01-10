import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os

from PIL import Image
import h5py

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

# Adding dataset class to handle patches from private dataset
class Npy_Patch_Bag(Dataset):
    def __init__(self, file_path, patch_dir, img_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing 'coords'
            patch_dir (string): Directory containing .npy patches (one folder per slide)
            img_transforms (callable): Torch transform (e.g. normalization, ToTensor)
        """
        self.file_path = file_path
        self.patch_dir = patch_dir
        self.roi_transforms = img_transforms

        with h5py.File(self.file_path, "r") as f:
            self.coords = f['coords'][:]
        
        self.length = len(self.coords)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        patch_name = f"patch_{idx}_x{x}_y{y}.npy"
        patch_path = os.path.join(self.patch_dir, patch_name)

        patch = np.load(patch_path)

        # If shape is (C, H, W), transpose to (H, W, C)
        if patch.shape[0] == 3:
            patch = patch.transpose(1, 2, 0)

        patch = patch.astype(np.uint8)

        # Convert to PIL Image
        patch = Image.fromarray(patch)

        # Apply torchvision transforms
        if self.roi_transforms:
            patch = self.roi_transforms(patch)

        return {'img': patch, 'coord': self.coords[idx]}

	
# ----------------------------------------------
# adding dataset class to handle patches from breakhis
class Png_Patch_Bag(Dataset):

    def __init__(self, file_path, patch_dir, img_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing 'coords'
            patch_dir (string): Directory containing .png patches
            img_transforms (callable): Torch transform (e.g. normalization, ToTensor)
        """
        self.file_path = file_path
        self.patch_dir = patch_dir
        self.roi_transforms = img_transforms

        with h5py.File(self.file_path, "r") as f:
            self.coords = f['coords'][:]

        self.patch_files = sorted([
            f for f in os.listdir(self.patch_dir) 
            if f.lower().endswith(".png")
        ])

        assert len(self.patch_files) == len(self.coords), \
            f"Number of patches ({len(self.patch_files)}) != number of coords ({len(self.coords)})"

        self.length = len(self.coords)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        patch_name = self.patch_files[idx]
        patch_path = os.path.join(self.patch_dir, patch_name)

        patch = Image.open(patch_path).convert("RGB")

        if self.roi_transforms:
            patch = self.roi_transforms(patch)

        return {'img': patch, 'coord': self.coords[idx]}
	
# ----------------------------------------------
# adding dataset class for Amoon MSI patches 
import re
from pathlib import Path


class MSI_Patch_Bag(torch.utils.data.Dataset):
# class DirNpyPatchBag(torch.utils.data.Dataset):
    FILENAME_RE = re.compile(r"^patch_(\d+)_(\d+)_(\d+)um\.npy$", re.IGNORECASE)
    def __init__(self, patch_dir: str, img_transforms=None):
        self.patch_dir = Path(patch_dir)
        self.img_transforms = img_transforms
        self.files, self.coords, self.sizes = [], [], []

        for p in sorted(self.patch_dir.glob("*.npy")):
            # (optional) skip tiny/placeholder files
            try:
                if p.stat().st_size <= 128:
                    continue
            except Exception:
                continue
            m = self.FILENAME_RE.match(p.name)
            if not m:
                continue
            row = int(m.group(1))      # y
            col = int(m.group(2))      # x
            size_um = int(m.group(3))  # 100, 200, ...
            self.files.append(p)
            self.coords.append(np.array([col, row], dtype=np.int32))  # [x, y]
            self.sizes.append(np.int32(size_um))

        if len(self.files) == 0:
            print(f"[warn] No files matched pattern in {self.patch_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # robust path-to-str for np.load on Windows
        try:
            arr = np.load(str(self.files[idx]))
        except Exception as e:
            print("Failed path:", self.files[idx])
            raise

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        if arr.dtype != np.uint8:
            a_min, a_max = arr.min(), arr.max()
            if arr.dtype.kind == 'f' and 0.0 <= a_min and a_max <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        return {
            'img': img,
            'coord': self.coords[idx],                          # np.int32[2]
            'size_um': torch.tensor(self.sizes[idx], dtype=torch.int32)  # tensor for collate
        }

# without size_um
# class MSI_Patch_Bag2(torch.utils.data.Dataset):	
#     """
#     Loads .npy patches directly from a directory where filenames follow:
#         patch_{row}_{col}_{size}um.npy
#     Returns:
#         {'img': Tensor[C,H,W], 'coord': np.array([x, y], dtype=int)} with x=col, y=row
#     """
#     FILENAME_RE = re.compile(r"^patch_(\d+)_(\d+)_(\d+)um\.npy$", re.IGNORECASE)

#     def __init__(self, patch_dir: str, img_transforms=None):
#         self.patch_dir = Path(patch_dir)
#         self.img_transforms = img_transforms
#         self.files = []
#         self.coords = []

#         for p in sorted(self.patch_dir.glob("*.npy")):
#             m = self.FILENAME_RE.match(p.name)
#             if not m:
#                 # Skip files that don't match the expected pattern
#                 continue
#             row = int(m.group(1))
#             col = int(m.group(2))
#             # size_um = int(m.group(3))  # available if you ever need it
#             self.files.append(p)
#             self.coords.append(np.array([col, row], dtype=np.int32))  # [x, y]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         arr = np.load(self.files[idx])  # expect HxW or HxWx3 (uint8 or float)

#         # ensure 3-channel PIL Image for transforms
#         if arr.ndim == 2:
#             # grayscale -> RGB
#             arr = np.repeat(arr[..., None], 3, axis=2)
#         elif arr.ndim == 3 and arr.shape[2] == 1:
#             arr = np.repeat(arr, 3, axis=2)

#         # convert to uint8 if needed for torchvision transforms
#         if arr.dtype != np.uint8:
#             # scale/clip to 0..255 if it looks like float in 0..1
#             a_min, a_max = arr.min(), arr.max()
#             if arr.dtype.kind == 'f' and 0.0 <= a_min and a_max <= 1.0:
#                 arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
#             else:
#                 arr = np.clip(arr, 0, 255).astype(np.uint8)

#         img = Image.fromarray(arr)
#         if self.img_transforms is not None:
#             img = self.img_transforms(img)

#         return {
#             'img': img,
#             'coord': self.coords[idx]
#         }	
# # ----------------------------------------------
# SPIDER dataset
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class Spider_Png_Patch_Bag(Dataset):
    """
    CLAM-compatible dataset: returns {'img': tensor, 'coord': (x,y)}.
    Coordinates are dummy unless you have real coords elsewhere.
    """
    def __init__(self, df_slide: pd.DataFrame, img_transforms=None):
        self.df = df_slide.reset_index(drop=True)
        self.roi_transforms = img_transforms

        # Dummy coords: shape (N,2)
        self.coords = np.zeros((len(self.df), 2), dtype=np.int32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patch_path = self.df.loc[idx, "patch_path"]
        patch = Image.open(patch_path).convert("RGB")

        if self.roi_transforms:
            patch = self.roi_transforms(patch)

        return {"img": patch, "coord": self.coords[idx]}
    
# ----------------------------------------------

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




