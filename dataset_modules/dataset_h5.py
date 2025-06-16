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

# Adding dataset class to handle patches
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

	
# ---------------------

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




