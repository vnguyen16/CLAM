# dataset_modules/dataset_wsi_coords.py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import javabridge, bioformats

class WSI_Coord_Bag(Dataset):
    def __init__(self, vsi_path, coords_path, patch_size=224, series=10, img_transforms=None):
        self.vsi_path = vsi_path
        self.coords = np.load(coords_path)  # (N,2) int
        self.patch_size = patch_size
        self.series = series
        self.img_transforms = img_transforms

        ImageReader = javabridge.JClassWrapper("loci.formats.ImageReader")
        self.reader = ImageReader()
        self.reader.setId(vsi_path)
        self.reader.setSeries(series)
        self.channels = self.reader.getRGBChannelCount()

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx].tolist()
        byte_array = self.reader.openBytes(0, int(x), int(y), self.patch_size, self.patch_size)
        tile = np.frombuffer(byte_array, dtype=np.uint8).reshape((self.patch_size, self.patch_size, self.channels))

        img = Image.fromarray(tile)
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        return {'img': img, 'coord': np.array([x, y], dtype=np.int32)}

    def close(self):
        try:
            self.reader.close()
        except Exception:
            pass
