import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
# import openslide
from tqdm import tqdm

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP, Npy_Patch_Bag
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()


# if __name__ == '__main__':
# 	print('initializing dataset')
# 	csv_path = args.csv_path
# 	if csv_path is None:
# 		raise NotImplementedError

# 	bags_dataset = Dataset_All_Bags(csv_path)
	
# 	os.makedirs(args.feat_dir, exist_ok=True)
# 	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
# 	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
# 	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

# 	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
# 	_ = model.eval()
# 	model = model.to(device)
# 	total = len(bags_dataset)

# 	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

# 	for bag_candidate_idx in tqdm(range(total)):
# 		# slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]

# 		# If slide_ext is provided, use it to extract slide_id ----
# 		if args.slide_ext:
# 			slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
# 		else:
# 			slide_id = bags_dataset[bag_candidate_idx]
# 		# -------------------------------------------------------

# 		bag_name = slide_id+'.h5'
# 		# h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name) # og
# 		h5_file_path = os.path.join(args.data_h5_dir, bag_name)
# 		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
# 		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
# 		print(slide_id)

# 		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
# 			print('skipped {}'.format(slide_id))
# 			continue 

# 		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
# 		time_start = time.time()
# 		# wsi = openslide.open_slide(slide_file_path)
# 		# dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
# 		# 					   		 wsi=wsi, 
# 		# 							 img_transforms=img_transforms)

# 		# replacing with Npy_Patch_Bag -----------------
# 		# Extract label from slide_id prefix (e.g., "FA 100 B1" → "FA")
# 		label = slide_id.split()[0]  # 'FA' or 'PT'
# 		patch_slide_dir = os.path.join(args.data_slide_dir, label, slide_id)
# 		# patch_slide_dir = os.path.join(args.data_slide_dir, slide_id)
# 		dataset = Npy_Patch_Bag(file_path=h5_file_path, patch_dir=patch_slide_dir, img_transforms=img_transforms)
# 		# -------------------------------------

# 		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
# 		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

# 		time_elapsed = time.time() - time_start
# 		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

# 		with h5py.File(output_file_path, "r") as file:
# 			features = file['features'][:]
# 			print('features size: ', features.shape)
# 			print('coordinates size: ', file['coords'].shape)

# 		features = torch.from_numpy(features)
# 		bag_base, _ = os.path.splitext(bag_name)
# 		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))

if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)

	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):
		if args.slide_ext:
			slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		else:
			slide_id = bags_dataset[bag_candidate_idx]

		bag_name = slide_id + '.h5'
		h5_file_path = os.path.join(args.data_h5_dir, bag_name)
		pt_file_path = os.path.join(args.feat_dir, 'pt_files', slide_id + '.pt')

		# Skip if features already extracted
		if not args.no_auto_skip and os.path.exists(pt_file_path):
			print(f"⏩ Skipping {slide_id} (features already extracted)")
			continue

		# Check for patch directory in 40x, then 20x
		label = slide_id.split()[0]  # 'FA' or 'PT'
		patch_dir_40x = os.path.join(args.data_slide_dir, "40x", label, slide_id)
		patch_dir_20x = os.path.join(args.data_slide_dir, "20x", label, slide_id)

		if os.path.isdir(patch_dir_40x):
			patch_slide_dir = patch_dir_40x
			print(f"🔍 Found 40x patches for {slide_id}")
		elif os.path.isdir(patch_dir_20x):
			patch_slide_dir = patch_dir_20x
			print(f"🔍 Found 20x patches for {slide_id}")
		else:
			print(f"⛔ Skipping {slide_id} (no patch directory found in 40x or 20x)")
			continue

		print('\n➡️ Progress: {}/{}'.format(bag_candidate_idx + 1, total))
		print(f"Processing: {slide_id}")

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()

		try:
			dataset = Npy_Patch_Bag(file_path=h5_file_path, patch_dir=patch_slide_dir, img_transforms=img_transforms)
			loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

			output_file_path = compute_w_loader(output_path, loader=loader, model=model, verbose=1)

			time_elapsed = time.time() - time_start
			print(f'\n✅ Feature extraction for {slide_id} completed in {time_elapsed:.2f} seconds.')

			with h5py.File(output_file_path, "r") as file:
				features = file['features'][:]
				print('features size: ', features.shape)
				print('coordinates size: ', file['coords'].shape)

			features = torch.from_numpy(features)
			torch.save(features, pt_file_path)

		except Exception as e:
			print(f"❌ Error processing {slide_id}: {e}")


