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
from torchvision import transforms
# import javabridge, bioformats

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP, Npy_Patch_Bag, Png_Patch_Bag, MSI_Patch_Bag, Spider_Png_Patch_Bag
# from dataset_modules.dataset_wsi_coords import WSI_Coord_Bag # extracting feats from WSI using coords
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

def compute_w_loader_MSI(output_path, loader, model, verbose=0):
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches')

    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():
            batch = data['img'].to(device, non_blocking=True)
            coords = data['coord'].numpy().astype(np.int32)        # shape [B,2]

            # optional size vector
            sizes = None
            if 'size_um' in data:
                sizes = data['size_um'].numpy().astype(np.int32)   # shape [B]

            features = model(batch).cpu().numpy().astype(np.float32)

            asset = {'features': features, 'coords': coords}
            if sizes is not None:
                asset['size_um'] = sizes

            save_hdf5(output_path, asset, attr_dict=None, mode=mode)
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


def main():
	"private FA PT dataset"
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


def main_breakhis():
    print('Initializing BreakHis dataset')

    allowed_subclasses = {"phyllodes_tumor", "fibroadenoma"}
    target_mag_level = "400X"  # <- Set target magnification level here

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)

    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    model.eval().to(device)

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    for subclass in os.listdir(args.data_slide_dir):
        if subclass.lower() not in allowed_subclasses:
            continue

        subclass_dir = os.path.join(args.data_slide_dir, subclass)
        if not os.path.isdir(subclass_dir):
            continue

        for slide_id in os.listdir(subclass_dir):
            slide_dir = os.path.join(subclass_dir, slide_id)
            if not os.path.isdir(slide_dir):
                continue

            mag_dir = os.path.join(slide_dir, target_mag_level)
            if not os.path.isdir(mag_dir):
                print(f"⛔ No patch dir for {slide_id} at {target_mag_level}")
                continue

            slide_name = slide_id
            bag_name = slide_name + '.h5'
            mag_level = target_mag_level.upper()

            h5_path = os.path.join(args.data_h5_dir, mag_level, bag_name)
            pt_path = os.path.join(args.feat_dir, 'pt_files', f'{slide_name}_{mag_level}.pt')
            save_path = os.path.join(args.feat_dir, 'h5_files', f'{slide_name}_{mag_level}.h5')

            if not os.path.exists(h5_path):
                print(f"⛔ No h5 file found for {slide_name} at {mag_level}")
                continue

            if not args.no_auto_skip and os.path.exists(pt_path):
                print(f"⏩ Skipping {slide_name} {mag_level} (features already exist)")
                continue

            print(f"\n➡️ Processing: {slide_name} ({mag_level})")
            time_start = time.time()

            try:
                img_transforms = transforms.Compose([
					transforms.Resize((224, 224)),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
				])

                dataset = Png_Patch_Bag(file_path=h5_path, patch_dir=mag_dir, img_transforms=img_transforms)
                loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

                output_file_path = compute_w_loader(save_path, loader=loader, model=model, verbose=1)

                features = torch.from_numpy(h5py.File(output_file_path, "r")['features'][:])
                torch.save(features, pt_path)

                elapsed = time.time() - time_start
                print(f"✅ Extracted features for {slide_name} {mag_level} in {elapsed:.2f} sec")

            except Exception as e:
                print(f"❌ Error processing {slide_name} {mag_level}: {e}")

def main_MSI():
    """
    Feature extraction using MSI_Patch_Bag (filename pattern: patch_{row}_{col}_{size}um.npy).

    Modes:
      - Template mode (preferred): --patch_dir_template ".../patches/{slide_id}"
        -> builds per-slide directory with slide_id substituted.
      - Flat-dir mode: no template given -> uses --data_slide_dir as-is.
        -> Suitable if you run one slide per folder/run.
    """
    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError("Please provide --csv_path")

    # list of slide identifiers (same as before)
    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)

    # encoder + transforms
    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    model.eval().to(device)

    loader_kwargs = {'num_workers': 0, 'pin_memory': False} if device.type == "cuda" else {}

    total = len(bags_dataset)
    for i in tqdm(range(total)):
        # Resolve slide_id
        if args.slide_ext:
            slide_id = bags_dataset[i].split(args.slide_ext)[0]
        else:
            slide_id = bags_dataset[i]

        # Output file paths
        bag_name = f"{slide_id}.h5"
        h5_file_path = os.path.join(args.data_h5_dir, bag_name)
        pt_file_path = os.path.join(args.feat_dir, 'pt_files', f"{slide_id}.pt")
        save_h5_path = os.path.join(args.feat_dir, 'h5_files', bag_name)

        # Skip if features already exist
        if not args.no_auto_skip and os.path.exists(pt_file_path):
            print(f"⏩ Skipping {slide_id} (features already extracted)")
            continue

        # Determine patch directory
        # if args.patch_dir_template:
        #     # e.g., "C:/.../patch_root/{slide_id}" -> substitute slide_id
        #     patch_dir = args.patch_dir_template.format(slide_id=slide_id)
        # else:
            # Flat directory provided directly in --data_slide_dir
        patch_dir = args.data_slide_dir

        if not patch_dir or not os.path.isdir(patch_dir):
            print(f"⛔ Skipping {slide_id}: patch directory not found -> {patch_dir}")
            continue

        print(f"\n➡️ Processing: {slide_id}")
        print(f"   • patch_dir = {patch_dir}")

        # Build dataset & loader
        try:
            dataset = MSI_Patch_Bag(patch_dir=patch_dir, img_transforms=img_transforms)
            if len(dataset) == 0:
                print(f"[warn] No .npy files matched 'patch_{{row}}_{{col}}_{{size}}um.npy' in {patch_dir}. Skipping.")
                continue

            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

            # Extract + write h5 (features + coords)
            t0 = time.time()
            output_file_path = compute_w_loader_MSI(save_h5_path, loader=loader, model=model, verbose=1)
            dt = time.time() - t0
            print(f"✅ Feature extraction for {slide_id} completed in {dt:.2f} s.")

            # Save .pt tensor for convenience (same content as features in h5)
            with h5py.File(output_file_path, "r") as file:
                feats_np = file['features'][:]
                coords_np = file['coords'][:]
                print("   • features size:", feats_np.shape)
                print("   • coordinates size:", coords_np.shape)
            feats_t = torch.from_numpy(feats_np)
            torch.save(feats_t, pt_file_path)

        except Exception as e:
            print(f"❌ Error processing {slide_id}: {e}")
            
def main_WSI():
    javabridge.start_vm(class_path=bioformats.JARS)

    vsi_path = r"Z:\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 47 B1.vsi"
    coords_path = r"C:\Users\Vivian\Documents\CONCH\patches_tiled_feats\FA_47_B1_coords.npy"

    slide_id = "FA_47_B1"
    series = 10            # MUST match patch extraction
    patch_size = 224

    model, img_transforms = get_encoder(
        args.model_name,
        target_img_size=patch_size
    )
    model.eval().to(device)

    dataset = WSI_Coord_Bag(
        vsi_path=vsi_path,
        coords_path=coords_path,
        patch_size=patch_size,
        series=series,
        img_transforms=img_transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True
    )

    output_h5 = os.path.join(args.feat_dir, "h5_files", f"{slide_id}.h5")
    compute_w_loader(output_h5, loader, model, verbose=1)

    dataset.close()
    javabridge.kill_vm()


def main_spider():
    import pandas as pd
    print("Initializing SPIDER dataset (FA vs Benign PT)")

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, "pt_files"), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, "h5_files"), exist_ok=True)

    # Load encoder (same as your BreakHis code)
    model, _ = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    model.eval().to(device)

    loader_kwargs = {"num_workers": 8, "pin_memory": True} if device.type == "cuda" else {}

    # Use the same transform style you used before
    img_transforms = transforms.Compose([
        transforms.Resize((args.target_patch_size, args.target_patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # CSV that already filters to FA + Benign PT and has patch_path per row
    index_csv = r'C:\Users\Vivian\Documents\spider\breast_spider_dataset\spider_index_fa_vs_benign_pt.csv'
    df = pd.read_csv(index_csv)

    # Optional: filter to just the 2 classes (extra safety)
    keep = {"Fibroadenoma", "Benign phyllodes tumor"}
    if "class" in df.columns:
        df = df[df["class"].isin(keep)].copy()

    # Optional: decide which split column to use
    split_col = "split_fel" if "split_fel" in df.columns else ("split" if "split" in df.columns else None)

    # Choose naming tag for compatibility with your downstream code
    mag_level = getattr(args, "spider_mag_tag", "SPIDER").upper()

    slide_ids = df["slide_id"].unique().tolist()
    print(f"Found {len(slide_ids)} slides.")

    for slide_id in slide_ids:
        df_slide = df[df["slide_id"] == slide_id].copy()
        if len(df_slide) == 0:
            continue

        slide_name = slide_id
        save_path = os.path.join(args.feat_dir, "h5_files", f"{slide_name}_{mag_level}.h5")
        pt_path   = os.path.join(args.feat_dir, "pt_files", f"{slide_name}_{mag_level}.pt")

        if (not args.no_auto_skip) and os.path.exists(pt_path) and os.path.exists(save_path):
            if split_col:
                split_val = df_slide[split_col].iloc[0]
                print(f"⏩ Skipping {slide_name} ({mag_level}) [{split_val}] (features already exist)")
            else:
                print(f"⏩ Skipping {slide_name} ({mag_level}) (features already exist)")
            continue

        split_msg = f" | {split_col}={df_slide[split_col].iloc[0]}" if split_col else ""
        print(f"\n➡️ Processing: {slide_name} ({mag_level}) | patches={len(df_slide)}{split_msg}")
        time_start = time.time()

        try:
            dataset = Spider_Png_Patch_Bag(df_slide=df_slide, img_transforms=img_transforms)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

            # This uses your existing compute_w_loader unchanged
            output_file_path = compute_w_loader(save_path, loader=loader, model=model, verbose=1)

            # Save .pt features (same as your BreakHis code)
            with h5py.File(output_file_path, "r") as f:
                features = torch.from_numpy(f["features"][:])
            torch.save(features, pt_path)

            elapsed = time.time() - time_start
            print(f"✅ Extracted features for {slide_name} {mag_level} in {elapsed:.2f} sec")

        except Exception as e:
            print(f"❌ Error processing {slide_name} {mag_level}: {e}")

if __name__ == "__main__":
	# private FA PT dataset
	# main()

	# breakhis dataset
	main_breakhis()
      
	# main MSI
    # main_MSI()

    # main WSI with coords
    # main_WSI()

    # main SPIDER
    # main_spider()