import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image


def create_coords_h5_from_patch_map(slide_dir, output_h5_path):
    csv_path = os.path.join(slide_dir, "patch_map.csv")
    if not os.path.isfile(csv_path):
        print(f"⚠️ No patch_map.csv found in: {slide_dir}")
        return

    df = pd.read_csv(csv_path)
    coords = df[['x', 'y']].values.astype(np.int32)

    with h5py.File(output_h5_path, 'w') as hdf:
        hdf.create_dataset('coords', data=coords)
    print(f"✅ Saved: {output_h5_path}")

def batch_generate_coords_h5(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for magnification in ['20x', '40x']:
        mag_dir = os.path.join(base_dir, magnification)
        for label in ['FA', 'PT']:
            label_dir = os.path.join(mag_dir, label)
            for slide_name in os.listdir(label_dir):
                slide_dir = os.path.join(label_dir, slide_name)
                if not os.path.isdir(slide_dir):
                    continue
                output_h5_path = os.path.join(output_dir, f"{slide_name}.h5")
                create_coords_h5_from_patch_map(slide_dir, output_h5_path)


def generate_dummy_h5_row_coords1(patch_root, save_root):
    os.makedirs(save_root, exist_ok=True)

    for subclass in os.listdir(patch_root):
        subclass_dir = os.path.join(patch_root, subclass)
        if not os.path.isdir(subclass_dir):
            continue

        for slide in os.listdir(subclass_dir):
            slide_dir = os.path.join(subclass_dir, slide)
            if not os.path.isdir(slide_dir):
                continue

            patch_files = sorted([
                f for f in os.listdir(slide_dir)
                if f.lower().endswith(".png")
            ])
            coords = []

            for i, patch in enumerate(patch_files):
                coords.append([i * 224, 0])  # dummy row coords spaced by patch size (224x224)

            coords = np.array(coords)

            save_path = os.path.join(save_root, f"{slide}.h5")
            with h5py.File(save_path, "w") as f:
                f.create_dataset("coords", data=coords)

            print(f"✅ Saved coords to {save_path}")

def generate_dummy_h5_row_coords(subclass_dir, save_root):
    os.makedirs(save_root, exist_ok=True)

    # for subclass in os.listdir(patch_root):
    #     subclass_dir = os.path.join(patch_root, subclass)
    #     print(f"Processing subclass: {subclass}") # debug
    #     if not os.path.isdir(subclass_dir):
    #         continue

    for slide in os.listdir(subclass_dir):
        print(f"  Processing slide: {slide}") # debug
        slide_dir = os.path.join(subclass_dir, slide)
        if not os.path.isdir(slide_dir):
            continue

        for mag in os.listdir(slide_dir):
            print(f"    Processing magnification: {mag}")
            mag_dir = os.path.join(slide_dir, mag)
            if not os.path.isdir(mag_dir):
                continue

            patch_files = sorted([
                f for f in os.listdir(mag_dir)
                if f.lower().endswith(".png")
            ])
            coords = []

            for i, patch in enumerate(patch_files):
                coords.append([i * 700, 0])  # 1D row layout

            coords = np.array(coords)

            # Save inside subdirectory for each mag level
            mag_save_dir = os.path.join(save_root, mag)
            os.makedirs(mag_save_dir, exist_ok=True)

            save_path = os.path.join(mag_save_dir, f"{slide}.h5")
            with h5py.File(save_path, "w") as f:
                f.create_dataset("coords", data=coords)

            print(f"✅ Saved coords to {save_path}")

def main():
    base_patch_dir = r"C:\Users\Vivian\Documents\CONCH\all_patches\patches_2.5x"
    output_h5_dir = r"C:\Users\Vivian\Documents\CLAM\CLAM\output_h5\uni_2.5x"
    batch_generate_coords_h5(base_patch_dir, output_h5_dir)

    # # generate dummy h5 files with row coordinates for breakhis patches
    # patch_root = r"C:\Users\Vivian\Documents\breakhis\BreaKHis_v1\BreaKHis_v1\histology_slides\breast\benign\SOB\fibroadenoma"  # path to class (PT or FA)
    # save_root = r"C:\Users\Vivian\Documents\CLAM\CLAM\output_h5\BreaKHis"
    # generate_dummy_h5_row_coords(patch_root, save_root)


if __name__ == "__main__":
    main()
   