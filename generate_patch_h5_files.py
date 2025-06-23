import os
import pandas as pd
import numpy as np
import h5py

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

# Example usage
base_patch_dir = r"C:\Users\Vivian\Documents\CONCH\all_patches\patches_5x"
output_h5_dir = r"C:\Users\Vivian\Documents\CLAM\CLAM\output_h5\5x"
batch_generate_coords_h5(base_patch_dir, output_h5_dir)
