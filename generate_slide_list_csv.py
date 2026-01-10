import os
import csv
import re
from collections import defaultdict

def generate_slide_list_csv(h5_dir, output_csv_path):
    """
    Generates a slide_list.csv file from a directory of .h5 coordinate files.
    Each .h5 filename (without extension) is treated as a slide ID.

    Args:
        h5_dir (str): Path to the directory containing .h5 files.
        output_csv_path (str): Path to save the generated CSV.
    """
    slide_ids = []

    for fname in os.listdir(h5_dir):
        if fname.endswith('.h5'):
            slide_id = os.path.splitext(fname)[0]
            slide_ids.append(slide_id)

    slide_ids.sort()

    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['slide_id'])
        for slide_id in slide_ids:
            writer.writerow([slide_id])

    print(f"✅ Saved slide list with {len(slide_ids)} entries to: {output_csv_path}")


def generate_clam_dataset_csv(pt_dir, output_csv, label_map={"FA": 0, "PT": 1}):
    """
    Generate a CLAM dataset CSV with columns: case_id, slide_id, label
    
    Assumes slide_id format like 'FA 47 B1', 'PT 91 C3', etc.
    Slides with the same patient (e.g. FA 47 B1, FA 47 B2) share a case_id (e.g. 'FA 47').
    
    Args:
        pt_dir (str): Path to pt_files directory containing slide .pt files
        output_csv (str): Path to output CSV
        label_map (dict): Mapping of class prefixes (e.g., 'FA', 'PT') to integer labels
    """
    slides = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
    slide_ids = [os.path.splitext(f)[0] for f in slides]

    records = []
    for slide_id in slide_ids:
        parts = slide_id.strip().split()  # e.g., ['FA', '47', 'B1']
        if len(parts) < 2:
            print(f"⚠️ Skipping malformed slide ID: {slide_id}")
            continue

        label_prefix = parts[0]
        patient_id = " ".join(parts[:2])  # e.g., 'FA 47'

        label = label_map.get(label_prefix)
        if label is None:
            print(f"⚠️ Unknown label prefix for slide: {slide_id}")
            continue

        records.append({
            "case_id": patient_id,
            "slide_id": slide_id,
            "label": label
        })

    # Write to CSV
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'slide_id', 'label'])
        writer.writeheader()
        writer.writerows(records)

    print(f"✅ CSV saved with {len(records)} entries at: {output_csv}")


def generate_breakhis_csv(pt_dir, output_csv, label_map={"F": 0, "PT": 1}):
    """
    Generate CLAM-compatible CSV for BreakHis from a flat pt_files directory.

    Args:
        pt_dir (str): Directory containing .pt files (e.g., SOB_B_F_14-9133_40X.pt)
        output_csv (str): Output CSV file path
        label_map (dict): Mapping of tumor subtype code (F, PT) to integer labels
    """
    records = []
    for file in os.listdir(pt_dir):
        if not file.endswith('.pt'):
            continue

        slide_id = os.path.splitext(file)[0]  # e.g., SOB_B_F_14-9133_40X
        parts = slide_id.split('_')  # ['SOB', 'B', 'F', '14-9133', '40X']

        if len(parts) < 4:
            print(f"⚠️ Skipping malformed slide ID: {slide_id}")
            continue

        subclass = parts[2]  # 'F' or 'PT'
        case_id = "_".join(parts[:4])  # e.g., SOB_B_F_14-9133

        label = label_map.get(subclass)
        if label is None:
            print(f"⚠️ Unknown subclass: {subclass} in slide {slide_id}")
            continue

        records.append({
            "case_id": case_id,
            "slide_id": slide_id,
            "label": label
        })

    # Save CSV
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'slide_id', 'label'])
        writer.writeheader()
        writer.writerows(records)

    print(f"✅ Saved {len(records)} entries to {output_csv}")


if __name__ == "__main__":
    # # generate slide_list.csv from .h5 files
    # h5_input_dir = r"C:\Users\Vivian\Documents\CLAM\CLAM\output_h5\BreaKHis\400X"
    # output_csv = r"C:\Users\Vivian\Documents\CLAM\CLAM\output_h5\BreaKHis\breakhis_400x_slide_list.csv"
    # generate_slide_list_csv(h5_input_dir, output_csv)

    # -------------------------------------
    # # generate dataset csv for CLAM
    # pt_dir = r"C:\Users\Vivian\Documents\CLAM\CLAM\FEATURES_DIR_5x\breakhis_uni\40x\pt_files"
    # output_csv = "C:/Users/Vivian/Documents/CLAM/CLAM/dataset_csv/breakhis_fa_pt.csv"
    # generate_clam_dataset_csv(pt_dir, output_csv)

    # -------------------------------------
    # generate dataset csv for BreakHis patches
    generate_breakhis_csv(
    pt_dir=r"C:\Users\Vivian\Documents\CLAM\CLAM\FEATURES_DIR_5x\breakhis_uni\40x\pt_files",
    output_csv="C:/Users/Vivian/Documents/CLAM/CLAM/dataset_csv/breakhis_fa_pt.csv"
    )


