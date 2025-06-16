import os
import csv

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


# Example usage:
if __name__ == "__main__":
    h5_input_dir = r"C:\Users\Vivian\Documents\CLAM\CLAM\output_h5\5x"
    output_csv = "5x_slide_list.csv"
    generate_slide_list_csv(h5_input_dir, output_csv)
