import pandas as pd

# put all of the code below into a function

def create_splits_csv(train_split=None, val_split=None, test_split=None, save_path=None):
        


    # Extract the 'Filename' column
    train = train_df["Filename"].reset_index(drop=True)
    val = val_df["Filename"].reset_index(drop=True)
    test = test_df["Filename"].reset_index(drop=True)

    # Ensure all columns have the same length
    max_len = max(len(train), len(val), len(test))
    train = train.reindex(range(max_len), fill_value="")
    val = val.reindex(range(max_len), fill_value="")
    test = test.reindex(range(max_len), fill_value="")

    # Create the combined DataFrame
    splits_df = pd.DataFrame({
        "train": train,
        "val": val,
        "test": test
    })

    # Save with index included
    splits_df.to_csv(save_path, index=True)


def create_binary_split_format(split_csv, save_csv=None):
    # Load wide-format splits
    splits = pd.read_csv(split_csv, index_col=0)

    # Flatten to long format
    long_entries = []

    for split in ['train', 'val', 'test']:
        for slide in splits[split].dropna():
            slide = slide.strip()
            if slide:  # skip empty strings
                long_entries.append((slide, split))

    # Build dataframe
    all_slides = set([slide for slide, _ in long_entries])
    rows = []

    for slide in sorted(all_slides):
        row = {
            "": slide,  # index column
            "train": False,
            "val": False,
            "test": False
        }
        for _, split in filter(lambda x: x[0] == slide, long_entries):
            row[split] = True
        rows.append(row)

    final_df = pd.DataFrame(rows)
    final_df.set_index("", inplace=True)
    final_df.to_csv(save_csv)

def create_descriptor_csv(train_df, val_df, test_df, save_path):

    # Define fixed class order
    class_order = ['FA', 'PT']

    # Count per class
    train_counts = train_df['Class'].value_counts()
    val_counts = val_df['Class'].value_counts()
    test_counts = test_df['Class'].value_counts()

    # Build dataframe with class 0 = FA, class 1 = PT
    summary = pd.DataFrame(columns=['train', 'val', 'test'])

    for cls in class_order:
        summary = pd.concat([
            summary,
            pd.DataFrame({
                'train': [train_counts.get(cls, 0)],
                'val': [val_counts.get(cls, 0)],
                'test': [test_counts.get(cls, 0)]
            })
        ], ignore_index=True)

    # Save to CSV
    summary.to_csv(save_path, index_label="")


if __name__ == "__main__":
    # Load the metadata CSVs for each split
    train_df = pd.read_csv(r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT_k=0\misc\train_split.csv")
    val_df = pd.read_csv(r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT_k=0\misc\val_split.csv")
    test_df = pd.read_csv(r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT_k=0\misc\test_split.csv")
    # create_splits_csv(train_df, val_df, test_df, save_path="splits/fa_vs_pt_701515_100/splits_0.csv")
    # print("Splits CSV created successfully.")

    # create_binary_split_format("splits/fa_vs_pt_701515_100/splits_0.csv", "splits/fa_vs_pt_701515_100/splits_0_bool.csv")

    create_descriptor_csv(train_df, val_df, test_df, save_path='splits/fa_vs_pt_701515_100/splits_0_descriptor.csv')
    print("Descriptor CSV created successfully.")