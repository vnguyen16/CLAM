import os
import h5py
import torch
import argparse
from tqdm import tqdm
import numpy as np

from models.model_clam import CLAM_SB, CLAM_MB
from utils.eval_utils import initiate_model
from utils.file_utils import save_hdf5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_attention_scores(h5_dir, ckpt_path, output_dir, model_type='clam_sb', embed_dim=1024, n_classes=2):
    os.makedirs(output_dir, exist_ok=True)
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith(".h5")]

    # Initialize model
    model_args = argparse.Namespace(
        model_type=model_type,
        drop_out=0.25,
        model_size='small',
        n_classes=n_classes,
        task=None,
        model_size_dict=None,
        model_kwargs=None,
        initiate_fn='initiate_model',
        embed_dim=embed_dim,
    )
    model = initiate_model(model_args, ckpt_path)
    model.to(device)
    model.eval()

    for h5_file in tqdm(h5_files, desc="Processing H5 files"):
        h5_path = os.path.join(h5_dir, h5_file)
        with h5py.File(h5_path, 'r') as f:
            features = torch.tensor(f['features'][:]).to(device)
            coords = f['coords'][:]

        # Forward pass through CLAM
        with torch.no_grad():
            # if isinstance(model, (CLAM_SB, CLAM_MB)): # this works to get attention scores 
            #     logits, Y_prob, Y_hat, A, _ = model(features)
            #     if isinstance(model, CLAM_MB):
            #         A = A[Y_hat.item()]
            #     attention_scores = A.view(-1, 1).cpu().numpy()
            
            if type(model) is CLAM_SB:
                logits, Y_prob, Y_hat, A, _ = model(features)
                # attention_scores = A.view(-1, 1).cpu().numpy()
                attention_scores = A.reshape(-1, 1).cpu().numpy()
                asset_dict = {'attention_scores': attention_scores, 'coords': coords}

            elif type(model) is CLAM_MB:
                logits, Y_prob, Y_hat, A_all, _ = model(features)
                A_all = A_all.cpu().numpy()  # shape (n_classes, n_patches)
                asset_dict = {'coords': coords}
                for class_idx in range(n_classes):
                    scores = A_all[class_idx].reshape(-1, 1)  # fix reshaping here
                    asset_dict[f'attention_scores_class{class_idx}'] = scores
            else:
                raise NotImplementedError("Model type not supported.")

        # Save to new h5
        output_path = os.path.join(output_dir, h5_file)
        # save_hdf5(output_path, {'attention_scores': attention_scores, 'coords': coords}, mode='w') # for og CLAM_SB

        save_hdf5(output_path, asset_dict, mode='w') # for CLAM_MB
        print(f"✅ Saved attention scores to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate attention blockmaps from features.h5")
    parser.add_argument('--h5_dir', type=str, required=True, help='Directory with input .h5 feature files')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to CLAM model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save blockmap .h5 files')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb'], default='clam_sb')
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--n_classes', type=int, default=2)
    args = parser.parse_args()

    generate_attention_scores(
        h5_dir=args.h5_dir,
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        embed_dim=args.embed_dim,
        n_classes=args.n_classes
    )

# -------------------------------------------