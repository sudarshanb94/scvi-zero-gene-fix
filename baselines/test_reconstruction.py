#!/usr/bin/env python3
"""
Quick test to verify reconstruction extraction is working correctly.
"""

import sys
from pathlib import Path
import pickle

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))

from state_sets_reproduce.models import SCVIPerturbationModel
from cell_load.utils.modules import get_datamodule
from omegaconf import OmegaConf
import yaml


def test_reconstruction(model, dataloader, device='cuda', num_batches=2):
    """Test that reconstruction extraction works correctly."""
    model.eval()
    model = model.to(device)
    
    print("Testing reconstruction extraction...")
    print("=" * 60)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            print(f"\nBatch {batch_idx + 1}:")
            print("-" * 60)
            
            # Extract tensors to check data format
            x_pert, x_basal, pert, cell_type, batch_ids = model.extract_batch_tensors(batch)
            
            print(f"x_pert shape: {x_pert.shape}")
            print(f"x_pert dtype: {x_pert.dtype}")
            print(f"x_pert min/max/mean: {x_pert.min():.2f} / {x_pert.max():.2f} / {x_pert.mean():.2f}")
            print(f"x_pert has zeros: {(x_pert == 0).any().item()}")
            print(f"x_pert zero percentage: {100 * (x_pert == 0).sum().item() / x_pert.numel():.2f}%")
            
            print(f"\nx_basal shape: {x_basal.shape}")
            print(f"x_basal min/max/mean: {x_basal.min():.2f} / {x_basal.max():.2f} / {x_basal.mean():.2f}")
            print(f"Is raw counts? (max > 25.0): {x_basal.max() > 25.0}")
            
            # Test predict_step (state repo method)
            try:
                batch_preds = model.predict_step(batch, batch_idx)
                
                print(f"\n✓ predict_step succeeded")
                print(f"  Keys in batch_preds: {list(batch_preds.keys())}")
                
                if "preds" in batch_preds and batch_preds["preds"] is not None:
                    x_recon = batch_preds["preds"].cpu().numpy()
                    print(f"  x_recon (preds) shape: {x_recon.shape}")
                    print(f"  x_recon min/max/mean: {x_recon.min():.2f} / {x_recon.max():.2f} / {x_recon.mean():.2f}")
                    print(f"  x_recon is log-normalized? (max <= 10.0): {x_recon.max() <= 10.0}")
                
                if "X" in batch_preds and batch_preds["X"] is not None:
                    x_real = batch_preds["X"].cpu().numpy()
                    print(f"  x_real (X) shape: {x_real.shape}")
                    print(f"  x_real min/max/mean: {x_real.min():.2f} / {x_real.max():.2f} / {x_real.mean():.2f}")
                    print(f"  x_real is log-normalized? (max <= 10.0): {x_real.max() <= 10.0}")
                    
                    # Check if they're in the same space
                    if "preds" in batch_preds and batch_preds["preds"] is not None:
                        print(f"\n  Comparison:")
                        print(f"    x_real and x_recon same shape: {x_real.shape == x_recon.shape}")
                        print(f"    x_real and x_recon same range: {abs(x_real.max() - x_recon.max()) < 5.0}")
                        print(f"    Mean difference: {np.abs(x_real - x_recon).mean():.4f}")
                
            except Exception as e:
                print(f"\n✗ predict_step failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # Test direct module.forward to get raw reconstruction
            try:
                # Convert to counts if needed
                if x_basal.max() <= 25.0:
                    x_basal_counts = torch.exp(x_basal) - 1
                    print(f"\n  Converted x_basal from log-normalized to counts")
                else:
                    x_basal_counts = x_basal
                    print(f"\n  Using x_basal as raw counts")
                
                encoder_outputs, decoder_outputs = model.module.forward(
                    x_basal_counts, pert, cell_type, batch_ids
                )
                
                if model.recon_loss == "gauss":
                    x_recon_raw = decoder_outputs["px"].loc
                else:
                    x_recon_raw = decoder_outputs["px"].mu
                
                print(f"  ✓ module.forward succeeded")
                print(f"    x_recon_raw (counts) shape: {x_recon_raw.shape}")
                print(f"    x_recon_raw min/max/mean: {x_recon_raw.min():.2f} / {x_recon_raw.max():.2f} / {x_recon_raw.mean():.2f}")
                print(f"    x_recon_raw is counts? (max > 25.0): {x_recon_raw.max() > 25.0}")
                
                # Test normalization
                x_recon_normalized = model._log_normalize_expression(x_recon_raw, target_sum=1e4)
                print(f"    After _log_normalize_expression:")
                print(f"      min/max/mean: {x_recon_normalized.min():.2f} / {x_recon_normalized.max():.2f} / {x_recon_normalized.mean():.2f}")
                
                # Compare with predict_step output
                if "preds" in batch_preds and batch_preds["preds"] is not None:
                    x_recon_from_predict = batch_preds["preds"].cpu().numpy()
                    x_recon_from_normalize = x_recon_normalized.cpu().numpy()
                    
                    print(f"\n    Comparison (predict_step vs manual normalization):")
                    print(f"      Same shape: {x_recon_from_predict.shape == x_recon_from_normalize.shape}")
                    print(f"      Mean absolute difference: {np.abs(x_recon_from_predict - x_recon_from_normalize).mean():.6f}")
                    print(f"      Max difference: {np.abs(x_recon_from_predict - x_recon_from_normalize).max():.6f}")
                    
                    if np.allclose(x_recon_from_predict, x_recon_from_normalize, atol=1e-4):
                        print(f"      ✓ They match! (within tolerance)")
                    else:
                        print(f"      ⚠️  They differ! This might indicate an issue.")
                
            except Exception as e:
                print(f"\n  ✗ module.forward failed: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    print("\n" + "=" * 60)
    print("✓ Reconstruction test completed successfully!")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test reconstruction extraction')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_module', type=str, required=True,
                       help='Path to data module (or config will be used)')
    parser.add_argument('--num_batches', type=int, default=2,
                       help='Number of batches to test')
    
    args = parser.parse_args()
    
    # Load data module
    if Path(args.data_module).exists():
        print(f"Loading data module from: {args.data_module}")
        with open(args.data_module, 'rb') as f:
            datamodule = pickle.load(f)
    else:
        print(f"Data module not found, recreating from config...")
        config_path = Path(args.checkpoint).parent.parent / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        data_config = OmegaConf.create(config['data'])
        datamodule = get_datamodule(
            name=data_config['name'],
            kwargs=data_config['kwargs'],
            batch_size=data_config['kwargs'].get('batch_size', 128),
            cell_sentence_len=data_config['kwargs'].get('cell_sentence_len', 1)
        )
        datamodule.setup()
    
    dataloader = datamodule.train_dataloader()
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from: {args.checkpoint}")
    model = SCVIPerturbationModel.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
        strict=False
    )
    
    # Test
    success = test_reconstruction(model, dataloader, device=device, num_batches=args.num_batches)
    
    if success:
        print("\n✓ All tests passed! Reconstruction extraction is working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")


if __name__ == '__main__':
    main()

