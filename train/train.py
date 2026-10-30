"""Training script for RAW NAFNet denoiser."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.nafnet import NAFNet
from train.dataset import RawDenoiseDataset


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (noisy, clean, target_iso) in enumerate(dataloader):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred = model(noisy)
            loss = criterion(pred, clean)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(
                f"  Batch [{batch_idx + 1:3d}/{len(dataloader):3d}] Loss: {loss.item():.5f} (Avg: {avg_loss:.5f}) - {elapsed:.2f}s",
                flush=True,
            )

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train RAW NAFNet Denoiser")
    parser.add_argument("--raw_dir", type=str, default="data/RAW", help="Directory of CR2 files")
    parser.add_argument("--cache_file", type=str, default="data/train_crops.npy", help="Cache file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--crop_size", type=int, default=128, help="Half-res crop size")
    parser.add_argument("--save_dir", type=str, default="models/checkpoints", help="Save directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    # Dataset & Loader
    dataset = RawDenoiseDataset(
        raw_dir=args.raw_dir,
        cache_file=args.cache_file,
        crop_size=args.crop_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset ready with {len(dataset)} training patches. Batches per epoch: {len(dataloader)}", flush=True)

    # Model
    model = NAFNet(img_channel=5, out_channel=4, width=32).to(device)
    print("Initialized NAFNet (10.64M params)", flush=True)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.6f}) ---", flush=True)
        epoch_loss = train_one_epoch(model, dataloader, optimizer, criterion, scaler, device)
        scheduler.step()
        epoch_sec = time.time() - t0

        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"nafnet_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            },
            checkpoint_path,
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(args.save_dir, "nafnet_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  --> Saved new best checkpoint to {best_path} (Loss: {best_loss:.5f}) in {epoch_sec:.2f}s", flush=True)
        else:
            print(f"  --> Epoch {epoch} completed in {epoch_sec:.2f}s (Loss: {epoch_loss:.5f})", flush=True)

    print("\nTraining completed successfully!", flush=True)


if __name__ == "__main__":
    main()
