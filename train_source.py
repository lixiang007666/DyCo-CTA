import importlib
import os
import argparse

import torch
import numpy as np

import wandb
from tqdm import tqdm

from dataloaders.TTA_dataloader import TTA_dataloader

from utils.metrics import compute_metrics
from monai.losses import DiceLoss

DEFAULT_DATASET_ROOT = os.environ.get("DYCO_CTA_DATASET_ROOT", "data/TTA_dataset")
DEFAULT_PRETRAINED_ENCODER = os.environ.get("DYCO_CTA_PRETRAINED_ENCODER", "models/pretrained/resnet3d_50.pth")


def train_epoch(model, epoch, num_epochs, train_loader, optimizer, criterion, device, num_classes=2):
    model.train()
    total_loss = 0
    all_metrics = {'dice': [], 'iou': [], 'hd95': []}
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

    for batch_idx, (images, labels, domain_labels) in progress_bar:
        images = images.to(device)
        labels = labels.to(device)        
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Compute metrics for this batch
        metrics = compute_metrics(outputs, labels, num_classes)
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{metrics["dice"]:.4f}',
            'IoU': f'{metrics["iou"]:.4f}',
            'HD95': f'{metrics["hd95"]:.4f}'
        })
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics['loss'] = total_loss / len(train_loader)
    
    return avg_metrics


def validate_epoch(model, val_loader, criterion, device, num_classes=2):
    model.eval()
    total_loss = 0
    all_metrics = {'dice': [], 'iou': [], 'hd95': []}
    
    with torch.no_grad():
        for batch_idx, (images, labels, domain_labels) in enumerate(tqdm(val_loader, desc="Validation")):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Compute metrics for this batch
            metrics = compute_metrics(outputs, labels, num_classes)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics['loss'] = total_loss / len(val_loader)
    
    return avg_metrics


def main(args):

    print(f"Using device: {args.device}")
    
    source_domains = args.source_domains
    print(f"Training on source domains: {source_domains}")
    
    # load dataloader
    train_loader, val_loader = TTA_dataloader(args, source_domains)
    
    # initialize model
    model_module = importlib.import_module(f"networks.{args.model}")
    Model = getattr(model_module, args.model)
    model = Model(num_classes=args.out_ch).to(args.device)
    
    # load pretrained encoder weights
    if args.pretrain and os.path.exists(args.pretrained_encoder_path):
        print(f"Loading pretrained encoder from {args.pretrained_encoder_path}")
        checkpoint = torch.load(args.pretrained_encoder_path, map_location=args.device, weights_only=True)
        # Extract encoder state dict from the pretrained checkpoint
        encoder_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('encoder.'):
                encoder_state_dict[key[8:]] = value

        model.encoder.load_state_dict(encoder_state_dict, strict=False)
        print("Pretrained encoder loaded successfully")
    
    # set loss function
    criterion = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)

    # set optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
    
    save_dir = os.path.join(args.model_root, args.experiment_name, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):        
        # train
        train_metrics = train_epoch(model, epoch, args.epochs, train_loader, optimizer, criterion, args.device, args.out_ch)
        
        # test
        val_metrics = validate_epoch(model, val_loader, criterion, args.device, args.out_ch)
        
        # save best model based on validation loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']

            best_model_filename = f"source_{'_'.join(source_domains)}.pth"
            best_model_path = os.path.join(save_dir, best_model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_metrics['loss'],
            }, best_model_path)
            print(f"Best model saved at {best_model_path}")
        
        # log to wandb
        wandb.log({
            "Train Loss": train_metrics['loss'],
            "Train Dice": train_metrics['dice'],
            "Train IoU": train_metrics['iou'],
            "Train HD95": train_metrics['hd95'],
            "Val Loss": val_metrics['loss'],
            "Val Dice": val_metrics['dice'],
            "Val IoU": val_metrics['iou'],
            "Val HD95": val_metrics['hd95'],
        })

        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Train Dice: {train_metrics['dice']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        print(f"Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
        print(f"Train HD95: {train_metrics['hd95']:.4f}, Val HD95: {val_metrics['hd95']:.4f}")
    
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument('--source_domains', type=str, nargs='+', default=['ADAM'], help='Space-separated list of source domains')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=[256, 256, 128])
    parser.add_argument('--shuffle', action='store_false', default=True)
    parser.add_argument('--augmentation', action='store_true', default=False)
    parser.add_argument('--aug_times', type=int, default=32)

    # Model
    parser.add_argument('--model', type=str, default='ResUnet3d') # SwinUnet3d ResUnet3d
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=2)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--pretrained_encoder_path', type=str, 
                       default=DEFAULT_PRETRAINED_ENCODER,
                       help='Path to pretrained autoencoder')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)

    # Paths
    parser.add_argument('--model_root', type=str, default='models')
    parser.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--experiment_name', type=str, default='source_train')

    # Cuda
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    
    # Start a new wandb run to track this script.
    run = wandb.init(
        project="dyco-cta-source",
        name=f"{'_'.join(args.source_domains)}_new",
        config={
            "learning_rate": args.lr,
            "dataset": "vessel_segmentation",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer
        },
        mode=os.environ.get("WANDB_MODE", "offline")
    )

    main(args)
