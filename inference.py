import os
import argparse
import importlib
from collections import defaultdict

import torch
import numpy as np

from tqdm import tqdm

from dataloaders.TTA_dataloader import TTA_dataloader

from utils.metrics import compute_metrics

torch.set_num_threads(1)
DEFAULT_DATASET_ROOT = os.environ.get("DYCO_CTA_DATASET_ROOT", "data/TTA_dataset")


class Inference:
    def __init__(self, args):
        self.args = args
        self.image_size = args.image_size
        
        # Model
        self.in_ch = args.in_ch
        self.out_ch = args.out_ch

        # GPU
        self.device = args.device

        # dataest
        self.source_domains = args.source_domains
        self.target_domains = args.target_domains

        # Initialize the pre-trained model
        self.build_model()

    def build_model(self):
        source_model_path = os.path.join(
            self.args.model_root,
            self.args.model,
            f"source_{'_'.join(self.args.source_domains)}.pth",
        )
        
        model_module = importlib.import_module(f"networks.{self.args.model}")
        Model = getattr(model_module, self.args.model)
        self.model = Model(num_classes=self.args.out_ch).to(self.args.device)

        if os.path.exists(source_model_path):
            print(f"Loading pre-trained model from {source_model_path}")
            checkpoint = torch.load(source_model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
            print("Pre-trained model loaded successfully")
        else:
            print(f"Pre-trained model not found at {source_model_path}")
            print("Initializing with random weights")

        self.model.eval()

    @torch.no_grad()
    def run(self):
        print("Inference...")
        
        # load dataloader
        train_loader, val_loader = TTA_dataloader(self.args, self.target_domains)
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[Inference]")
        
        all_metrics = {'dice': [], 'iou': [], 'hd95': []}
        domain_metrics = defaultdict(lambda: {'dice': [], 'iou': [], 'hd95': []})
        
        for batch_idx, (images, labels, domain_labels) in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            
            metrics = compute_metrics(outputs, labels, self.out_ch)
            
            progress_bar.set_postfix({
                'Dice': f'{metrics["dice"]:.4f}',
                'IoU': f'{metrics["iou"]:.4f}',
                'HD95': f'{metrics["hd95"]:.4f}'
            })

            for key in all_metrics:
                all_metrics[key].append(metrics[key])
                for domain in domain_labels:
                    domain_metrics[domain][key].append(metrics[key])

        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

        print(f"Dice: {avg_metrics['dice']:.4f}")
        print(f"IoU: {avg_metrics['iou']:.4f}")
        print(f"HD95: {avg_metrics['hd95']:.4f}")
        for domain, values in domain_metrics.items():
            averages = {key: np.mean(metric_values) for key, metric_values in values.items()}
            print(
                f"{domain}: Dice={averages['dice']:.4f} "
                f"IoU={averages['iou']:.4f} HD95={averages['hd95']:.4f}"
            )

        print("Inference finished")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--source_domains', type=str, nargs='+', default=['ADAM'],
                       help='Space-separated list of source domains')
    parser.add_argument('--target_domains', type=str, nargs='+', default=['IXI-HH', 'IXI-Guys', 'IXI-IOP', 'LocH1', 'ICBM'], 
                       help='Space-separated list of target domains')
    
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256, 128])
    parser.add_argument('--shuffle', action='store_false', default=False)
    parser.add_argument('--augmentation', action='store_true', default=False)
    parser.add_argument('--aug_times', type=int, default=32)

    # Model
    parser.add_argument('--model', type=str, default='ResUnet3d') # SwinUnet3d ResUnet3d
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=2)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)

    # Paths
    parser.add_argument('--model_root', type=str, default='models/source_train')
    parser.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT)

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    inference = Inference(args)

    inference.run()
