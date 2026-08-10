import os
import argparse
import importlib
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from dataloaders.TTA_dataloader import TTA_dataloader
from utils.metrics import set_seed, compute_metrics

torch.set_num_threads(1)
DEFAULT_DATASET_ROOT = os.environ.get("DYCO_CTA_DATASET_ROOT", "data/TTA_dataset")

class CoTTA3D:
    def __init__(self, args):
        self.args = args
        self.image_size = args.image_size
        
        # Dimensions
        self.in_ch = args.in_ch
        self.out_ch = args.out_ch
        self.device = args.device

        # Dataset & Domains
        self.source_domains = args.source_domains
        self.target_domains = args.target_domains

        # CoTTA Hyperparameters
        self.restoration_factor = args.restoration_factor  # e.g., 0.01
        self.ema_factor = args.ema_factor                  # e.g., 0.999
        self.consistency_weight = args.consistency_weight
        
        # Initialize Models
        self.build_model()
        
        # Setup CoTTA specific models (Teacher & Anchor)
        self.setup_cotta_models()

    def build_model(self):
        source_model_path = os.path.join(args.model_root, args.model, f"source_{'_'.join(args.source_domains)}.pth")
        
        model_module = importlib.import_module(f"networks.{args.model}")
        Model = getattr(model_module, args.model)
        self.model = Model(num_classes=args.out_ch).to(args.device)

        if os.path.exists(source_model_path):
            print(f"Loading pre-trained model from {source_model_path}")
            checkpoint = torch.load(source_model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
            print("Pre-trained model loaded successfully")
        else:
            print(f"Pre-trained model not found at {source_model_path}")
            print("Initializing with random weights")

            raise FileNotFoundError(f"Source model not found at {source_model_path}")

    def setup_cotta_models(self):
        """
        CoTTA requires 3 models:
        1. Student (self.model): Adapts to target domain.
        2. Teacher (self.model_ema): Provides stable pseudo-labels.
        3. Anchor (self.model_anchor): Frozen source model for restoration.
        """
        # Copy model for Anchor and Teacher
        self.model_anchor = copy.deepcopy(self.model)
        self.model_ema = copy.deepcopy(self.model)

        # Freeze Anchor and Teacher completely
        for param in self.model_anchor.parameters():
            param.detach_()
            param.requires_grad = False

        for param in self.model_ema.parameters():
            param.detach_()
            param.requires_grad = False
            
        # Configure Optimizer for Student
        self.configure_optimizer()

    def configure_optimizer(self):
        """Collect parameters for optimization."""
        # CoTTA usually updates all parameters or specific layers (e.g. Norm layers).
        # Here we update all requires_grad parameters as per original logic.
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.args.lr,
                momentum=self.args.momentum,
            )
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.args.lr,
                betas=(self.args.beta1, self.args.beta2)
            )

    def transform_input(self, x):
        """
        Apply simple 3D augmentation for the Student model.
        CoTTA relies on Student seeing augmented views while Teacher sees clean views.
        """
        # Example: Gaussian Noise + Random Flip
        # Assuming input shape [B, C, D, H, W]
        aug_x = x.clone()
        
        # 1. Gaussian Noise
        noise = torch.randn_like(aug_x) * 0.05
        aug_x = aug_x + noise

        # 2. Random Flip (e.g., horizontal)
        if random.random() > 0.5:
            aug_x = torch.flip(aug_x, dims=[-1])
            
        return aug_x

    def update_ema_variables(self):
        """Update Teacher model with EMA of Student weights."""
        alpha = self.ema_factor
        for ema_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def stochastic_restore(self):
        """
        Stochastically restore Student weights to Anchor weights.
        This prevents catastrophic forgetting of the source domain.
        """
        factor = self.restoration_factor
        for nm, m in self.model.named_modules():
            # Identify convolution or linear layers (parameters to restore)
            if isinstance(m, (nn.Conv3d, nn.Linear, nn.Conv2d)): 
                for param_name, param in m.named_parameters():
                    # Get corresponding anchor param
                    anchor_param = dict(self.model_anchor.named_parameters())[f"{nm}.{param_name}"]
                    
                    # Create a random mask
                    mask = (torch.rand_like(param) < factor).float()
                    
                    # Restore weights: param = mask * anchor + (1-mask) * param
                    param.data.copy_(mask * anchor_param.data + (1 - mask) * param.data)

    def run(self):
        # Load dataloaders
        # Note: Warmup loader is not typically used in standard Online TTA, 
        # but kept if you need source statistics (not used in this logic).
        _, _ = TTA_dataloader(self.args, self.source_domains)
        train_loader, val_loader = TTA_dataloader(self.args, self.target_domains)
        
        print("Starting CoTTA Process (Fixing Overlap)...")
        
        # 【修改 1】降低阈值，由 0.9 -> 0.68 (CoTTA 论文常用值)
        # 0.9 太高了，会导致初期没有像素参与反向传播
        confidence_threshold = 0.68 
        
        accumulation_steps = 4
        min_pixel_threshold = 50

        self.model.train()
        self.model_ema.eval()
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[CoTTA]")
        
        # 初始化记录字典
        all_metrics = {'dice': [], 'iou': [], 'hd95': []}
        
        self.optimizer.zero_grad() 

        for batch_idx, (images, labels, domain_labels) in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # ... (Teacher 预测部分不变) ...
            with torch.no_grad():
                teacher_outputs = self.model_ema(images)
                teacher_prob = F.softmax(teacher_outputs, dim=1)
                max_prob, teacher_preds = torch.max(teacher_prob, dim=1)
                
                foreground_pixels = torch.sum(teacher_preds > 0).item()
                skip_update = foreground_pixels < min_pixel_threshold

            # ... (Student 训练部分不变) ...
            if not skip_update:
                images_aug = self.transform_input(images)
                student_outputs = self.model(images_aug)
                student_log_prob = F.log_softmax(student_outputs, dim=1)

                # 使用新的阈值 0.68
                mask = (max_prob > confidence_threshold).float()
                
                loss_pixel = -torch.sum(teacher_prob * student_log_prob, dim=1)
                # 防止分母为 0 的保护更强一点
                loss = torch.sum(loss_pixel * mask) / (torch.sum(mask) + 1e-6)
                
                loss = loss / accumulation_steps
                loss.backward()
                loss_val = loss.item() * accumulation_steps
            else:
                loss_val = 0.0

            # ... (优化器更新部分不变) ...
            if (batch_idx + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.update_ema_variables()
                self.stochastic_restore()

            # --- 【修改 2】关键修改：分别评估 Student 和 Teacher ---
            with torch.no_grad():
                # 1. 评估 Student (查看当前适应情况，这个变化会很剧烈)
                # 注意：Student 训练时用了 Aug，评估时用原始图像
                student_eval = self.model(images)
                metrics_student = compute_metrics(student_eval, labels, self.out_ch)

                # 2. 评估 Teacher (查看最终稳定输出，这个变化很慢)
                # eval_outputs = self.model_ema(images) 
                # metrics_teacher = compute_metrics(eval_outputs, labels, self.out_ch)

            # 在进度条显示 Student 的结果，看它是否在变
            status_str = f'Stu_Dice:{metrics_student["dice"]:.3f}'
            if skip_update: status_str += " [SKIP]"
            progress_bar.set_postfix_str(status_str)

            for key in all_metrics:
                all_metrics[key].append(metrics_student[key])

        # Final Summary
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        print("\n=== CoTTA Adaptation Finished ===")
        print(f"Avg Dice: {avg_metrics['dice']:.4f}")
        print(f"Avg IoU:  {avg_metrics['iou']:.4f}")
        print(f"Avg HD95: {avg_metrics['hd95']:.4f}")


if __name__ == '__main__':
    set_seed()

    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument('--source_domains', type=str, nargs='+', default=['ADAM'], 
                       help='Space-separated list of source domains')
    parser.add_argument('--target_domains', type=str, nargs='+', default=['IXI-HH', 'IXI-Guys', 'IXI-IOP', 'LocH1', 'ICBM'], 
                       help='Space-separated list of target domains')
    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256, 128])
    parser.add_argument('--shuffle', action='store_true', default=False)

    # Model
    parser.add_argument('--model', type=str, default='ResUnet3d')
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=2)
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=1e-5) # Adam usually needs lower LR
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # CoTTA specific parameters (Standard Paper Values)
    parser.add_argument('--ema_factor', type=float, default=0.999, 
                       help='Alpha for EMA update')
    parser.add_argument('--restoration_factor', type=float, default=0.01, 
                       help='Probability of restoring weights to source')
    parser.add_argument('--consistency_weight', type=float, default=1.0)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    
    # Paths
    parser.add_argument('--model_root', type=str, default='models/source_train')
    parser.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT)

    # Cuda
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    runner = CoTTA3D(args)
    runner.run()
