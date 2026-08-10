import argparse
import copy
import importlib
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders.TTA_dataloader import TTA_dataloader
from utils.metrics import compute_metrics, set_seed
from utils.structural_perturbation import build_inpainted_view, build_pseudo_break_view
from utils.topology_regularization import TOPOLOGY_BACKEND_AVAILABLE, topology_consistency_loss

torch.set_num_threads(1)
DEFAULT_DATASET_ROOT = os.environ.get("DYCO_CTA_DATASET_ROOT", "data/TTA_dataset")


class DyCoCTA3D:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.build_models()
        self.configure_optimizers()

    def _source_model_path(self) -> str:
        return os.path.join(
            self.args.model_root,
            self.args.model,
            f"source_{'_'.join(self.args.source_domains)}.pth",
        )

    def _load_single_model(self):
        model_module = importlib.import_module(f"networks.{self.args.model}")
        model_cls = getattr(model_module, self.args.model)
        model = model_cls(num_classes=self.args.out_ch).to(self.device)

        source_model_path = self._source_model_path()
        if not os.path.exists(source_model_path):
            raise FileNotFoundError(f"Source model not found at {source_model_path}")

        checkpoint = torch.load(source_model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def build_models(self):
        self.model_primary = self._load_single_model()
        self.model_auxiliary = self._load_single_model()

        self.anchor_primary = copy.deepcopy(self.model_primary)
        self.anchor_auxiliary = copy.deepcopy(self.model_auxiliary)
        self.source_model = copy.deepcopy(self.model_primary).eval()

        for anchor in (self.anchor_primary, self.anchor_auxiliary, self.source_model):
            for param in anchor.parameters():
                param.requires_grad = False
                param.detach_()

    def configure_optimizers(self):
        params_primary = [p for p in self.model_primary.parameters() if p.requires_grad]
        params_auxiliary = [p for p in self.model_auxiliary.parameters() if p.requires_grad]

        if self.args.optimizer == "SGD":
            self.optimizer_primary = torch.optim.SGD(
                params_primary,
                lr=self.args.lr,
                momentum=self.args.momentum,
            )
            self.optimizer_auxiliary = torch.optim.SGD(
                params_auxiliary,
                lr=self.args.lr,
                momentum=self.args.momentum,
            )
        else:
            self.optimizer_primary = torch.optim.Adam(
                params_primary,
                lr=self.args.lr,
                betas=(self.args.beta1, self.args.beta2),
            )
            self.optimizer_auxiliary = torch.optim.Adam(
                params_auxiliary,
                lr=self.args.lr,
                betas=(self.args.beta1, self.args.beta2),
            )

    def _stochastic_restore(self, model: nn.Module, anchor: nn.Module):
        for model_param, anchor_param in zip(model.parameters(), anchor.parameters()):
            restore_mask = (torch.rand_like(model_param) < self.args.restoration_factor).to(model_param.dtype)
            model_param.data.copy_(restore_mask * anchor_param.data + (1.0 - restore_mask) * model_param.data)

    def _volume_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        voxel_entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1)
        return voxel_entropy.mean(dim=(1, 2, 3))

    def _entropy_objective(self, logits: torch.Tensor) -> torch.Tensor:
        return self._volume_entropy(logits).mean()

    def _teacher_relation_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        teacher_prob = F.softmax(teacher_logits.detach(), dim=1)
        student_log_prob = F.log_softmax(student_logits, dim=1)
        relation_map = -(teacher_prob * student_log_prob).sum(dim=1)

        if self.args.confidence_threshold <= 0:
            return relation_map.mean()

        teacher_confidence = teacher_prob.max(dim=1).values
        confidence_mask = (teacher_confidence > self.args.confidence_threshold).float()
        return (relation_map * confidence_mask).sum() / confidence_mask.sum().clamp_min(1.0)

    def _build_inpainted(self, images: torch.Tensor):
        with torch.no_grad():
            source_prob = F.softmax(self.source_model(images), dim=1)[:, 1:2]
            source_mask = (source_prob > self.args.break_mask_threshold).float()
        return build_inpainted_view(
            images, source_mask, dilation_iterations=self.args.inpaint_dilation
        )

    def run(self):
        train_loader, _ = TTA_dataloader(self.args, self.args.target_domains)

        self.model_primary.train()
        self.model_auxiliary.train()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="[DyCo-CTA]")
        all_metrics = {"dice": [], "iou": [], "hd95": []}

        for batch_idx, (images, labels, _) in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            inpainted = self._build_inpainted(images)

            for _ in range(self.args.adaptation_steps):
                logits_original_primary = self.model_primary(images)
                logits_original_auxiliary = self.model_auxiliary(images)
                entropy_primary = self._volume_entropy(logits_original_primary)
                entropy_auxiliary = self._volume_entropy(logits_original_auxiliary)
                primary_is_teacher = bool(entropy_primary.mean() <= entropy_auxiliary.mean())

                if primary_is_teacher:
                    teacher_model, student_model = self.model_primary, self.model_auxiliary
                    teacher_logits_original, student_logits_original = logits_original_primary, logits_original_auxiliary
                    student_optimizer, student_anchor, role_flag = self.optimizer_auxiliary, self.anchor_auxiliary, "M1->M2"
                else:
                    teacher_model, student_model = self.model_auxiliary, self.model_primary
                    teacher_logits_original, student_logits_original = logits_original_auxiliary, logits_original_primary
                    student_optimizer, student_anchor, role_flag = self.optimizer_primary, self.anchor_primary, "M2->M1"

                with torch.no_grad():
                    teacher_mask = (
                        F.softmax(teacher_logits_original, dim=1)[:, 1:2]
                        > self.args.break_mask_threshold
                    ).float()
                    break_view = build_pseudo_break_view(
                        images,
                        inpainted,
                        teacher_mask,
                        num_centers=self.args.break_regions,
                        patch_size=self.args.break_patch_size,
                        sigma=self.args.break_sigma,
                    )
                    teacher_logits_break = teacher_model(break_view)
                student_logits_break = student_model(break_view)

                teacher_relation_loss = self._teacher_relation_loss(teacher_logits_break, student_logits_break)
                student_entropy_loss = self._entropy_objective(student_logits_original)
                total_loss = self.args.entropy_weight * student_entropy_loss
                total_loss = total_loss + self.args.consistency_weight * teacher_relation_loss

                topo_loss_value = images.new_zeros(())
                if self.args.topology_weight > 0:
                    teacher_vessel_prob = F.softmax(teacher_logits_break, dim=1)[:, 1]
                    student_vessel_prob = F.softmax(student_logits_break, dim=1)[:, 1]
                    topo_loss_value = topology_consistency_loss(
                        student_vessel_prob,
                        teacher_vessel_prob,
                        patch_size=self.args.topology_patch_size,
                        persistence_threshold=self.args.topology_persistence,
                        max_slices=self.args.topology_max_slices,
                    )
                    total_loss = total_loss + self.args.topology_weight * topo_loss_value

                self.optimizer_primary.zero_grad(set_to_none=True)
                self.optimizer_auxiliary.zero_grad(set_to_none=True)
                total_loss.backward()
                student_optimizer.step()

                self._stochastic_restore(student_model, student_anchor)

            with torch.no_grad():
                final_primary = self.model_primary(images)
                final_auxiliary = self.model_auxiliary(images)
                selected_logits = final_primary if self._volume_entropy(final_primary).mean() <= self._volume_entropy(final_auxiliary).mean() else final_auxiliary
                metrics = compute_metrics(selected_logits, labels, self.args.out_ch)

            progress_bar.set_postfix_str(
                f"Dice:{metrics['dice']:.3f} Role:{role_flag} Ent1:{entropy_primary.mean().item():.4f} Ent2:{entropy_auxiliary.mean().item():.4f}"
            )

            for key in all_metrics:
                all_metrics[key].append(metrics[key])

        avg_metrics = {key: float(np.mean(values)) for key, values in all_metrics.items()}
        print("\n=== DyCo-CTA Adaptation Finished ===")
        print(f"Avg Dice: {avg_metrics['dice']:.4f}")
        print(f"Avg IoU:  {avg_metrics['iou']:.4f}")
        print(f"Avg HD95: {avg_metrics['hd95']:.4f}")
        if self.args.topology_weight > 0 and not TOPOLOGY_BACKEND_AVAILABLE:
            print("Topology regularization was requested but skipped because the topology backend is unavailable.")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_domains", type=str, nargs="+", default=["ADAM"])
    parser.add_argument("--target_domains", type=str, nargs="+", default=["IXI-HH", "IXI-Guys", "IXI-IOP", "LocH1", "ICBM"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--shuffle", action="store_true", default=False)

    parser.add_argument("--model", type=str, default="ResUnet3d")
    parser.add_argument("--in_ch", type=int, default=1)
    parser.add_argument("--out_ch", type=int, default=2)

    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)

    parser.add_argument("--model_root", type=str, default="models/source_train")
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--entropy_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=1.0)
    parser.add_argument("--topology_weight", type=float, default=0.05)
    parser.add_argument("--restoration_factor", type=float, default=0.01)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)

    parser.add_argument("--break_regions", type=int, default=6)
    parser.add_argument("--adaptation_steps", type=int, default=4)
    parser.add_argument("--break_patch_size", type=int, default=15)
    parser.add_argument("--break_sigma", type=float, default=3.0)
    parser.add_argument("--break_mask_threshold", type=float, default=0.5)
    parser.add_argument("--inpaint_dilation", type=int, default=3)

    parser.add_argument("--topology_patch_size", type=int, default=96)
    parser.add_argument("--topology_persistence", type=float, default=0.7)
    parser.add_argument("--topology_max_slices", type=int, default=4)
    return parser


if __name__ == "__main__":
    set_seed()
    parser = build_parser()
    args = parser.parse_args()

    DyCoCTA3D(args).run()
