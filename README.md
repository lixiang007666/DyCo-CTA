# DyCo-CTA

> 🎉 **MICCAI 2026 Oral Paper**

Official implementation of **Dynamic Collaborative Continual Test-Time Adaptation for 3D Vessel Segmentation**.

DyCo-CTA adapts a source-trained 3D vessel segmentation model to a non-stationary stream of unlabeled target volumes. It combines three components:

- dynamic teacher-student role assignment using volume-averaged prediction entropy;
- pseudo-break transformation using vessel removal, inpainting, and local Gaussian blending;
- persistence-diagram regularization for matching stable structures and removing topological noise.

## Installation

The experiments use Python 3.10, PyTorch 2.4.0, and CUDA 12.4. A reproducible Conda specification is provided in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate dyco-cta
```

## Data

Datasets and model weights are not distributed in this repository. Arrange preprocessed volumes as follows:

```text
data/TTA_dataset/
├── imagesTr/<domain>/*.nii.gz
├── labelsTr/<domain>/*.nii.gz
├── imagesTs/<domain>/*.nii.gz
└── labelsTs/<domain>/*.nii.gz
```

The code recognizes the domains used in the paper, including `IXI-HH`, `IXI-Guys`, `IXI-IOP`, `LocH1`, `ICBM`, and `ADAM`. Set a different dataset location with `--dataset_root` or `DYCO_CTA_DATASET_ROOT`.

Source checkpoints follow this naming convention:

```text
models/source_train/ResUnet3d/source_<domain>.pth
```

Each checkpoint must contain a `model_state_dict` entry.

## Source training

```bash
python train_source.py \
  --source_domains ADAM \
  --model ResUnet3d \
  --lr 0.01
```

## Continual test-time adaptation

```bash
python dyco_cta.py \
  --source_domains ADAM \
  --target_domains IXI-HH IXI-Guys IXI-IOP LocH1 ICBM \
  --model ResUnet3d
```

The default command uses the stable long-stream configuration for the released ADAM source checkpoint: batch size 1, four updates per test volume, learning rate `1e-5`, entropy weight `0.1`, topology weight `0.01`, stochastic restoration factor `0.1`, pseudo-label threshold `0.5`, three mask-dilation iterations, `15 x 15` pseudo-break patches, and persistence threshold `0.7`.

To run the adaptation settings stated in the paper, pass them explicitly:

```bash
python dyco_cta.py \
  --source_domains ADAM \
  --target_domains IXI-HH IXI-Guys IXI-IOP LocH1 ICBM \
  --lr 1e-4 \
  --entropy_weight 1.0 \
  --topology_weight 0.05 \
  --restoration_factor 0.01
```

`--source_blend` optionally mixes frozen source logits into the final prediction. `--volume_ratio_min` and `--volume_ratio_max` optionally enable a label-free fallback when the adapted foreground volume deviates substantially from the frozen source prediction. Both safeguards are disabled by default.

Use `inference.py` to evaluate a source model without adaptation.

## Acknowledgements

The dynamic collaboration design builds on [DiCo](https://github.com/lixiangcog/DiCo), and the pseudo-break transformation builds on [CoLeTra](https://github.com/lixiangcog/CoLeTra).

## Citation

```bibtex
@inproceedings{li2026dycocta,
  title={Dynamic Collaborative Continual Test-Time Adaptation for 3D Vessel Segmentation},
  author={Li, Xiang and Fang, Yuqi and Zhang, Dan and Zhang, Jiong and Shan, Caifeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2026}
}
```
