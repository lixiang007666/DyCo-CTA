# DyCo-CTA

DyCo-CTA is a dynamic collaborative continual test-time adaptation framework for 3D vessel segmentation.

## Overview

Morphological characteristics of vessels are important biomarkers for many diseases, which makes accurate vessel segmentation a key step in computer-aided diagnosis. In real deployment, however, cross-center domain shifts often degrade segmentation performance. Continual Test-Time Adaptation (CTA) addresses this by updating the model online with unlabeled target-domain test data, but standard CTA can easily introduce topology-breaking artifacts such as vessel disconnections and spurious branches.

This project implements **DyCo-CTA**, which extends a source-trained 3D segmentation model with:

- **Dynamic collaborative adaptation**: two subnetworks are initialized from the same source model and dynamically swap teacher-student roles according to prediction entropy on the original volume.
- **Pseudo-break transformation**: a structure-aware perturbation creates complementary inputs that simulate local vessel interruption and encourage structural integrity.
- **Topological regularization**: a topology consistency loss aligns critical topological points beyond voxel-level supervision.


## Dynamic Collaborative Mechanism

At each adaptation step, the framework builds an original input volume `x` and a pseudo-break view `x_pb`.

Both subnetworks `M1` and `M2` process both inputs:

- `Y^(1), Y_pb^(1) = M1(x), M1(x_pb)`
- `Y^(2), Y_pb^(2) = M2(x), M2(x_pb)`

Teacher-student roles are assigned dynamically by comparing the volume-averaged entropy on the original volume:

- lower entropy -> higher confidence
- the lower-entropy model becomes the teacher for the current step
- the higher-entropy model becomes the student for the current step

The detached teacher prediction on `x_pb` supervises the student prediction on `x_pb`, and only the current student is updated. The student is also optimized with an entropy minimization term on the original volume. Optional topology regularization is applied between teacher and student foreground probabilities on the pseudo-break view.

## Environment

Recommended Python version: `3.9+`

Core dependencies used by the current implementation:

```bash
pip install torch torchvision monai nibabel numpy tqdm wandb
pip install cripser gudhi ripser connected-components-3d opencv-python scipy SimpleITK
```


## Data and Paths

The code now avoids hard-coded personal paths.

Default dataset root:

```bash
data/TTA_dataset
```

You can override paths with environment variables:

```bash
set DYCO_CTA_DATASET_ROOT=your_dataset_root
set DYCO_CTA_PRETRAINED_ENCODER=your_pretrained_encoder_path
set WANDB_MODE=offline
```

On PowerShell:

```powershell
$env:DYCO_CTA_DATASET_ROOT="path\\to\\TTA_dataset"
$env:DYCO_CTA_PRETRAINED_ENCODER="path\\to\\resnet3d_50.pth"
$env:WANDB_MODE="offline"
```

Expected dataset structure for `main/dataloaders/TTA_dataloader.py`:

```text
TTA_dataset/
  imagesTr/
    DOMAIN_A/
    DOMAIN_B/
  labelsTr/
    DOMAIN_A/
    DOMAIN_B/
  imagesTs/
    DOMAIN_A/
    DOMAIN_B/
  labelsTs/
    DOMAIN_A/
    DOMAIN_B/
```

## Usage

### 1. Train the source model

```bash
python train_source.py --source_domains ADAM --model ResUnet3d
```

This saves the source checkpoint under:

```text
models/source_train/<ModelName>/source_<domains>.pth
```

### 2. Run DyCo-CTA adaptation

```bash
python dyco_cta.py --source_domains ADAM --target_domains IXI-HH IXI-Guys ICBM --model ResUnet3d
```

Useful arguments:

- `--consistency_weight`: weight of the teacher-relation loss on the pseudo-break view
- `--entropy_weight`: weight of the student entropy minimization loss
- `--topology_weight`: weight of the topology regularization term
- `--break_regions`: number of pseudo-break regions
- `--break_roi_size`: pseudo-break region size
- `--confidence_threshold`: confidence mask threshold for teacher supervision

### 3. Run source-model inference

```bash
python inference.py --source_domains ADAM --target_domains IXI-HH ICBM --model ResUnet3d
```

## Notes

- The current DyCo-CTA implementation is centered in [main/dyco_cta.py](C:\Users\Admin\Desktop\DyCo-CTA\main\dyco_cta.py).
- Topology regularization is optional. If `cripser` or `gudhi` is unavailable, the script will skip that term and print a warning.
- `wandb` defaults to offline mode unless you explicitly change `WANDB_MODE`.

## Status

The repository currently includes:

- pseudo-break view generation
- topology consistency loss
- dynamic teacher-student role switching based on entropy
