import os
import copy
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import monai.transforms as transforms

DEFAULT_DATASET_ROOT = os.environ.get("DYCO_CTA_DATASET_ROOT", "data/TTA_dataset")


class TTADataset(Dataset):
    def __init__(self, args, domain_list, data_type='imagesTr', transform=None):
        """
        domain_list: for example['ADAM', 'ICBM', 'IXI-Guys', 'IXI-HH']
        data_type: 'imagesTr' or 'imagesTs'
        """
        self.args = args
        self.domain_list = domain_list
        self.data_type = data_type
        self.transform = transform
        
        self.data_root = os.path.join(args.dataset_root, data_type)
        self.labels_root = os.path.join(args.dataset_root, 'labelsTr' if data_type == 'imagesTr' else 'labelsTs')
        
        self.image_paths = []
        self.label_paths = []
        self.domain_labels = []
        
        for domain in domain_list:
            domain_path = os.path.join(self.data_root, domain)
            labels_domain_path = os.path.join(self.labels_root, domain)
            if os.path.exists(domain_path):
                for file_name in sorted(os.listdir(domain_path)):
                    if file_name.endswith('.nii.gz') or file_name.endswith('.nii'):
                        self.image_paths.append(os.path.join(domain_path, file_name))
                        self.domain_labels.append(domain)
                        label_path = os.path.join(labels_domain_path, file_name)

                        assert os.path.exists(label_path), f"Label file {label_path} does not exist"
                        self.label_paths.append(label_path)
                            
        
        print(f"Found {len(self.image_paths)} images in domains: {domain_list}")

        valid_labels = sum(1 for path in self.label_paths if path is not None)
        print(f"Found {valid_labels} corresponding labels")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        domain_label = self.domain_labels[idx]
        label_path = self.label_paths[idx]
        
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata()
        if len(img_data.shape) == 4:
            img_data = img_data[:, :, :, 0]

        label_nii = nib.load(label_path)
        label_data = np.round(label_nii.get_fdata())

        if len(label_data.shape) == 4:
            label_data = label_data[:, :, :, 0]
    
        # ensure 4d (C, H, W, D)
        if len(img_data.shape) == 3:
            img_data = np.expand_dims(img_data, axis=0)
        if len(label_data.shape) == 3:
            label_data = np.expand_dims(label_data, axis=0)    

        if self.transform is not None:
            data_dict = {"image": img_data,
                "label": label_data,
                "domain_label": domain_label}
            transformed_data = self.transform(data_dict)
            return transformed_data["image"].float(), transformed_data["label"].float(), domain_label
        else:
            return torch.from_numpy(img_data).float(), torch.from_numpy(label_data).float(), domain_label


def TTA_dataloader(args, domain_list, augmentation=False):
    print("=> creating TTA dataloader")

    if not hasattr(args, 'dataset_root'):
        args.dataset_root = DEFAULT_DATASET_ROOT

    transform = get_transforms(args, augmentation)

    # source domain
    train_dataset = TTADataset(args, domain_list, data_type='imagesTr', transform=transform)
    
    # target domain
    val_dataset = TTADataset(args, domain_list, data_type='imagesTs', transform=transform)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print("=> finish creating TTA dataloader")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}\n")

    return train_dataloader, val_dataloader


class MinMaxNormalizationd(transforms.MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            # Handle negative values
            mask = (img != 0)
            min_val = img.min()
            img = torch.where(mask, img + np.abs(min_val), img)
            # min-max normalization
            img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-6)
            d[key] = img_normalized
        return d


class MultiAugmentationd(transforms.MapTransform):
    """
    repeat transforms {times} times
    """
    def __init__(self, keys, transform, times=32):
        super().__init__(keys)
        self.transform = transform
        self.times = times - 1

    def __call__(self, data):
        aug_data_list = [data]
        
        for _ in range(self.times):
            item_data = copy.deepcopy(data)
            aug_data = self.transform(item_data)
            aug_data_list.append(aug_data)

        output_dict = {}
        for key in data.keys():
            if key in self.keys:
                tensors = [d[key] for d in aug_data_list]
                if isinstance(tensors[0], torch.Tensor):
                    output_dict[key] = torch.stack(tensors, dim=0)
                else:
                    output_dict[key] = np.stack(tensors, axis=0)
            else:
                output_dict[key] = data[key]
            
        return output_dict
    

def get_transforms(args, augmentation=False):
    pre_transform = [
            transforms.ToTensord(keys=["image", "label"]),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                select_fn=lambda x: x != 0),
            transforms.Resized(
                keys=["image", "label"],
                spatial_size=args.image_size, 
                mode=("bilinear", "nearest")),
            # transforms.SpatialPadd(
            #     keys=["image", "label"],
            #     spatial_size=(144, 192, 160), 
            #     method='end'),
            # transforms.CenterSpatialCropd(
            #     keys=["image", "label"],
            #     roi_size=tuple(args.image_size)),
        ]
    
    if augmentation:
        single_aug_pipeline = transforms.Compose([
            MinMaxNormalizationd(keys=["image"]),
            transforms.RandAdjustContrastd(keys=["image"], gamma=(0.85, 1.15), prob=0.5),
            transforms.RandHistogramShiftd(keys=["image"], num_control_points=8, prob=0.3),
            # transforms.RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(np.pi / 18, np.pi / 18), translate_range=(8, 8), 
            #                     scale_range=(0.05, 0.05), mode=["bilinear", "nearest"], padding_mode="border"),
            # transforms.RandFlipd(keys=["image", "label"], spatial_axis=(0, 1, 2), prob=0.5),
            transforms.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # transforms.RandGaussianNoised(keys=["image"], std=0.01, prob=0.4),
        ])

        transform = transforms.Compose(pre_transform + [
            MultiAugmentationd(keys=["image", "label"], transform=single_aug_pipeline, times=args.aug_times),
        ])
        
    else:
        transform = transforms.Compose(pre_transform + [
            transforms.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ])
    
    return transform
