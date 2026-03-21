import os
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import monai.transforms as transforms


class MMsDataset(Dataset):
    def __init__(self, args, domain_list, data_type='imagesTr', transform=None, ED=True):
        """
        domain_list: for example['GE', 'Philips', 'Siemens', 'Canon']
        data_type: 'imagesTr' or 'imagesTs'
        """
        self.args = args
        self.domain_list = domain_list
        self.data_type = data_type
        self.transform = transform
        self.ED = ED

        self.csv = pd.read_csv(os.path.join(args.dataset_root, 'participants.csv'))

        
        self.data_root = os.path.join(args.dataset_root, data_type)
        self.labels_root = os.path.join(args.dataset_root, 'labelsTr' if data_type == 'imagesTr' else 'labelsTs')
        
        self.image_paths = []
        self.label_paths = []
        self.domain_labels = []
        self.ED_list = []
        self.ES_list = []
        
        for domain in domain_list:
            domain_path = os.path.join(self.data_root, domain)
            labels_domain_path = os.path.join(self.labels_root, domain)
            if os.path.exists(domain_path):
                for file_name in os.listdir(domain_path):
                    if file_name.endswith('.nii.gz') or file_name.endswith('.nii'):
                        self.image_paths.append(os.path.join(domain_path, file_name))
                        self.domain_labels.append(domain)
                        self.ED_list.append(self.csv[self.csv['External code'] == file_name[:6]]['ED'].values[0])
                        self.ES_list.append(self.csv[self.csv['External code'] == file_name[:6]]['ES'].values[0])

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
            # img_data = img_data.permute(3, 0, 1, 2)

        label_nii = nib.load(label_path)
        label_data = np.round(label_nii.get_fdata())
        if len(label_data.shape) == 4:
            if self.ED:
                label_data = label_data[:, :, :, self.ED_list[idx]]
            else:
                label_data = label_data[:, :, :, self.ES_list[idx]]
            # label_data = label_data.permute(3, 0, 1, 2)
            # label_data = (label_data > 0).astype(int)

        # ensure 4d (C, H, W, D)
        img_data = np.expand_dims(img_data, axis=0)
        label_data = np.expand_dims(label_data, axis=0)    

        if self.transform is not None:
            data_dict = {"image": img_data,
                "label": label_data,
                "domain_label": domain_label}
            transformed_data = self.transform(data_dict)
            return transformed_data["image"], transformed_data["label"], domain_label
        else:
            return torch.from_numpy(img_data).float(), torch.from_numpy(label_data).float(), domain_label


def MMs_dataloader(args, domain_list, ED=True):
    print("=> creating MMs dataloader")

    transform = transforms.Compose([
        transforms.CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            select_fn=lambda x: x!=0,
            margin=0),
        transforms.Resized(
            keys=["image", "label"],
            spatial_size=[256, 256, 12], 
            mode=["trilinear", "nearest"]),
        transforms.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        transforms.ToTensord(keys=["image", "label"])
    ])

    # source domain
    train_dataset = MMsDataset(args, domain_list, data_type='imagesTr', transform=transform, ED=ED)
    
    # target domain
    val_dataset = MMsDataset(args, domain_list, data_type='imagesTs', transform=transform, ED=ED)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    
    print("=> finish creating MMs dataloader")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    return train_dataloader, val_dataloader