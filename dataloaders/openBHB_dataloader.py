import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import monai.transforms as transforms
from nilearn.masking import unmask


class OpenBHB_Dataloader(DataLoader):
    def __init__(self, dataset, args, normalization_type='min-max', **kwargs):
        super().__init__(dataset, **kwargs)
        self.args = args
        self.normalization_type = normalization_type
        self.template_path = "/sharedData/datasets/quasiraw_space-MNI152_desc-brain_T1w.nii.gz"
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.CropForeground(
                select_fn=lambda x: x != 0,
                margin=0),
            transforms.SpatialPad(
                spatial_size=(144, 192, 160), 
                method='end'),
            transforms.CenterSpatialCrop(
                roi_size=tuple(args.image_size)),
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
            transforms.ToTensor()
        ])

    def preproc_openBHB(self, ndarray):
        img = unmask(ndarray, self.template_path).get_fdata().transpose((3, 0, 1, 2))
        img = self.transform(img)[:, np.newaxis, ...]

        if self.normalization_type == 'min-max':
            # min-max normalization
            img_reshaped = img.reshape(img.shape[0], -1)
            img_min = img_reshaped.min(dim=1, keepdim=True)[0].reshape(img.shape[0], 1, 1, 1, 1)
            img_max = img_reshaped.max(dim=1, keepdim=True)[0].reshape(img.shape[0], 1, 1, 1, 1)
            img_normalized = (img - img_min) / (img_max - img_min + 1e-8)
        else:
            img_normalized = img

        return img_normalized.to(torch.float32)

    def __iter__(self):
        for batch in super().__iter__():
            data, labels = batch
            # Apply preprocessing to the data
            processed_data = self.preproc_openBHB(data.numpy())
            yield processed_data, labels


def OpenBHB(args, index=1000):
    print("=> creating dataloader")

    data_path = '/sharedData/datasets/OpenBHB'

    # participants file
    tsv_path_train = os.path.join(data_path, 'train.tsv')
    tsv_path_val = os.path.join(data_path, 'test.tsv')
    df_train = pd.read_csv(tsv_path_train, sep='\t')
    df_val = pd.read_csv(tsv_path_val, sep='\t')

    age_list_train = df_train['age'].tolist()
    age_list_val = df_val['age'].tolist()

    label_train = torch.tensor(age_list_train).to(torch.float32)
    label_val = torch.tensor(age_list_val).to(torch.float32)

    img_train = np.load(os.path.join(data_path, 'train.npy'), mmap_mode='r')
    img_val =  np.load(os.path.join(data_path, 'test.npy'), mmap_mode='r')

    img_train_tensor = torch.from_numpy(img_train[:index, 519945:2347040].copy())
    img_val_tensor = torch.from_numpy(img_val[:int(index/10), 519945:2347040].copy())

    # dataset
    train_dataset = TensorDataset(img_train_tensor, label_train[:index])
    val_dataset = TensorDataset(img_val_tensor, label_val[:int(index/10)])
    del img_train_tensor, img_val_tensor

    # dataloader
    train_dataloader = OpenBHB_Dataloader(train_dataset, 
                                          args,
                                          batch_size=args.batch_size, 
                                          shuffle=True, 
                                          num_workers=args.num_workers)
    val_dataloader = OpenBHB_Dataloader(val_dataset, 
                                        args,
                                        batch_size=args.val_batch_size, 
                                        shuffle=False, 
                                        num_workers=args.num_workers)
    
    print("=> finish creating dataloader")

    return train_dataloader, val_dataloader
