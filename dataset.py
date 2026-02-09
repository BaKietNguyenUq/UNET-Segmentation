import os
from glob import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision

class MRISegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        
        if len(self.img_paths) == 0 or len(self.mask_paths) == 0:
            raise RuntimeError("No images or masks found")

        assert len(self.img_paths) == len(self.mask_paths), \
            f"Number of images ({len(self.img_paths)}) and masks ({len(self.mask_paths)}) do not match"

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L")     # grayscale MRI
        mask = Image.open(self.mask_paths[idx]).convert("L")   # segmentation mask (labels)

        img = self.transform(img)
        mask = self.target_transform(mask)
        return img, mask

def get_transforms():
    # Transforms for MRI images
    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.1317], [0.1864])
    ])

    # Masks do not need normalization - just convert to tensor of class IDs
    mask_transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64))),
        torchvision.transforms.Lambda(lambda x: torch.where(x==85, torch.tensor(1),
                            torch.where(x==170, torch.tensor(2),
                            torch.where(x==255, torch.tensor(3), x))))
    ])
    return img_transform, mask_transform

def get_datasets(base_dir):
    train_dir = f"{base_dir}/keras_png_slices_train"
    train_dir_seg = f"{base_dir}/keras_png_slices_seg_train"
    test_dir  = f"{base_dir}/keras_png_slices_test"
    test_dir_seg  = f"{base_dir}/keras_png_slices_seg_test"
    
    img_transform, mask_transform = get_transforms()

    train_dataset = MRISegmentationDataset(train_dir, train_dir_seg, transform=img_transform, target_transform=mask_transform)
    test_dataset  = MRISegmentationDataset(test_dir,  test_dir_seg, transform=img_transform, target_transform=mask_transform)
    
    return train_dataset, test_dataset

def get_loaders(base_dir, batch_size=32):
    train_dataset, test_dataset = get_datasets(base_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    base_dir = "/mnt/c/Users/Acer/Downloads/keras_png_slices_data/keras_png_slices_data"
    batch_size = 32
    train_loader, test_loader = get_loaders(base_dir, batch_size)

    print("Train samples:", len(train_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Train loader:", len(train_loader))
    print("Test loader:", len(test_loader))
    
    imgs, masks = next(iter(train_loader))
    print("Image batch:", imgs.shape)   # (B,1,H,W)
    print("Mask batch:", masks.shape)   # (B,H,W)