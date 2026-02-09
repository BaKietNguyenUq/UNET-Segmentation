import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from modules import Unet
from dataset import get_loaders
from modules import config

def dice_coefficient(preds, target, num_classes, epsilon = 1e-7):
    # convert logits -> predicted class ids
    preds = preds.softmax(dim=1).argmax(dim=1)  # (B, H, W)

    dice_per_class = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_per_class.append(dice)

    dice_per_class = torch.stack(dice_per_class)
    mean_dice = dice_per_class.mean()

    return dice_per_class, mean_dice

def test_model(model: Unet, test_loader: DataLoader, device: str, num_classes: int) -> None:
    model.eval()
    all_dice = []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device).long()   

            logits = model(imgs)              
            dice_per_class, _ = dice_coefficient(logits, masks, num_classes=num_classes)

            all_dice.append(dice_per_class.cpu())

    # Stack results and average across test set
    all_dice = torch.cat(all_dice, dim=1)   
    mean_per_class = all_dice.mean(dim=1)

    print("\n=== Test Set Dice ===")
    for i, d in enumerate(mean_per_class):
        print(f"Class {i}: {d:.4f}")
    print(f"Mean Dice: {mean_per_class.mean():.4f}")
    
def visualize_results(model, test_loader, device):
    model.eval()
    imgs, masks = next(iter(test_loader))         
    imgs, masks = imgs.to(device), masks.to(device).long()

    with torch.no_grad():
        logits = model(imgs)                      
        preds = logits.softmax(1).argmax(1)       

    # convert preds {0,1,2,3} -> {0,85,170,255}
    mapping = torch.tensor([0,85,170,255], device=preds.device)
    preds_val = mapping[preds]                    

    mean, std = 0.1317, 0.1864

    # Show first 8 samples
    n = min(4, imgs.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))

    for i in range(n):
        img_np   = imgs[i][0].cpu().numpy()
        mask_np  = masks[i].cpu().numpy()
        pred_np  = preds_val[i].cpu().numpy()

        # Undo normalization for MRI
        img_np = img_np * std + mean
        img_np = img_np.clip(0,1)

        axes[i,0].imshow(img_np, cmap="gray")
        axes[i,0].set_title("MRI")
        axes[i,0].axis("off")

        axes[i,1].imshow(mask_np, cmap="gray")
        axes[i,1].set_title("Ground Truth")
        axes[i,1].axis("off")

        axes[i,2].imshow(pred_np, cmap="gray")
        axes[i,2].set_title("Prediction")
        axes[i,2].axis("off")

    plt.tight_layout()
    plt.show()
    
def result_model(model, test_loader, device, num_classes):
    model.eval()
    test_model(model, test_loader, device, num_classes)
    visualize_results(model, test_loader, device)            