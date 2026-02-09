import torch
import torch.nn as nn

from modules import Unet, config
from dataset import get_loaders
from predict import result_model


def train(model: Unet, train_loader, val_loader, optimizer, criterion, config_dict, device: str) -> None:
    
    print_every = 50
    min_val_loss = float('inf')
    
    for epoch in range(config_dict["num_epochs"]):
        running_loss = 0.0
        for i, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            y_pred = model(imgs)
            loss = criterion(y_pred, masks)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{config_dict['num_epochs']}], Step [{i+1}/{len(train_loader)}], "
                    f"Loss: {running_loss / print_every:.4f}")
                running_loss = 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{config_dict['num_epochs']}], Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model with loss: {min_val_loss:.4f}")
        
        model.train()


def main() -> None:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(1 , 4).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config()["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    config_dict = config()
    train_loader, val_loader, test_loader = get_loaders(config_dict)
    
    
    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Train loader:", len(train_loader))
    print("Validation loader:", len(val_loader))
    print("Test loader:", len(test_loader))
    
    imgs, masks = next(iter(train_loader))
    print("Image batch:", imgs.shape)  
    print("Mask batch:", masks.shape)  

    model.train()
    train(model, train_loader, val_loader, optimizer, criterion, config_dict, device)
        
    model.load_state_dict(torch.load("best_model.pth"))    
    result_model(model, test_loader, device, config_dict["out_channels"])
        
    if __name__ == "__main__":
        main()