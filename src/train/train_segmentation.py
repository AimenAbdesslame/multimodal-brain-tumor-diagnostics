# block 1 : imports & environment setup 
import os 
import torch 
import glob 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader 
import wandb as w 
from src.data.tumor_dataset import BraTS2DDataset
from src.models.unet_segmentation import UNet
from src.losses.SegmentationLoss import SegmentationLoss
import src.config as config
import time 


# block 2 : metrics & data helpers : 
## the evaluation metrics "dice score" : 
def dice_score(preds, targets, smooth=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    # 2. Flatten spatial dimensions per batch sample: (B, C, H, W) -> (B, C*H*W)
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


## slice path collector (replaces build_slice_indexes for pre-extracted .npz files):
def get_slice_paths(patient_dirs):
    """
    Collects pre-extracted .npz slice files belonging strictly to the assigned patient directories.
    Maintains strict patient-level separation to prevent data leakage.
    """
    slice_paths = []
    for p_dir in patient_dirs:
        if os.path.isdir(p_dir):
            slices = glob.glob(os.path.join(p_dir, "*.npz"))
        else:
            # Handles flattened file structures with patient prefix matching
            slices = glob.glob(f"{p_dir}*.npz")
        slice_paths.extend(slices)
    return sorted(slice_paths)


if __name__ == "__main__":
    # block 3 : init experiment tracker WandB , split data "patient_dir" , instantiate the dataset and a dataloader using the config.py
    ## initialize wandb :
    w.init(
        project=config.PROJECT_NAME,
        config={
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "device": str(config.DEVICE),
        },
    )
    
    ## patient directory spliting :
    ## why patient-level split is important :
    ## In medical imaging, especially in tasks like brain tumor segmentation, it's crucial to split the dataset at the patient level rather than at the slice level. This is because slices from the same patient are often highly correlated, and if slices from the same patient appear in both the training and validation sets, it can lead to data leakage. This means that the model might perform well on the validation set not because it has learned to generalize, but because it has seen very similar data during training. By ensuring that all slices from a single patient are only in either the training or validation set, we can better assess the model's ability to generalize to unseen patients. 
    patient_dirs = sorted(
        glob.glob(os.path.join(config.DATA_DIR, "BraTS20_Training_*")) 
    )
    train_data, val_data = train_test_split(patient_dirs, test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED)
    
    ## pass patient splits to collect .npz slice file paths: 
    train_slice_paths = get_slice_paths(train_data)
    val_slice_paths = get_slice_paths(val_data)
    
    ## use the dataSet class to create the 2D datasets for train and validation using .npz file paths: 
    train_dataset = BraTS2DDataset(train_slice_paths)
    val_dataset = BraTS2DDataset(val_slice_paths)
    
    ## instantiate the dataloaders for the train and validation DS : 
    train_loader = DataLoader(
        dataset=train_dataset, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
        batch_size=config.BATCH_SIZE
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True,
        batch_size=config.BATCH_SIZE
    )
    
    # block 4 : model , loss function and optimizer instantiation :
    model = UNet(in_channels=config.IN_CHANNELS, out_channels=config.OUT_CHANNELS).to(config.DEVICE)
    loss_fn = SegmentationLoss().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    best_val_dice = 0.0 # instance variable to keep track of the best validation dice score  
    patience = 5
    patience_counter = 0
    # Ensure output directory exists before training starts
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # block 5 : training loop : 
    for epoch in range(config.EPOCHS):
        print(f"Epoch: [{epoch+1}/{config.EPOCHS}] - Training and Validation")
        model.train()
        running_train_loss = 0.0
        epoch_start_time = time.time()

        # training phase : 
        for images, masks in train_loader: 
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        epoch_train_loss = running_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] - Training Loss: {epoch_train_loss:.4f}")
        
        # validation and evaluation phase : 
        model.eval()
        running_val_loss = 0.0
        running_val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader: 
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                logits = model(images)
                loss = loss_fn(logits, masks)
                running_val_loss += loss.item()
                running_val_dice += dice_score(logits, masks).item()
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_dice = running_val_dice / len(val_loader)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] - Validation Loss: {epoch_val_loss:.4f} - Validation Dice: {epoch_val_dice:.4f}")
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] completed in {epoch_time:.2f} seconds.")
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) # Memory in MB
        
        ## loging the metrics to wandb : 
        w.log(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "val_dice": epoch_val_dice,
                "epoch_time_sec": epoch_time,
                "peak_gpu_memory_MB": peak_gpu_mb
            }
        )
        
        ## checkpointing the model if the validation dice score improves :
        # --- Checkpointing & Early Stopping Logic ---
        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            patience_counter = 0
            checkpoint_path = os.path.join(config.OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--> Saved new best model (Val Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            print(f"--> No improvement in Val Dice. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
    print("Training process finished.")
    w.finish()