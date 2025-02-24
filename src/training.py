import torch
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from Model import LaneSegmentationModel
from Dataset import TuSimpleDataset
from utils import visualize_batch

dataset = TuSimpleDataset(
    json_paths=[
        "/home/luis_t2/OpenCV/assets/TUSimple/train_set/label_data_0313.json",
        "/home/luis_t2/OpenCV/assets/TUSimple/train_set/label_data_0531.json",
        "/home/luis_t2/OpenCV/assets/TUSimple/train_set/label_data_0601.json"
    ],
    img_dir="/home/luis_t2/OpenCV/assets/TUSimple/train_set/",
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():  # For Apple Silicon
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("Using CPU")
model = LaneSegmentationModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    num_epochs = 20
    batch_size = 8
    best_loss = float('inf')
    best_model_path = "best_lane_segmentation.pth"

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % 5 == 0:
                # visualize_batch(images, masks, outputs)
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Loss: {loss.item():.4f}")

    # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Save the model if it has the best loss so far
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"Saved new best model with loss: {best_loss:.4f}")

    print("Training finished!")
    print(f"Best model saved with loss: {best_loss:.4f}")