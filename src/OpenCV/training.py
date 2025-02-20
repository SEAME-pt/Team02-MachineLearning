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
        # "/Users/ruipedropires/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set/label_data_0313.json",
        # "/Users/ruipedropires/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set/label_data_0531.json",
        "/Users/ruipedropires/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set/label_data_0601.json"
    ],
    img_dir="/Users/ruipedropires/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set",
)
 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = LaneSegmentationModel().to(device)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    total_samples = len(dataset)
    print(f"Total training samples: {total_samples}")

    num_epochs = 1
    batch_size = 16
    steps_per_epoch = total_samples // batch_size
    print(f"Steps per epoch: {steps_per_epoch}")

    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                visualize_batch(images, masks, outputs)
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Step [{batch_idx+1}/{steps_per_epoch}], "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    print("Training finished!")
    torch.save(model.state_dict(), "lane_segmentation.pth")