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
        "/Users/ruipedropires/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set/label_data_0313.json",
        "/Users/ruipedropires/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set/label_data_0531.json",
        "/Users/ruipedropires/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set/label_data_0601.json"
    ],
    img_dir="/Users/ruipedropires/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set",
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = LaneSegmentationModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    num_epochs = 1
    batch_size = 16

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 5 == 0:
                visualize_batch(images, masks, outputs)
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Loss: {loss.item():.4f}")

    print("Training finished!")
    torch.save(model.state_dict(), "lane_segmentation.pth")