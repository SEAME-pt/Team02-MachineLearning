import torch
from tqdm import tqdm

# Train YOLO model
def train_yolo_model(model, train_loader, criterion, optimizer, device, epochs=20):
    """
    Train the YOLO model specifically for object detection
    """
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch_idx, (images, obj_targets) in enumerate(train_bar):
            # Move images to device
            images = images.to(device)
            
            # Prepare targets with batch indices
            targets_with_batch = []
            for i, target in enumerate(obj_targets):
                if target.size(0) > 0:
                    batch_indices = torch.full((target.size(0), 1), i, 
                                             dtype=torch.float32, device=device)
                    target = torch.cat([batch_indices, target.to(device)], dim=1)
                    targets_with_batch.append(target)
                else:
                    targets_with_batch.append(torch.zeros((0, 6), device=device))
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss using YOLO loss
            loss = criterion(predictions, targets_with_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f'Models/Obj/yolo_model_epoch_{epoch+1}.pth')
    
    return model
