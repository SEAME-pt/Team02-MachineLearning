import torch
from tqdm import tqdm
import numpy as np
from src.ObjectDetection import SimpleYOLO, YOLOLoss, generate_anchors

# Training function
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Train and validate model
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function 
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs to train for
    """
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                        leave=True, position=0, 
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for batch_idx, (images, masks, obj_targets) in enumerate(train_bar):
            # Move images to device
            images = images.to(device)
            
            # For YOLO loss, we need to add batch indices to targets
            targets_with_batch = []
            for i, target in enumerate(obj_targets):
                if target.size(0) > 0:  # Check if there are any targets
                    # Add batch index as first column
                    batch_indices = torch.full((target.size(0), 1), i, 
                                             dtype=torch.float32, device=device)
                    target = torch.cat([batch_indices, target.to(device)], dim=1)
                    targets_with_batch.append(target)
                else:
                    # Empty target
                    targets_with_batch.append(torch.zeros((0, 6), device=device))
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss = criterion(predictions, targets_with_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            
            # Update progress bar
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Calculate epoch statistics
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}")
        
        # # Validation phase
        # model.eval()
        # val_loss = 0.0
        
        # val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]', 
        #               leave=True, position=0, 
        #               bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        # # Disable gradients during validation
        # with torch.no_grad():
        #     for inputs, targets in val_bar:
        #         inputs = inputs.to(device)
        #         targets = targets.to(device)
                
        #         outputs = model(inputs)
        #         loss = criterion(outputs, targets)
                
        #         val_loss += loss.item()
        #         val_bar.set_postfix(loss=f'{loss.item():.4f}')
        
        # avg_val_loss = val_loss / len(val_loader)
        
        # # Print epoch results
        # print(f'\nEpoch {epoch+1}/{epochs}:')
        # print(f'  Training Loss: {avg_train_loss:.4f}')
        # print(f'  Validation Loss: {avg_val_loss:.4f}')
        
        # # Save model if validation loss improved
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        print(f'  Validation loss improved! Saving model...')
        torch.save(model.state_dict(), f'Models/temp/lane_model2_epoch_{epoch+1}.pth')
    

# Train YOLO model
def train_yolo_model(model, train_loader, criterion, optimizer, device, epochs=20):
    """
    Train the YOLO model specifically for object detection
    """
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch_idx, (images, masks, obj_targets) in enumerate(train_bar):
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
        torch.save(model.state_dict(), f'Models/yolo_model_epoch_{epoch+1}.pth')
    
    return model
