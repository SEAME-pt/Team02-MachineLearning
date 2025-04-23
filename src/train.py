import torch
from tqdm import tqdm

# Train YOLO model
def train_yolo_model(model, train_loader, criterion, optimizer, device, epochs=20):
    """
    Train the YOLO model specifically for object detection
    """

    old_lr = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Warm-up learning rate
    warmup_epochs = 3
    warmup_lr_scheduler = None
    if warmup_epochs > 0:
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs * len(train_loader)
        )

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

            if epoch < warmup_epochs:
                warmup_lr_scheduler.step()
            
            # Update statistics
            running_loss += loss.item()
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f'Models/Obj/yolo4_model_epoch_{epoch+1}.pth')
        
        # Update learning rate
        scheduler.step(avg_loss)
        new_lr = [group['lr'] for group in optimizer.param_groups][0]
        if new_lr != old_lr:
            print(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
        old_lr = new_lr
    
    return model
