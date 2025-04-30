import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from .loss import DiscriminativeLoss

# Training function
def train_model(model, train_loader, optimizer, device, epochs=10):
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

    criterion_ce = nn.CrossEntropyLoss()
    criterion_disc = DiscriminativeLoss(delta_var=0.5,
                                        delta_dist=1.5,
                                        norm=2,
                                        usegpu=True)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                        leave=True, position=0, 
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for i, (inputs, bin_labels, ins_labels, n_lanes) in enumerate(train_bar):
            inputs = inputs.to(device)
            bin_labels = bin_labels.to(device)
            ins_labels = ins_labels.to(device)
            n_lanes = n_lanes.to(device)
            
            bin_preds, ins_preds = model(inputs)

            _, bin_labels_ce = bin_labels.max(1)
            be_loss = criterion_ce(
            bin_preds.permute(0, 2, 3, 1).contiguous().view(-1, 2), bin_labels_ce.view(-1))

            disc_loss = criterion_disc(ins_preds, ins_labels, n_lanes)
            loss = be_loss + disc_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress
            train_loss += loss.item()
            train_bar.set_postfix(loss=f'{loss.item():.4f}')
        
        # Save model
        torch.save(model.state_dict(), f'Models/lane/lane_unet3_ins_ce_epoch_{epoch+1}.pth')
    
    return model                                                                                                               