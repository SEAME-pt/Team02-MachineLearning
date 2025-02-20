import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_batch(images, masks=None, outputs=None):
    img = images[0].cpu().permute(1, 2, 0).numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')
    
    # Plot ground truth mask (single channel)
    if masks is not None:
        mask = masks[0].cpu().numpy()
        plt.subplot(132)
        plt.imshow(mask[0], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
    
    # Plot prediction if available
    if outputs is not None:
        pred = torch.sigmoid(outputs[0]).cpu().detach().numpy()
        plt.subplot(133)
        plt.imshow(pred[0], cmap='jet')
        plt.colorbar()
        plt.title('Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)
    

def visualize_output_batch(image, outputs=None):
    img = image[0].cpu().permute(1, 2, 0).numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')
    
    # Plot prediction if available
    if outputs is not None:
        pred = torch.sigmoid(outputs[0]).cpu().detach().numpy()  # Changed to sigmoid for single channel
        plt.subplot(132)
        plt.imshow(pred[0], cmap='jet')
        plt.colorbar()
        plt.title('Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)