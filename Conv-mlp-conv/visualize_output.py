import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from train import GridDataset
from Conv_MLP import ConvMLPConv
from generate_data import CAPTURED_GRID_SIZE

def load_model(model_path):
    """Load the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvMLPConv()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model, device

def load_test_samples(num_samples=10):
    """Load test samples from the dataset."""
    dataset = GridDataset("grids_tensor.pt")
    indices = range(num_samples)
    return [dataset[i] for i in indices]

def visualize_basic(model, device, input_frames, target_frame=None):
    """Basic visualization comparing input, prediction, and ground truth."""
    with torch.no_grad():
        # Add batch dimension and move to device
        input_tensor = input_frames.unsqueeze(0).to(device)
        
        # Get model prediction
        output = model(input_tensor)
        
        # Convert tensors to numpy for visualization
        input_np = input_frames.cpu().numpy()
        output_np = output.squeeze(0).cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        
        # Plot input frames
        for i in range(3):
            axes[i].imshow(input_np[i], cmap='Greys', interpolation='none')
            axes[i].set_title(f"Input t={i}")
            axes[i].axis("off")
        
        # Plot predicted output
        axes[3].imshow(output_np[0], cmap='Greys', interpolation='none')
        axes[3].set_title("Predicted t=3")
        axes[3].axis("off")
        
        # Plot ground truth if available and calculate metrics
        if target_frame is not None:
            target_np = target_frame.cpu().numpy()
            axes[4].imshow(target_np[0], cmap='Greys', interpolation='none')
            
            # Calculate metrics
            mse = np.mean((output_np[0] - target_np[0]) ** 2)
            mae = np.mean(np.abs(output_np[0] - target_np[0]))
            
            axes[4].set_title(f"Ground Truth\nMSE: {mse:.4f}, MAE: {mae:.4f}")
            axes[4].axis("off")
        else:
            axes[4].axis("off")
            
        plt.tight_layout()
        return fig

def create_output_directory():
    """Create output directory for visualizations."""
    os.makedirs("visualizations", exist_ok=True)
    return "visualizations"

def main():
    # Check if model file exists
    model_path = "conv_mlp_trained.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please run train.py first to train the model.")
        return
        
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model, device = load_model(model_path)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Use samples from dataset
    print("Visualizing samples from dataset...")
    try:
        test_samples = load_test_samples(num_samples=10)
        
        for i, (input_frames, target_frame) in enumerate(test_samples):
            # Basic visualization
            fig = visualize_basic(model, device, input_frames, target_frame)
            fig.savefig(f"{output_dir}/sample_{i}.png")
            plt.close(fig)
            
            print(f"Processed dataset sample {i+1}/10")
    except Exception as e:
        print(f"Error loading dataset samples: {e}")
    
    print(f"\nVisualization complete! Check the files in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()