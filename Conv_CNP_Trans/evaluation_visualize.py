import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os
import numpy as np
from CNN_CNMP import CNN_CNMP

def visualize_predictions(model, data_tensor, context_steps, query_steps, num_samples=5, save_dir="evaluation_viz"):
    """
    Visualize model predictions for specific context-query configurations
    
    Args:
        model: Trained CNN_CNMP model
        data_tensor: Tensor of shape [observation_count, steps, grid_size, grid_size]
        context_steps: Number of steps to use as context
        query_steps: Number of steps to predict
        num_samples: Number of data samples to visualize
        save_dir: Directory to save visualizations
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get grid size from model
    grid_size = model.grid_size
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Process each requested sample
    for sample_idx in range(min(num_samples, len(data_tensor))):
        # Get the simulation data
        simulation = data_tensor[sample_idx]  # shape: (steps, grid_size, grid_size)
        steps = simulation.shape[0]
        
        # Make sure we have enough steps
        if steps < context_steps + query_steps:
            print(f"Sample {sample_idx} has only {steps} steps, need {context_steps + query_steps}. Skipping.")
            continue
        
        # Prepare context inputs (on the same device as the model)
        padded_time_indices = torch.zeros(context_steps, 1, dtype=torch.float32).to(device)
        padded_grids = torch.zeros(context_steps, grid_size, grid_size, dtype=torch.float32).to(device)
        encodings_mask = torch.ones(context_steps, dtype=torch.float32).to(device)

        # Fill context tensors
        for i in range(context_steps):
            padded_time_indices[i, 0] = i / steps
            padded_grids[i] = simulation[i].to(device)

        # Prepare query inputs
        query_indices = list(range(context_steps, context_steps + query_steps))
        padded_query_indices = torch.zeros(query_steps, 1, dtype=torch.float32).to(device)
        queries_mask = torch.ones(query_steps, dtype=torch.float32).to(device)
        
        for i, step_idx in enumerate(query_indices):
            padded_query_indices[i, 0] = step_idx / steps

        # Query the model for predictions
        with torch.no_grad():
            output = model(
                padded_time_indices.unsqueeze(0),
                padded_grids.unsqueeze(0),
                encodings_mask.unsqueeze(0),
                padded_query_indices.unsqueeze(0),
                queries_mask.unsqueeze(0)
            )

        # Get predictions and move to CPU for visualization
        predicted_grids = output[0].cpu().numpy()
        
        # Calculate total columns (context + query)
        total_cols = context_steps + query_steps
        
        # Create visualization with 2 rows (ground truth/context on top, predictions on bottom)
        fig = plt.figure(figsize=(15, 6))
        
        # First row: Context frames followed by ground truth frames
        # Context frames
        for i in range(context_steps):
            ax = plt.subplot(2, total_cols, i + 1)
            ax.imshow(simulation[i].numpy(), cmap='Greys', interpolation='none')
            ax.set_title(f"Bağlam {i}")
            rect = patches.Rectangle(
                (-0.5, -0.5), grid_size, grid_size,
                linewidth=1, edgecolor='blue', facecolor='none'
            )
            ax.add_patch(rect)
            ax.axis('off')
        
        # Ground truth for query steps (in top row after context)
        for i in range(query_steps):
            ax = plt.subplot(2, total_cols, context_steps + i + 1)
            ax.imshow(simulation[context_steps + i].numpy(), cmap='Greys', interpolation='none')
            ax.set_title(f"Gerçek {context_steps + i}")
            rect = patches.Rectangle(
                (-0.5, -0.5), grid_size, grid_size,
                linewidth=1, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)
            ax.axis('off')
        
        # Second row: Empty space below context, predictions below ground truth
        # Empty space below context (or you could use it for other information)
        # Then predictions aligned with ground truth
        for i in range(query_steps):
            ax = plt.subplot(2, total_cols, total_cols + context_steps + i + 1)
            ax.imshow(predicted_grids[i], cmap='Greys', interpolation='none')
            ax.set_title(f"Tahmin {context_steps + i}")
            rect = patches.Rectangle(
                (-0.5, -0.5), grid_size, grid_size,
                linewidth=1, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(save_dir, f"sample_{sample_idx+1}_context{context_steps}_query{query_steps}.png")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved visualization: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize model evaluation results')
    parser.add_argument('--model_path', type=str, default="best_model_standard_run.pth", help='Path to the trained model')
    parser.add_argument('--data_path', type=str, default="grids_tensor.pt", help='Path to the data tensor')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default="evaluation_visualizations", help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    grids_tensor = torch.load(args.data_path)
    
    # Get grid size from data
    grid_size = grids_tensor.shape[2]
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = CNN_CNMP(
        t_dim=1,
        grid_size=grid_size,
        encoder_hidden_dims=[394, 394],
        decoder_hidden_dims=[394, 394, 394],
        latent_dim=394,
        cnn_channels=[20, 20, 20],
        dropout_rate=0.125
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Define evaluation configurations - same as in evaluation.py
    configurations = [
        {"context_steps": 2, "query_steps": 5},
        {"context_steps": 3, "query_steps": 5},
        {"context_steps": 5, "query_steps": 5},
        {"context_steps": 7, "query_steps": 3},
        {"context_steps": 9, "query_steps": 1}
    ]
    
    # Visualize each configuration
    for config in configurations:
        context_steps = config["context_steps"]
        query_steps = config["query_steps"]
        
        print(f"\nVisualizing with {context_steps} context steps and {query_steps} query steps...")
        visualize_predictions(
            model, 
            grids_tensor, 
            context_steps, 
            query_steps, 
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )
    
    print(f"\nAll visualizations saved to {args.save_dir}")

if __name__ == "__main__":
    main()