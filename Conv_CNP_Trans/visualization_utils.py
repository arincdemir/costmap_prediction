import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for headless environments
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
from PIL import Image
import wandb


def visualize_model_predictions(model, data_tensor, num_samples=3, num_encoding=5, num_query=5):
    """
    Generate visualizations for model predictions on multiple data samples
    
    Args:
        model: The trained CNN_CNMP model
        data_tensor: Tensor containing all grid data samples
        num_samples: Number of different data samples to visualize
        num_encoding: Number of encoding steps to use
        num_query: Number of query/prediction steps to generate
        
    Returns:
        List of PIL Image objects containing the visualizations
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get grid size from model
    grid_size = model.grid_size
    max_encodings = 5  # as used in the model and dataset
    
    visualizations = []
    
    # Process each requested sample
    for sample_idx in range(min(num_samples, len(data_tensor))):
        # Get the simulation data
        simulation = data_tensor[sample_idx]  # shape: (steps, grid_size, grid_size)
        steps = simulation.shape[0]
        
        # Prepare encoding inputs
        padded_time_indices = torch.zeros(max_encodings, 1, dtype=torch.float32)
        padded_grids = torch.zeros(max_encodings, grid_size, grid_size, dtype=torch.float32)
        encodings_mask = torch.zeros(max_encodings, dtype=torch.bool)

        for i in range(num_encoding):
            # Normalize time index using total steps
            padded_time_indices[i, 0] = float(i) / steps
            padded_grids[i] = simulation[i]
            encodings_mask[i] = True

        # Prepare query inputs
        query_steps = list(range(num_encoding, num_encoding + num_query))
        padded_query_indices = (torch.tensor(query_steps, dtype=torch.float32).unsqueeze(1) / steps)
        queries_mask = torch.ones(num_query, dtype=torch.bool)

        # Query the model for predictions
        with torch.no_grad():
            output = model(
                padded_time_indices.unsqueeze(0),
                padded_grids.unsqueeze(0),
                encodings_mask.unsqueeze(0),
                padded_query_indices.unsqueeze(0),
                queries_mask.unsqueeze(0)
            )

        # Get predictions
        predicted_grids = output[0].cpu().numpy()

         # Create visualization
        fig = plt.figure(figsize=(15, 6))
        
        # Top row: Overview of ground truth steps
        for i in range(min(10, steps)):
            ax = plt.subplot(2, 10, i + 1)
            ax.imshow(simulation[i].numpy(), cmap='Greys', interpolation='none')
            ax.set_title(f"GT Step {i}")
            rect = patches.Rectangle(
                (-0.5, -0.5), grid_size, grid_size,
                linewidth=1, edgecolor='black', facecolor='none'
            )
            ax.add_patch(rect)
            ax.axis('off')
            
        # Bottom row: First 5 ground truth steps followed by 5 predicted steps
        # Ground truth images
        for i in range(5):
            ax = plt.subplot(2, 10, 10 + i + 1)
            if i < steps:
                ax.imshow(simulation[i].numpy(), cmap='Greys', interpolation='none')
            ax.set_title(f"GT {i}")
            rect = patches.Rectangle(
                (-0.5, -0.5), grid_size, grid_size,
                linewidth=1, edgecolor='black', facecolor='none'
            )
            ax.add_patch(rect)
            ax.axis('off')
        
        # Predicted images
        for i in range(5):
            ax = plt.subplot(2, 10, 10 + 5 + i + 1)
            if i < predicted_grids.shape[0]:
                ax.imshow(predicted_grids[i], cmap='Greys', interpolation='none')
            ax.set_title(f"Pred {i}")
            rect = patches.Rectangle(
                (-0.5, -0.5), grid_size, grid_size,
                linewidth=1, edgecolor='black', facecolor='none'
            )
            ax.add_patch(rect)
            ax.axis('off')
        
        plt.suptitle(f"Sample {sample_idx + 1}")
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        image = Image.open(buf)
        visualizations.append(image)
        plt.close(fig)
    
    return visualizations


def log_visualizations_to_wandb(model_path, data_path, wandb_run, num_samples=3):
    """
    Generate visualizations and log them to wandb
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the data tensor
        wandb_run: Active wandb run
        num_samples: Number of samples to visualize
    """
    # Import model class
    from CNN_CNMP import CNN_CNMP
    
    # Load the data
    all_grids_tensor = torch.load(data_path, map_location=torch.device("cpu"))
    
    # Extract model parameters from wandb config
    config = wandb_run.config
    
    # Initialize the model with the same parameters used in training
    model = CNN_CNMP(
        t_dim=config.t_dim,
        grid_size=config.grid_size,
        encoder_hidden_dims=config.encoder_hidden_dims,
        decoder_hidden_dims=config.decoder_hidden_dims,
        latent_dim=config.latent_dim,
        cnn_channels=config.cnn_channels,
        dropout_rate=config.dropout_rate
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    
    # Generate visualizations
    print(f"Generating visualizations for {num_samples} samples...")
    visualizations = visualize_model_predictions(model, all_grids_tensor, num_samples=num_samples)
    
    # Log images to wandb
    print(f"Uploading {len(visualizations)} visualizations to wandb...")
    for idx, img in enumerate(visualizations):
        wandb_run.log({f"visualization_sample_{idx+1}": wandb.Image(img)})



if __name__ == '__main__':
    import torch
    from CNN_CNMP import CNN_CNMP


    # Load the data tensor
    all_grids_tensor = torch.load("Conv_CNP_Trans/grids_tensor.pt", map_location=torch.device('cpu'))
    
    # Define a dummy config as used during training.
    # Adjust these parameters as needed to match your trained model.
    dummy_config = {
        't_dim': 1,
        'grid_size': 32,
        'encoder_hidden_dims': [512,512],
        'decoder_hidden_dims': [512,1024,1024],
        'latent_dim': 384,
        'cnn_channels': [63,128,256],
        'dropout_rate': 0.14754
    }
    
    # Initialize the model using the dummy config
    model = CNN_CNMP(
        t_dim=dummy_config['t_dim'],
        grid_size=dummy_config['grid_size'],
        encoder_hidden_dims=dummy_config['encoder_hidden_dims'],
        decoder_hidden_dims=dummy_config['decoder_hidden_dims'],
        latent_dim=dummy_config['latent_dim'],
        cnn_channels=dummy_config['cnn_channels'],
        dropout_rate=dummy_config['dropout_rate']
    )
    
    # Load model weights
    model.load_state_dict(torch.load("Conv_CNP_Trans/best_model_3uolv1we.pth", map_location=torch.device('cpu')))
    
    # Generate visualizations and save each image
    visualizations = visualize_model_predictions(model, all_grids_tensor, num_samples=3)
    for idx, img in enumerate(visualizations):
        out_file = f"visualization_sample_{idx+1}.png"
        img.save(out_file)
        print(f"Saved visualization: {out_file}")