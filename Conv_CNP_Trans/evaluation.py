import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from CNN_CNMP import CNN_CNMP

class FixedContextDataset(Dataset):
    def __init__(self, grids_tensor, context_steps, query_steps):
        """
        Dataset for fixed context and query steps
        
        Args:
            grids_tensor: Tensor of shape [observation_count, steps, grid_size, grid_size]
            context_steps: Number of initial steps to use as context
            query_steps: Number of subsequent steps to predict
        """
        self.inputs = grids_tensor
        self.context_steps = context_steps
        self.query_steps = query_steps
        self.grid_size = grids_tensor.shape[2]
        self.input_size = grids_tensor.shape[0]
        
        # Verify we have enough steps
        assert grids_tensor.shape[1] >= context_steps + query_steps, \
            f"Not enough time steps in data. Need at least {context_steps + query_steps} steps."

    def __len__(self):
        return self.input_size

    def __getitem__(self, idx):
        input_grid = self.inputs[idx]
        num_steps, grid_size, _ = input_grid.shape
        
        # Get specific context and query indices
        context_indices = list(range(self.context_steps))
        query_indices = list(range(self.context_steps, self.context_steps + self.query_steps))
        
        # Create tensors for context
        padded_time_indices = torch.zeros(self.context_steps, 1, dtype=torch.float32)
        padded_grids = torch.zeros(self.context_steps, grid_size, grid_size, dtype=torch.float32)
        encodings_mask = torch.ones(self.context_steps, dtype=torch.float32)

        # Create tensors for queries
        padded_query_targets = torch.zeros(self.query_steps, grid_size, grid_size, dtype=torch.float32)
        padded_query_indices = torch.zeros(self.query_steps, 1, dtype=torch.float32)
        queries_mask = torch.ones(self.query_steps, dtype=torch.float32)
        
        # Fill context tensors
        for i, step_index in enumerate(context_indices):
            padded_time_indices[i, 0] = step_index / num_steps
            padded_grids[i] = input_grid[step_index]

        # Fill query tensors
        for i, step_index in enumerate(query_indices):
            padded_query_targets[i] = input_grid[step_index]
            padded_query_indices[i, 0] = step_index / num_steps
        
        return padded_time_indices, padded_grids, encodings_mask, padded_query_indices, padded_query_targets, queries_mask

def evaluate_model(model, data_loader, device):
    """Evaluate the model on the given data loader and return the average loss"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            # Move data to device
            batch = [tensor.to(device) for tensor in batch]
            padded_time_indices, padded_grids, encodings_mask, padded_query_indices, padded_query_targets, queries_mask = batch
            
            # Forward pass
            output = model(padded_time_indices, padded_grids, encodings_mask, padded_query_indices, queries_mask)
            loss = model.loss(output, padded_query_targets, queries_mask)
            
            # Accumulate loss
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def main():
    data_path = "grids_tensor.pt"
    model_path = "best_model_standard_run.pth"
    batch_size = 64
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {data_path}")
    grids_tensor = torch.load(data_path)
    
    # Create validation set
    val_size = int(len(grids_tensor))
    val_data = grids_tensor[:val_size]
    
    # Get grid size from data
    grid_size = grids_tensor.shape[2]
    
    # Load model
    print(f"Loading model from {model_path}")
    model = CNN_CNMP(
        t_dim=1,
        grid_size=grid_size,
        encoder_hidden_dims=[394, 394],
        decoder_hidden_dims=[394, 394, 394],
        latent_dim=394,
        cnn_channels=[20, 20, 20],
        dropout_rate=0.125
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Define evaluation configurations
    configurations = [
        {"context_steps": 2, "query_steps": 5},
        {"context_steps": 3, "query_steps": 5},
        {"context_steps": 5, "query_steps": 5},
        {"context_steps": 7, "query_steps": 3},
        {"context_steps": 9, "query_steps": 1}
    ]
    
    # Evaluate each configuration
    print("\n=== Evaluation Results ===")
    for config in configurations:
        context_steps = config["context_steps"]
        query_steps = config["query_steps"]
        
        print(f"\nEvaluating with first {context_steps} steps as context and predicting the next {query_steps} steps...")
        
        # Create dataset and loader
        dataset = FixedContextDataset(val_data, context_steps, query_steps)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        # Evaluate and print results
        loss = evaluate_model(model, loader, device)
        print(f"Configuration: {context_steps} context → {query_steps} query steps")
        print(f"Average loss: {loss:.6f}")
    
if __name__ == "__main__":
    main()