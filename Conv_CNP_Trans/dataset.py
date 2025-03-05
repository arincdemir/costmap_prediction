import random
import torch
from torch.utils.data import Dataset

class GridDataset(Dataset):
    def __init__(self, grids_tensor, max_encodings=5, max_queries=5, populate_factor=4):
        # grids_tensor shape: [observation_count, steps, grid_size, grid_size]
        self.max_encodings = max_encodings
        self.max_queries = max_queries
        self.inputs = grids_tensor
        self.populate_factor = populate_factor
        self.grid_size = grids_tensor.shape[2]  # Get grid size from the tensor
        self.input_size = grids_tensor.shape[0]  # the size of the actual dataset

    def __len__(self):
        # multiply it by that factor so that you see a trajectory multiple times with different random points
        return self.input_size * self.populate_factor   
    

    def __getitem__(self, idx):
        idx = idx % self.input_size
        input_grid = self.inputs[idx]
        num_steps, grid_size, _ = input_grid.shape
        
        num_encodings = random.randint(1, self.max_encodings)
        num_queries = random.randint(1, self.max_queries)
        
        steps_permuted = torch.randperm(num_steps)
        encoding_indices = steps_permuted[:num_encodings]
        query_indices = steps_permuted[num_encodings:num_encodings + num_queries]
        
        # Create padded tensors for time indices and corresponding grids
        padded_time_indices = torch.zeros(self.max_encodings, 1, dtype=torch.float32)
        padded_grids = torch.zeros(self.max_encodings, grid_size, grid_size, dtype=torch.float32)
        encodings_mask = torch.zeros(self.max_encodings, dtype=torch.float32)

        # For queries - keep 2D grid structure for targets
        padded_query_targets = torch.zeros(self.max_queries, grid_size, grid_size, dtype=torch.float32)
        padded_query_indices = torch.zeros(self.max_queries, 1, dtype=torch.float32)
        queries_mask = torch.zeros(self.max_queries, dtype=torch.float32)
        
        # Fill encoding tensors
        for i, step_index in enumerate(encoding_indices):
            # Store the time index
            step_index_normalized = step_index / num_steps
            padded_time_indices[i, 0] = step_index_normalized
            
            # Store the grid directly (not flattened)
            padded_grids[i] = input_grid[step_index]
            
            # Set mask for valid entries
            encodings_mask[i] = 1.0

        # Fill query tensors
        for i, step_index in enumerate(query_indices):
            # Store grid as target without flattening
            padded_query_targets[i] = input_grid[step_index]
            
            # Store time index
            step_index_normalized = step_index / num_steps
            padded_query_indices[i, 0] = step_index_normalized
            
            # Set mask for valid entries
            queries_mask[i] = 1.0
        
        return padded_time_indices, padded_grids, encodings_mask, padded_query_indices, padded_query_targets, queries_mask