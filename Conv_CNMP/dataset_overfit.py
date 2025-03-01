import random
import torch
from torch.utils.data import Dataset

class GridDataset(Dataset):
    def __init__(self, grids_tensor, max_encodings=5, max_queries=5, repeat=100):
        # grids_tensor shape: [observation_count, steps, grid_size, grid_size]
        self.max_encodings = max_encodings
        self.max_queries = max_queries
        self.inputs = grids_tensor[:, :, :, :]
        self.targets = grids_tensor[:, :, :, :]
        self.repeat = repeat  # use to artificially inflate length if needed
        self.grid_size = grids_tensor.shape[2]  # Get grid size from the tensor

    def __len__(self):
        # if only one data item, return repeat value; otherwise, return actual length
        if self.inputs.shape[0] == 1:
            return self.repeat
        else:
            return self.inputs.shape[0]

    def __getitem__(self, idx):
        # if only one data item, always use index 0
        if self.inputs.shape[0] == 1:
            idx = 0

        input_grid = self.inputs[idx]
        num_steps, grid_size, _ = input_grid.shape

        # For overfit dataset, use fixed indices - first 5 frames for encoding and next 5 frames for querying
        encoding_indices = torch.arange(0, 5)
        query_indices = torch.arange(5, 10)

        # Create padded tensors for time indices and corresponding grids
        padded_time_indices = torch.zeros(self.max_encodings, 1, dtype=torch.float32)
        padded_grids = torch.zeros(self.max_encodings, grid_size, grid_size, dtype=torch.float32)
        encodings_mask = torch.zeros(self.max_encodings, dtype=torch.float32)

        # For queries
        padded_query_targets = torch.zeros(self.max_queries, self.grid_size * self.grid_size, dtype=torch.float32)
        padded_query_indices = torch.zeros(self.max_queries, 1, dtype=torch.float32)
        queries_mask = torch.zeros(self.max_queries, dtype=torch.float32)
        
        # Fill encoding tensors
        for i, step_index in enumerate(encoding_indices):
            # Safety check if num_steps is less than required index
            if step_index >= num_steps:
                break
                
            # Store the time index
            step_index_normalized = step_index / num_steps
            padded_time_indices[i, 0] = step_index_normalized
            
            # Store the grid directly (not flattened)
            padded_grids[i] = input_grid[step_index]
            
            # Set mask for valid entries
            encodings_mask[i] = 1.0

        # Fill query tensors
        for i, step_index in enumerate(query_indices):
            if step_index >= num_steps:
                break
                
            # Store flattened grid as target
            padded_query_targets[i] = input_grid[step_index].flatten()
            
            # Store time index
            step_index_normalized = step_index / num_steps
            padded_query_indices[i, 0] = step_index_normalized
            
            # Set mask for valid entries
            queries_mask[i] = 1.0
        
        return padded_time_indices, padded_grids, encodings_mask, padded_query_indices, padded_query_targets, queries_mask