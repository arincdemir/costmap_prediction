import random
import torch
from torch.utils.data import Dataset, DataLoader

class GridDataset(Dataset):
    def __init__(self, grids_tensor, max_encodings=5, max_queries=5, repeat=100):
        # grids_tensor shape: [observation_count, steps, grid_size, grid_size]
        self.max_encodings = max_encodings
        self.max_queries = max_queries
        self.inputs = grids_tensor[:, :, :, :] 
        self.targets = grids_tensor[:, :, :, :]  
        self.repeat = repeat  # use to artificially inflate length if needed

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

        # Use first 5 frames for encoding and next 5 frames for querying.
        encoding_indices = torch.arange(0, 5)
        query_indices = torch.arange(5, 10)

        encoding_size = grid_size * grid_size + 1
        query_size = grid_size * grid_size

        padded_encodings = torch.zeros(self.max_encodings, encoding_size, dtype=torch.float32)
        encodings_mask = torch.zeros(self.max_encodings, dtype=torch.float32)

        padded_query_targets = torch.zeros(self.max_queries, query_size, dtype=torch.float32)
        padded_query_indices = torch.zeros(self.max_queries, 1, dtype=torch.float32)
        queries_mask = torch.zeros(self.max_queries, dtype=torch.float32)
        
        for i, step_index in enumerate(encoding_indices):
            # Safety check if num_steps is less than required index
            if step_index >= num_steps:
                break
            flattened = input_grid[step_index].flatten()
            step_index_normalized = step_index / num_steps
            grid_with_step_index = torch.cat((torch.tensor([step_index_normalized], dtype=torch.float32), flattened))
            padded_encodings[i] = grid_with_step_index
            encodings_mask[i] = 1

        for i, step_index in enumerate(query_indices):
            if step_index >= num_steps:
                break
            flattened = input_grid[step_index].flatten()
            padded_query_targets[i] = flattened
            step_index_normalized = step_index / num_steps
            padded_query_indices[i, 0] = step_index_normalized
            queries_mask[i] = 1
        
        return padded_encodings, encodings_mask, padded_query_indices, padded_query_targets, queries_mask