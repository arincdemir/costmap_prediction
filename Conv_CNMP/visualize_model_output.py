import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from CNMP import CNMP

# Hyperparameters (must match training)
grid_size = 32
max_encodings = 5  # as used in the model and dataset
max_queries = 5    # new parameter for querying

# Hyperparameters
t_dim = 1                   # step index dimension
SM_dim = 32 ** 2            # grid flattened dimension (16*16)
encoder_hidden_dims = [1024, 1024, 1024]
latent_dim = 1024
decoder_hidden_dims = [1024, 1024, 1024]

# Load the trained model
model = CNMP(
    t_dim=t_dim,
    SM_dim=SM_dim,
    encoder_hidden_dims=encoder_hidden_dims,
    decoder_hidden_dims=decoder_hidden_dims,
    latent_dim=latent_dim
)
model_path = os.path.join(os.path.dirname(__file__), "trained_cnmp_best.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Load simulation data (all grids)
data_path = os.path.join(os.path.dirname(__file__), "grids_tensor.pt")
all_grids_tensor = torch.load(data_path, map_location=torch.device("cpu"))

# Pick one simulation sample (here sample index 0)
simulation = all_grids_tensor[-1]  # shape: (steps, grid_size, grid_size)
steps = simulation.shape[0]

# Use the first 5 steps as encoding and query the next 5 steps
num_encoding = 5
num_query = 5
flat_size = SM_dim + 1  # 1 for the step index + flattened grid

# Create padded encoding inputs and mask
padded_inputs = torch.zeros(max_encodings, flat_size, dtype=torch.float32)
mask = torch.zeros(max_encodings, dtype=torch.bool)

for i in range(num_encoding):
    grid = simulation[i]           # shape: (grid_size, grid_size)
    flat = grid.flatten()          # shape: (SM_dim,)
    index_tensor = torch.tensor([i], dtype=torch.float32) / steps
    padded_inputs[i] = torch.cat((index_tensor, flat))
    mask[i] = True

# Prepare query inputs: time steps from num_encoding to num_encoding+num_query-1
query_steps = list(range(num_encoding, num_encoding + num_query))
padded_query_indices = torch.tensor(query_steps, dtype=torch.float32).unsqueeze(1) / steps  # shape: (num_query, 1)
query_mask = torch.ones(num_query, dtype=torch.bool)

#print(padded_inputs, mask, padded_query_indices)

# Query the model for the next 5 time steps at once
with torch.no_grad():
    # Add batch dimension; model expects (batch, max_encodings, feature) for encodings and similar for queries
    output = model(padded_inputs.unsqueeze(0),
                   mask.unsqueeze(0),
                   padded_query_indices.unsqueeze(0),
                   query_mask.unsqueeze(0))
    
# Output shape: (1, num_query, SM_dim*2); extract the first SM_dim as mean prediction and reshape
predicted_grids = []
standard_deviations = []
for i in range(num_query):
    pred_flat = output[0, i, :SM_dim]
    grid_pred = pred_flat.reshape(grid_size, grid_size).numpy()
    grid_pred = np.clip(grid_pred, 0, 1)  # Clamp the predictions between 0 and 1
    predicted_grids.append(grid_pred)

    stds_flat = output[0, i, SM_dim:]
    standard_deviations.append(stds_flat)
plt.figure(figsize=(15, 3))
total_plots = 10
# Plot ground truth encoding steps
for i in range(total_plots):
    plt.subplot(1, total_plots, i+1)
    plt.imshow(simulation[i].numpy(), cmap='Greys', interpolation='none')
    plt.title(f"GT Step {i}")
    plt.axis('off')
plt.tight_layout()

# Visualize the input (ground truth) and model predictions in one window
total_plots = num_encoding + num_query
plt.figure(figsize=(15, 3))
# Plot ground truth encoding steps
for i in range(num_encoding):
    plt.subplot(1, total_plots, i+1)
    plt.imshow(simulation[i].numpy(), cmap='Greys', interpolation='none')
    plt.title(f"GT Step {i}")
    plt.axis('off')
# Plot model predictions for query steps
for idx, t in enumerate(query_steps):
    plt.subplot(1, total_plots, num_encoding+idx+1)
    plt.imshow(predicted_grids[idx], cmap='Greys', interpolation='none')
    plt.title(f"Pred Step {t}")
    plt.axis('off')
plt.tight_layout()
plt.show()


#print(standard_deviations[0])