import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from CNN_CNMP import CNN_CNMP

# Hyperparameters (must match training)
grid_size = 32
max_encodings = 5  # as used in the model and dataset
max_queries = 5    # new parameter for querying

# Hyperparameters
t_dim = 1                      # step index dimension
grid_size = 32                 # grid size (32x32)

# Try simpler architecture first
cnn_channels = [16, 32, 64]  # Reduced complexity
encoder_hidden_dims = [256, 128]  # Simplified
latent_dim = 128  # Smaller latent space
decoder_hidden_dims = [256, 512]  # Simplified


# Initialize the CNN_CNMP model
model = CNN_CNMP(
    t_dim=t_dim,
    grid_size=grid_size,
    encoder_hidden_dims=encoder_hidden_dims,
    decoder_hidden_dims=decoder_hidden_dims,
    latent_dim=latent_dim,
    cnn_channels=cnn_channels
)
model_path = os.path.join(os.path.dirname(__file__), "trained_model_best.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Load simulation data (all grids)
data_path = os.path.join(os.path.dirname(__file__), "grids_tensor.pt")
all_grids_tensor = torch.load(data_path, map_location=torch.device("cpu"))

# Pick one simulation sample (here using the last simulation)
simulation = all_grids_tensor[-3]  # shape: (steps, grid_size, grid_size)
steps = simulation.shape[0]

# Set encoding and query lengths
num_encoding = 5
num_query = 5

# Prepare encoding inputs: separate time indices and grids and create mask
padded_time_indices = torch.zeros(max_encodings, 1, dtype=torch.float32)
padded_grids = torch.zeros(max_encodings, grid_size, grid_size, dtype=torch.float32)
encodings_mask = torch.zeros(max_encodings, dtype=torch.bool)

for i in range(num_encoding):
    # Normalize time index using total steps
    padded_time_indices[i, 0] = float(i) / steps
    padded_grids[i] = simulation[i]
    encodings_mask[i] = True

# Prepare query inputs: time steps from num_encoding to num_encoding+num_query-1
query_steps = list(range(num_encoding, num_encoding + num_query))
padded_query_indices = (torch.tensor(query_steps, dtype=torch.float32).unsqueeze(1) / steps)  # shape: (num_query, 1)
queries_mask = torch.ones(num_query, dtype=torch.bool)

# Query the model for the next steps using the CNN_CNMP forward interface
with torch.no_grad():
    # Adding batch dimension for each input
    output = model(
        padded_time_indices.unsqueeze(0),  # (1, max_encodings, t_dim)
        padded_grids.unsqueeze(0),         # (1, max_encodings, grid_size, grid_size)
        encodings_mask.unsqueeze(0),       # (1, max_encodings)
        padded_query_indices.unsqueeze(0), # (1, num_query, t_dim)
        queries_mask.unsqueeze(0)          # (1, num_query)
    )

# New model output shape: (1, num_query, grid_size, grid_size)
predicted_grids = output[0].cpu().numpy()

# Plot ground truth encoding steps with grid outlines
plt.figure(figsize=(15, 3))
total_plots = 10
for i in range(total_plots):
    ax = plt.subplot(1, total_plots, i+1)
    ax.imshow(simulation[i].numpy(), cmap='Greys', interpolation='none')
    ax.set_title(f"GT Step {i}")
    # Add rectangle outline
    rect = patches.Rectangle(
        (-0.5, -0.5), grid_size, grid_size, 
        linewidth=1, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)
    ax.axis('off')
plt.tight_layout()

# Visualize both the ground truth and model predictions with grid outlines
total_plots = num_encoding + num_query
plt.figure(figsize=(15, 3))
for i in range(num_encoding):
    ax = plt.subplot(1, total_plots, i+1)
    ax.imshow(simulation[i].numpy(), cmap='Greys', interpolation='none')
    ax.set_title(f"GT Step {i}")
    rect = patches.Rectangle(
        (-0.5, -0.5), grid_size, grid_size,
        linewidth=1, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)
    ax.axis('off')

for idx, t in enumerate(query_steps):
    ax = plt.subplot(1, total_plots, num_encoding + idx + 1)
    ax.imshow(predicted_grids[idx], cmap='Greys', interpolation='none')
    ax.set_title(f"Pred Step {t}")
    rect = patches.Rectangle(
        (-0.5, -0.5), grid_size, grid_size,
        linewidth=1, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)
    ax.axis('off')
plt.tight_layout()
plt.show()