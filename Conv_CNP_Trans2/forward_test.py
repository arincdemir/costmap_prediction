import torch
import torch.nn as nn
from Conv_CNMP.CNN_CNMP import CNMP

# Define dimensions
t_dim = 10
SM_dim = 20
hidden_dim = 50
latent_dim = 30

# Create a CNMP instance
model = CNMP(t_dim, SM_dim, hidden_dim, latent_dim)

# Create some dummy data
batch_size = 5
encode_size = 7

encoding_inputs = torch.randn(batch_size, encode_size, t_dim + SM_dim)
t_query = torch.randn(batch_size, t_dim)
true_SM = torch.randn(batch_size, SM_dim)

# Run the forward pass
output = model.forward(encoding_inputs, t_query)

# Calculate the custom loss
loss = model.loss(output, true_SM)
print("Loss:", loss.item())