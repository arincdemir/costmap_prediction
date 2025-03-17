import os
import torch
import time
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from CNN_CNMP import CNN_CNMP
from dataset import GridDataset  # see [CNMP_only/dataset.py]


# Hyperparameters
t_dim = 1                   # step index dimension
SM_dim = 16 ** 2            # grid flattened dimension (16*16)
encoder_hidden_dims = [216, 216, 216, 216]
latent_dim = 216             # adjust as required
decoder_hidden_dims = [216, 216, 216, 216]

batch_size = 2
num_epochs = 20
learning_rate = 0.0001


# Load data generated earlier
data_path = "ped_forecasting/CNMP_only/grids_tensor.pt"
grids_tensor = torch.load(data_path)

dataset = GridDataset(grids_tensor, 2, 2)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CNN_CNMP(
    t_dim=t_dim,
    SM_dim=SM_dim,
    encoder_hidden_dims=encoder_hidden_dims,
    decoder_hidden_dims=decoder_hidden_dims,
    latent_dim=latent_dim
)

for epoch in range(num_epochs):
    train_loss = 0.0
    for padded_encodings, encodings_mask, padded_query_indices, padded_query_targets, queries_mask in train_loader:
        """
        print(f"epoch: {epoch}")
        print(padded_encodings)
        print(encodings_mask)
        print(padded_query_indices)
        print(padded_query_targets)
        print(queries_mask)
        """
        output = model(padded_encodings, encodings_mask, padded_query_indices, queries_mask)
        loss = model.loss(output, padded_query_targets, queries_mask)
        #print(output)
        print(loss)
    