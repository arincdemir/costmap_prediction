import os
import torch
import time
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from CNMP import CNMP  # see [CNMP_only/CNMP.py]
from dataset import GridDataset  # see [CNMP_only/dataset.py]
import wandb
import math
# Hyperparameters
t_dim = 1                   # step index dimension
SM_dim = 32 ** 2            # grid flattened dimension (16*16)
encoder_hidden_dims = [2048, 2048, 2048]
latent_dim = 1024
decoder_hidden_dims = [2048, 2048, 2048]

batch_size = 15
num_epochs = 100000
learning_rate = 0.001

wandb.init(
    project="ped_forecasting",
    config={"learning_rate": 0.0001,
            "architecture": "CNMP",
            "epochs": num_epochs,
            "encoder_hidden_dims": encoder_hidden_dims,
            "latent_dim": latent_dim,
            "decoder_hidden_dims": decoder_hidden_dims}
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load data generated earlier
data_path = "grids_tensor.pt"
grids_tensor = torch.load(data_path)

# Create the dataset and split into train and validation sets
dataset = GridDataset(grids_tensor, max_encodings=10, max_queries=10)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset = torch.utils.data.Subset(dataset, range(val_size, len(dataset)))
val_dataset = torch.utils.data.Subset(dataset, range(val_size))

# TODO I did this for trying to overfit the model with just one simulation.
#train_dataset = dataset
#val_dataset = dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print(f"Device: {device}")

# Initialize the model and optimizer
model = CNMP(
    t_dim=t_dim,
    SM_dim=SM_dim,
    encoder_hidden_dims=encoder_hidden_dims,
    decoder_hidden_dims=decoder_hidden_dims,
    latent_dim=latent_dim
).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
start_time = time.time()
best_val_loss = math.inf  # Initialize best validation loss

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    epoch_train_loss = 0.0
    for padded_encodings, encodings_mask, padded_query_indices, padded_query_targets, queries_mask in train_loader:
        # Move data to GPU
        padded_encodings = padded_encodings.to(device)
        encodings_mask = encodings_mask.to(device)
        padded_query_indices = padded_query_indices.to(device)
        padded_query_targets = padded_query_targets.to(device)
        queries_mask = queries_mask.to(device)
        
        optimizer.zero_grad()

        output = model(padded_encodings, encodings_mask, padded_query_indices, queries_mask)
        loss = model.loss(output, padded_query_targets, queries_mask)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    
    avg_train_loss = epoch_train_loss / len(train_dataset)
    
    # Add validation every 100 epochs
    if (epoch + 1) % 100 == 0:
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for padded_encodings, encodings_mask, padded_query_indices, padded_query_targets, queries_mask in val_loader:
                padded_encodings = padded_encodings.to(device)
                encodings_mask = encodings_mask.to(device)
                padded_query_indices = padded_query_indices.to(device)
                padded_query_targets = padded_query_targets.to(device)
                queries_mask = queries_mask.to(device)
                output = model(padded_encodings, encodings_mask, padded_query_indices, queries_mask)
                loss = model.loss(output, padded_query_targets, queries_mask)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_dataset)
        wandb.log({"validation_loss": avg_val_loss})
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "trained_cnmp_best.pth")
            print(f"New best model found and saved with validation loss: {best_val_loss:.4f}")


    epoch_duration = time.time() - epoch_start_time
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / (epoch + 1) * num_epochs
    estimated_time_left = estimated_total_time - elapsed_time
    wandb.log({"loss": avg_train_loss})
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f} Time Left: {estimated_time_left:.2f} seconds")
# Save the trained model
wandb.finish()
torch.save(model.state_dict(), "trained_cnmp.pth")
print("Training complete. Model saved to trained_cnmp.pth")
print(f"Best validation loss: {best_val_loss}")