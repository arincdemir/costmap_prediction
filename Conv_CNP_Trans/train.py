import os
import sys
import torch
import time
from torch.utils.data import DataLoader
import torch.optim as optim
from CNN_CNMP import CNN_CNMP
from dataset import GridDataset
import wandb
import math

# Hyperparameters
t_dim = 1                      # step index dimension
grid_size = 32                 # grid size (32x32)

# Try simpler architecture first
cnn_channels = [16, 32, 64]  # Reduced complexity
encoder_hidden_dims = [256, 128]  # Simplified
latent_dim = 128  # Smaller latent space
decoder_hidden_dims = [256, 512]  # Simplified

dropout_rate = 0.2
batch_size = 128
num_epochs = 30000
learning_rate = 0.00036

early_stopping_patience = 20  # Number of epochs to wait before stopping
early_stopping_min_delta = 0.000001  # Minimum change to qualify as an improvement
early_stopping_counter = 0  # Counter for patience

scheduler_patience = 5
scheduler_factor = 0.7


wandb.init(
    project="ped_forecasting",
    config={
        "learning_rate": learning_rate,
        "architecture": "CNN_CNMP",
        "epochs": num_epochs,
        "batch_siize": batch_size,
        "encoder_hidden_dims": encoder_hidden_dims,
        "latent_dim": latent_dim,
        "decoder_hidden_dims": decoder_hidden_dims,
        "cnn_channels": cnn_channels,
        "grid_size": grid_size,
        "dropout_rate": dropout_rate,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data generated earlier
data_path = "grids_tensor.pt"
grids_tensor = torch.load(data_path)

# Create the dataset and split into train and validation sets
dataset = GridDataset(grids_tensor, max_encodings=5, max_queries=5)
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

# Initialize the model and optimizer with the CNN component
model = CNN_CNMP(
    t_dim=t_dim,
    grid_size=grid_size,
    encoder_hidden_dims=encoder_hidden_dims,
    decoder_hidden_dims=decoder_hidden_dims,
    latent_dim=latent_dim,
    cnn_channels=cnn_channels,
    dropout_rate=dropout_rate  # Add dropout parameter
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    patience=scheduler_patience,       # Wait longer before reducing
    factor=scheduler_factor,        # Reduce LR by smaller amount (30% reduction)
    min_lr=1e-6,       # Don't let LR go below this
)

start_time = time.time()
best_val_loss = math.inf  # Initialize best validation loss

try:
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_train_loss = 0.0

        for padded_time_indices, padded_grids, encodings_mask, padded_query_indices, padded_query_targets, queries_mask in train_loader:
            # Move data to GPU
            padded_time_indices = padded_time_indices.to(device)
            padded_grids = padded_grids.to(device)
            encodings_mask = encodings_mask.to(device)
            padded_query_indices = padded_query_indices.to(device)
            padded_query_targets = padded_query_targets.to(device)
            queries_mask = queries_mask.to(device)

            optimizer.zero_grad()
            # Forward pass with the new model interface
            output = model(padded_time_indices, padded_grids, encodings_mask, padded_query_indices, queries_mask)
            loss = model.loss(output, padded_query_targets, queries_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        if (epoch + 1) % 5 == 0:
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for padded_time_indices, padded_grids, encodings_mask, padded_query_indices, padded_query_targets, queries_mask in val_loader:
                    padded_time_indices = padded_time_indices.to(device)
                    padded_grids = padded_grids.to(device)
                    encodings_mask = encodings_mask.to(device)
                    padded_query_indices = padded_query_indices.to(device)
                    padded_query_targets = padded_query_targets.to(device)
                    queries_mask = queries_mask.to(device)
                    
                    output = model(padded_time_indices, padded_grids, encodings_mask, padded_query_indices, queries_mask)
                    loss = model.loss(output, padded_query_targets, queries_mask)
                    epoch_val_loss += loss.item()
                    
            avg_val_loss = epoch_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            wandb.log({"validation_loss": avg_val_loss}, step=epoch)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.8f}")
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr}, step=epoch)
            # Early stopping check
            if avg_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "trained_model_best.pth")
                print(f"New best model found and saved with validation loss: {best_val_loss:.8f}")
                early_stopping_counter = 0  # Reset counter
            else:
                early_stopping_counter += 1
                print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                
            # Check if should stop training
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {early_stopping_patience} validation checks.")
                break  # Exit the training loop


        epoch_duration = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (epoch + 1) * num_epochs
        estimated_time_left = estimated_total_time - elapsed_time
        
        wandb.log({"train_loss": avg_train_loss}, step=epoch)
        if epoch > 20:
            wandb.log({"time_left": round(estimated_time_left)}, step=epoch)
    
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.8f} Time Left: {estimated_time_left:.2f} seconds")

except KeyboardInterrupt:
    print("Training interrupted. Saving current model state...")
    torch.save(model.state_dict(), "trained_model_interrupt.pth")
    print("Model state saved. Exiting training loop.")
    print(f"Best validation loss: {best_val_loss}")
    
    try:
        wandb.finish()  # Try to finish wandb gracefully
    except BrokenPipeError:
        print("Handled BrokenPipeError during wandb.finish()")
    except Exception as e:
        print(f"Error finishing wandb: {e}")
    
try:
    wandb.finish()
except BrokenPipeError:
    print("Warning: BrokenPipeError caught during wandb.finish(). Continuing to save the model...")

torch.save(model.state_dict(), "trained_model.pth")
print("Training complete. Model saved to trained_model.pth")
print(f"Best validation loss: {best_val_loss}")