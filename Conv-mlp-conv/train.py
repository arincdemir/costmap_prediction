import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import wandb
import argparse

from Conv_MLP import ConvMLPConv

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_INPUT_FRAMES = 3
GRID_SIZE = 32

# Dataset: for each observation (a simulation sequence of 10 frames), extract sliding windows:
# input: frames [i, i+1, i+2] (stacked as channels) and target: frame at i+3.
class GridDataset(Dataset):
    def __init__(self, tensor_file):
        # grid_tensor shape: (num_obs, 10, 32, 32)
        if not os.path.exists(tensor_file):
            raise FileNotFoundError(f"{tensor_file} not found.")
        grids = torch.load(tensor_file)
        self.samples = []
        # For each simulation, create (3->target) sliding windows.
        for sim in grids:  # sim shape: (10, 32, 32)
            # Ensure sim is a tensor of type float32
            sim = sim.float()
            # Create samples only if sim has at least 4 frames.
            if sim.shape[0] >= NUM_INPUT_FRAMES + 1:
                # Slide over time: 0 to (10 - (3+1)) inclusive
                for i in range(sim.shape[0] - NUM_INPUT_FRAMES):
                    # Input: stack three frames as channels -> shape (3, 32, 32)
                    inp = sim[i:i+NUM_INPUT_FRAMES]
                    # Target: next frame -> shape (32, 32); we will add channel dim later.
                    target = sim[i+NUM_INPUT_FRAMES]
                    self.samples.append((inp, target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        inp, target = self.samples[idx]
        # inputs are (3,32,32), target add channel dim to be (1,32,32)
        return inp, target.unsqueeze(0)

def train(config=None):
    # Initialize wandb
    with wandb.init(project="costmap_prediction", config=config) as run:
        # Access hyperparameters through wandb.config
        batch_size = wandb.config.batch_size
        num_epochs = wandb.config.num_epochs
        learning_rate = wandb.config.learning_rate
        model_output_name = wandb.config.get('model_output_name', run.id)
        
        # Load dataset
        full_dataset = GridDataset("grids_tensor.pt")
        
        # Split dataset: first 20% for validation, last 80% for training
        total_samples = len(full_dataset)
        val_size = int(0.2 * total_samples)
        train_size = total_samples - val_size

        # Create subsets based on indices
        indices = list(range(total_samples))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Use ConvMLPConv directly with input_channels=3 and output_channels=1
        model = ConvMLPConv(
            conv_channels=wandb.config.conv_channels,
            mlp_features=wandb.config.mlp_features,
            input_channels=3,  # 3 input channels from three frames
            output_channels=1, # Output a single channel
            input_size=GRID_SIZE
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Initialize best validation loss tracker
        best_val_loss = float('inf')
        best_model_path = f"best_model_{model_output_name}.pth"
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)    # shape: (B, 3, 32, 32)
                targets = targets.to(device)  # shape: (B, 1, 32, 32)
                
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, targets)
                loss.backward()
                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            wandb.log({"train_loss": avg_loss}, step=epoch)
            
            # Estimate remaining time
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (epoch + 1) * num_epochs
            estimated_time_left = estimated_total_time - elapsed_time
            
            if epoch > 10:
                wandb.log({"time_left": round(estimated_time_left)}, step=epoch)
                
            # Every 10 epochs, evaluate on the validation set.
            if (epoch + 1) % 10 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        output = model(inputs)
                        loss = criterion(output, targets)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                wandb.log({"validation_loss": avg_val_loss}, step=epoch)
                
                # Save best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved with validation loss: {best_val_loss:.8f}")
                
                print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.6f}")
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.6f}, Time Left: {estimated_time_left:.2f} seconds")

        # Save the final trained model
        final_model_path = f"conv_mlp_{model_output_name}.pth"
        torch.save(model.state_dict(), final_model_path)
        
        # Log artifacts to wandb
        artifact = wandb.Artifact(f"model-{run.id}", type="model")
        artifact.add_file(best_model_path)
        run.log_artifact(artifact)
        
        print(f"Training complete. Best model (val loss: {best_val_loss:.6f}) saved to {best_model_path}")
        print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvMLPConv model")
    parser.add_argument('--model_name', type=str, default="default_run", help='Name for the saved model')
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'conv_channels': [16, 32, 64],
        'mlp_features': [512, 512],
        'model_output_name': args.model_name
    }
    
    train(config)