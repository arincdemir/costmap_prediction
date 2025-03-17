import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from Conv_MLP import ConvMLPConv

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 300
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

def train():
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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use ConvMLPConv directly with input_channels=3 and output_channels=1
    model = ConvMLPConv(
        conv_channels=[16, 32, 64],
        mlp_features=[512, 512],
        input_channels=3,  # 3 input channels from three frames
        output_channels=1, # Output a single channel
        input_size=GRID_SIZE
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)    # shape: (B, 3, 32, 32)
            targets = targets.to(device)  # shape: (B, 1, 32, 32)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {avg_loss:.6f}")

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
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), "conv_mlp_trained.pth")
    print("Training complete. Model saved to conv_mlp_trained.pth.")

if __name__ == "__main__":
    train()