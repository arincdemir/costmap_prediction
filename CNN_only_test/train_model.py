import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
from tqdm import tqdm

from GridCNN import GridCNN

class GridDataset(Dataset):
    def __init__(self, grids_tensor):
        # grids_tensor shape: [observation_count, steps, grid_size, grid_size]
        self.inputs = grids_tensor[:, :3, :, :]  # First two grids
        self.targets = grids_tensor[:, -1, :, :]  # Last grid

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # Do not add channel dimension to inputs
        input_grid = self.inputs[idx]  # Shape: [2, 16, 16]
        
        # Add channel dimension to targets
        target_grid = self.targets[idx].unsqueeze(0)
        
        return input_grid, target_grid

if __name__ == "__main__":

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the tensor from the file
    grids_tensor = torch.load('./ped_forecasting/grids_tensor.pt')
    
    # Ensure there are at least 3 steps
    assert grids_tensor.shape[1] >= 3, "Each simulation must have at least 3 steps."
    
    # Create the dataset
    dataset = GridDataset(grids_tensor)
    
    # Get the first 1000 observations
    subset_size = min(1000, len(dataset))
    subset_indices = range(subset_size)
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    # Split the subset dataset into training and testing sets
    train_size = int(0.8 * len(subset_dataset))
    test_size = len(subset_dataset) - train_size

    test_dataset = torch.utils.data.Subset(subset_dataset, range(test_size))
    train_dataset = torch.utils.data.Subset(subset_dataset, range(test_size, len(subset_dataset)))
        
        
    # Create DataLoader for training and testing
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
    
    # Initialize the model
    model = GridCNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss().to(device)  # Move criterion to device if necessary
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 200
    
    try:
        for epoch in range(num_epochs):
            running_loss = 0.0
            start_time = time.time()
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs) 
                
                # Compute loss
                loss = criterion(outputs, targets)  
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataset)
            elapsed_time = time.time() - start_time
            estimated_time_remaining = elapsed_time * (num_epochs - epoch - 1)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time elapsed: {elapsed_time:.2f}s, Estimated time remaining: {estimated_time_remaining:.2f}s")

    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_dataset)
    print(f"Test Loss: {test_loss:.4f}")
        
    # Save the trained model
    torch.save(model.state_dict(), './ped_forecasting/grid_cnn.pth')
    print("Model trained and saved to grid_cnn.pth")