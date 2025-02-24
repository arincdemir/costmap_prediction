import torch
import matplotlib.pyplot as plt
from GridCNN import GridCNN
import random
import time

def load_grids(path):
    return torch.load(path)

def load_model(model_path):
    model = GridCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_next_grid(model, input_grids):
    with torch.no_grad():
        prediction = model(input_grids.unsqueeze(0))
    return prediction.squeeze().cpu().numpy()

def plot_grids(input_grids, ground_truth, predicted_grid):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ['Grid 1', 'Grid 2', 'Grid 3', 'Ground Truth', 'Predicted']
    for i in range(3):
        axes[i].imshow(input_grids[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    axes[3].imshow(ground_truth, cmap='gray')
    axes[3].set_title(titles[3])
    axes[3].axis('off')
    axes[4].imshow(predicted_grid, cmap='gray')
    axes[4].set_title(titles[4])
    axes[4].axis('off')
    plt.show()

def main():
    grids_tensor = load_grids("./ped_forecasting/grids_tensor.pt")
    model = load_model("./ped_forecasting/grid_cnn.pth")
    
    while True:
        percentile_20_index = int(0.2 * len(grids_tensor)) # since we trained from the last 0.8 part, we can test from this part
        random_simulation = random.choice(grids_tensor[:percentile_20_index])
        input_grids = random_simulation[:3].cpu().numpy()
        ground_truth = random_simulation[-1].cpu().numpy()
        
        predicted_grid = predict_next_grid(model, random_simulation[:3])
        print(predicted_grid.shape)
        
        plot_grids(input_grids, ground_truth, predicted_grid)
        time.sleep(1)  # Delay for 5 seconds before showing the next figure

if __name__ == "__main__":
    main()