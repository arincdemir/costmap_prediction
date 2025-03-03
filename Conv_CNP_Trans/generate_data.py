import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import os

# The size of the output captured grid (the middle part)
CAPTURED_GRID_SIZE = 32
# The full simulation environment is 3x3 of the captured grid (i.e. 9x area)
ENV_SIZE = CAPTURED_GRID_SIZE

PEDESTRIAN_MAX_VEL = 1
PEDESTRIAN_COUNT = 3
PEDESTRIAN_RADIUS = 2

class Pedestrian:
    def __init__(self, x: int, y: int, xVel: int, yVel: int):
        self.x = x
        self.y = y
        self.xVel = xVel
        self.yVel = yVel
    
    def move(self) -> None:
        # Compute tentative new positions
        new_x = self.x + self.xVel
        new_y = self.y + self.yVel

        # Bounce off the vertical boundaries
        if new_x < PEDESTRIAN_RADIUS or new_x >= ENV_SIZE - PEDESTRIAN_RADIUS:
            self.xVel = -self.xVel
            new_x = self.x + self.xVel

        # Bounce off the horizontal boundaries
        if new_y < PEDESTRIAN_RADIUS or new_y >= ENV_SIZE - PEDESTRIAN_RADIUS:
            self.yVel = -self.yVel
            new_y = self.y + self.yVel

        self.x = new_x
        self.y = new_y

def plot_grid(grid, step):
    plt.imshow(grid, cmap='Greys', interpolation='none')
    plt.title(f"Step {step}")
    plt.show()

def mark_pedestrian_on_grid(grid, ped):
    for i in range(max(0, ped.x - PEDESTRIAN_RADIUS), min(ENV_SIZE, ped.x + PEDESTRIAN_RADIUS + 1)):
        for j in range(max(0, ped.y - PEDESTRIAN_RADIUS), min(ENV_SIZE, ped.y + PEDESTRIAN_RADIUS + 1)):
            if (i - ped.x) ** 2 + (j - ped.y) ** 2 <= PEDESTRIAN_RADIUS ** 2:
                grid[i][j] = 1

def simulate(pedestrians: set[Pedestrian], steps: int):
    grids = []
    for step_count in range(steps):
        # Create a grid for the captured environment directly
        grid = [[0 for j in range(ENV_SIZE)] for i in range(ENV_SIZE)]
        
        for ped in pedestrians:
            mark_pedestrian_on_grid(grid, ped)
            ped.move()
        
        grids.append(grid)

    grids_tensor = torch.tensor(grids, dtype=torch.float32)
    return grids_tensor

# Define random initial positions in the full environment
x_min, x_max = int(ENV_SIZE * 1 / 5), int(ENV_SIZE * 4 / 5)
y_min, y_max = int(ENV_SIZE * 1 / 5), int(ENV_SIZE * 4 / 5)

if __name__ == "__main__":
    observation_count = 2560
    all_grids = []
    for i in range(observation_count):
        pedestrians = set()
        for _ in range(PEDESTRIAN_COUNT):
            x, y = random.randint(x_min, x_max), random.randint(y_min, y_max)
            xVel = random.choice([-PEDESTRIAN_MAX_VEL, PEDESTRIAN_MAX_VEL])
            yVel = random.choice([-PEDESTRIAN_MAX_VEL, PEDESTRIAN_MAX_VEL])
            pedestrians.add(Pedestrian(x, y, xVel, yVel))
        
        grids_tensor = simulate(pedestrians, 10) # no step
        all_grids.append(grids_tensor)
    
    # Stack all tensors into one tensor
    all_grids_tensor = torch.stack(all_grids)
    output_path = os.path.join(os.path.dirname(__file__), "grids_tensor.pt")
    torch.save(all_grids_tensor, output_path)
    print("All tensors saved to grids_tensor.pt")

    if (True):
        plt.figure(figsize=(15, 3))
        total_plots = 10
        # Plot ground truth encoding steps
        for i in range(total_plots):
            ax = plt.subplot(1, total_plots, i+1)
            ax.imshow(all_grids_tensor[0,i].numpy(), cmap='Greys', interpolation='none')
            # Add an outline rectangle: modify edgecolor and linewidth as desired
            rect = patches.Rectangle((-1, -1), ENV_SIZE + 1, ENV_SIZE + 1, 
                                     linewidth=1, edgecolor='black', facecolor='none', 
                                     clip_on=False)
            ax.add_patch(rect)
            ax.set_title(f"GT Step {i}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()