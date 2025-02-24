import numpy as np
import random
import matplotlib.pyplot as plt
import torch



GRID_SIZE = 64
PEDESTRIAN_MAX_VEL = 3
PEDESTRIAN_COUNT = 4
PEDESTRIAN_RADIUS = 4

class Pedestrian:
    def __init__(self, x: int, y: int, xVel: int, yVel: int):
        self.x = x
        self.y = y
        self.xVel = xVel
        self.yVel = yVel
    
    def move(self) -> bool: # returns true if need to be removed
        self.x += self.xVel
        self.y += self.yVel
        if self.x < 0 or self.x >= GRID_SIZE or self.y < 0 or self.y >= GRID_SIZE:
            return True
        else:
            return False

def plot_grid(grid, step):
    plt.imshow(grid, cmap='Greys', interpolation='none')
    plt.title(f"Step {step}")
    plt.show()

def mark_pedestrian_on_grid(grid, ped):
    for i in range(max(0, ped.x - PEDESTRIAN_RADIUS), min(GRID_SIZE, ped.x + PEDESTRIAN_RADIUS + 1)):
        for j in range(max(0, ped.y - PEDESTRIAN_RADIUS), min(GRID_SIZE, ped.y + PEDESTRIAN_RADIUS + 1)):
            if (i - ped.x) ** 2 + (j - ped.y) ** 2 <= PEDESTRIAN_RADIUS ** 2:
                grid[i][j] = 1

def simulate(pedestrians: set[Pedestrian], steps: int):
    grids = []
    for step_count in range(steps):
        grid = [[0 for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]
        ped_to_remove: set[Pedestrian] = set()
        for ped in pedestrians:
            mark_pedestrian_on_grid(grid, ped)
            if ped.move():
                ped_to_remove.add(ped)
        for ped in ped_to_remove:
            pedestrians.remove(ped)
        
        grids.append(grid)
        plot_grid(grid, step_count)  # Add this line to plot the grid at each step

    grids_tensor = torch.tensor(grids, dtype=torch.float32)
    return grids_tensor

if __name__ == "__main__":
    observation_count = 10000
    all_grids = []
    for i in range(observation_count):
        pedestrians = set()
        for _ in range(PEDESTRIAN_COUNT):
            x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            xVel, yVel = random.choice([-PEDESTRIAN_MAX_VEL, 0, PEDESTRIAN_MAX_VEL]), random.choice([-PEDESTRIAN_MAX_VEL, 0, PEDESTRIAN_MAX_VEL])
            pedestrians.add(Pedestrian(x, y, xVel, yVel))
        
        grids_tensor = simulate(pedestrians, 6)  # Ensure this generates at least 3 steps
        all_grids.append(grids_tensor)
    
    # Stack all tensors into one tensor
    all_grids_tensor = torch.stack(all_grids)
    torch.save(all_grids_tensor, "./ped_forecasting/grids_tensor.pt")
    print("All tensors saved to grids_tensor.pt")