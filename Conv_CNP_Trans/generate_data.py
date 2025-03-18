import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

GRID_SIZE = 32
PEDESTRIAN_MAX_VEL = 1

class Pedestrian:
    def __init__(self, x: int, y: int, xVel: int, yVel: int, radius: int = 2):
        self.x = x
        self.y = y
        self.xVel = xVel
        self.yVel = yVel
        self.radius = radius

    def move(self) -> None:
        self.x += self.xVel
        self.y += self.yVel

    def is_inside_grid(self) -> bool:
        return 0 <= self.x < GRID_SIZE and 0 <= self.y < GRID_SIZE

def mark_pedestrian_on_grid(grid, ped: Pedestrian):
    if ped.is_inside_grid():
        cx, cy = int(ped.x), int(ped.y)
        r = ped.radius
        for i in range(max(0, cx - r), min(GRID_SIZE, cx + r + 1)):
            for j in range(max(0, cy - r), min(GRID_SIZE, cy + r + 1)):
                if (i - cx) ** 2 + (j - cy) ** 2 <= r ** 2:
                    grid[i][j] = 1

def simulate_multiple(pedestrians: list, steps: int):
    grids = []
    pedestrians_states = []  # list of pedestrian states per step, each is a list for all pedestrians

    for step in range(steps):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        step_states = []  # states for each pedestrian at current step
        new_pedestrians = []
        for ped in pedestrians:
            if ped is not None and ped.is_inside_grid():
                mark_pedestrian_on_grid(grid, ped)
                step_states.append((ped.x, ped.y, ped.xVel, ped.yVel, ped.radius))
                ped.move()
                new_pedestrians.append(ped)
            else:
                step_states.append(None)
                new_pedestrians.append(None)
        grids.append(grid)
        pedestrians_states.append(step_states)
        pedestrians = new_pedestrians

    grids_tensor = torch.tensor(grids, dtype=torch.float32)
    return grids_tensor, pedestrians_states

if __name__ == "__main__":
    observation_count = 10000
    num_pedestrians = 3
    all_grids = []
    all_states = []

    for i in range(observation_count):
        pedestrians = []
        for _ in range(num_pedestrians):
            x = random.randint(int(GRID_SIZE * 1/5), int((GRID_SIZE - 1) * 4/5))
            y = random.randint(int(GRID_SIZE * 1/3), int((GRID_SIZE - 1) * 2/3))
            while True:
                xVel = random.randint(-PEDESTRIAN_MAX_VEL, PEDESTRIAN_MAX_VEL)
                yVel = random.randint(-PEDESTRIAN_MAX_VEL, PEDESTRIAN_MAX_VEL)
                if xVel != 0 or yVel != 0:
                    break
            pedestrians.append(Pedestrian(x, y, xVel, yVel, radius=2))
            
        grids_tensor, states = simulate_multiple(pedestrians, 10)
        all_grids.append(grids_tensor)
        all_states.append(states)

    all_grids_tensor = torch.stack(all_grids)
    torch.save(all_grids_tensor, "grids_tensor.pt")
    print("Tensors saved to grids_tensor.pt")

    # Visualize a sample simulation sequence for the first observation.
    plt.figure(figsize=(15, 3))
    for step in range(10):
        ax = plt.subplot(1, 10, step + 1)
        ax.imshow(all_grids[0][step], cmap='Greys', interpolation='none')
        rect = patches.Rectangle((0, 0), GRID_SIZE - 1, GRID_SIZE - 1, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"Step {step}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()