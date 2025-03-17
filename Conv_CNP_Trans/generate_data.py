import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import os

# The size of the output captured grid (the visible part)
CAPTURED_GRID_SIZE = 32
# The full simulation environment is larger than the captured grid
ENV_SIZE = CAPTURED_GRID_SIZE * 3  # 3x larger in each dimension

PEDESTRIAN_MAX_VEL = 1
PEDESTRIAN_COUNT = 25  # Increased to have more activity
PEDESTRIAN_RADIUS = 2

class Pedestrian:
    def __init__(self, x: int, y: int, xVel: int, yVel: int):
        self.x = x
        self.y = y
        self.xVel = xVel
        self.yVel = yVel
    
    def move(self) -> None:
        # Update position
        self.x = self.x + self.xVel
        self.y = self.y + self.yVel
        
        # Apply wraparound boundary conditions to the full environment
        self.x = self.x % ENV_SIZE
        self.y = self.y % ENV_SIZE

def plot_grid(grid, step):
    plt.imshow(grid, cmap='Greys', interpolation='none')
    plt.title(f"Step {step}")
    plt.show()

def mark_pedestrian_on_grid(grid, ped):
    for i in range(ped.x - PEDESTRIAN_RADIUS, ped.x + PEDESTRIAN_RADIUS + 1):
        for j in range(ped.y - PEDESTRIAN_RADIUS, ped.y + PEDESTRIAN_RADIUS + 1):
            # Apply wraparound to handle circle drawing at boundaries
            wrapped_i = i % ENV_SIZE
            wrapped_j = j % ENV_SIZE
            
            # Check if the point is within the circular pedestrian shape
            if (i - ped.x) ** 2 + (j - ped.y) ** 2 <= PEDESTRIAN_RADIUS ** 2:
                grid[wrapped_i][wrapped_j] = 1

def get_visible_grid(full_grid):
    # Calculate the offset to get the center portion
    offset = (ENV_SIZE - CAPTURED_GRID_SIZE) // 2
    
    # Extract the center portion of the full grid
    visible = np.array(full_grid)[offset:offset+CAPTURED_GRID_SIZE, 
                                  offset:offset+CAPTURED_GRID_SIZE].tolist()
    return visible

def simulate(pedestrians: set[Pedestrian], steps: int):
    visible_grids = []
    pedestrian_states = []  # Store pedestrian positions and velocities at each step
    
    for step_count in range(steps):
        # Create a grid for the full environment
        full_grid = [[0 for j in range(ENV_SIZE)] for i in range(ENV_SIZE)]
        
        # Store current pedestrian states
        step_states = []
        for ped in pedestrians:
            mark_pedestrian_on_grid(full_grid, ped)
            step_states.append((ped.x, ped.y, ped.xVel, ped.yVel))
        
        # Get the visible portion of the grid
        visible_grid = get_visible_grid(full_grid)
        visible_grids.append(visible_grid)
        pedestrian_states.append(step_states)
        
        # Move pedestrians for the next step
        for ped in pedestrians:
            ped.move()

    # Convert to tensor
    visible_grids_tensor = torch.tensor(visible_grids, dtype=torch.float32)
    return visible_grids_tensor, pedestrian_states

if __name__ == "__main__":
    observation_count = 10240
    all_grids = []
    all_pedestrian_states = []
    
    for i in range(observation_count):
        # Initialize pedestrians distributed throughout the full environment
        pedestrians = set()
        for _ in range(PEDESTRIAN_COUNT):
            x = random.randint(0, ENV_SIZE - 1)
            y = random.randint(0, ENV_SIZE - 1)
            
            # Give velocities that ensure movement (no zero velocities)
            while True:
                xVel = random.randint(-PEDESTRIAN_MAX_VEL, PEDESTRIAN_MAX_VEL)
                yVel = random.randint(-PEDESTRIAN_MAX_VEL, PEDESTRIAN_MAX_VEL)
                if xVel != 0 or yVel != 0:  # Ensure some movement
                    break
                    
            pedestrians.add(Pedestrian(x, y, xVel, yVel))
        
        grids_tensor, pedestrian_states = simulate(pedestrians, 10)
        all_grids.append(grids_tensor)
        all_pedestrian_states.append(pedestrian_states)
    
    # Stack all tensors into one tensor
    all_grids_tensor = torch.stack(all_grids)
    output_path = os.path.join(os.path.dirname(__file__), "grids_tensor.pt")
    torch.save(all_grids_tensor, output_path)
    print("All tensors saved to grids_tensor.pt")

    # Visualize a sample sequence
    if True:
        plt.figure(figsize=(15, 8))
        total_plots = 3   # Number of plots to show
        step_interval = 3 # Skip steps in between

        # Plot sequence steps
        for plot_index in range(total_plots):
            step = plot_index * step_interval  # Calculate the step index to display
            ax = plt.subplot(1, total_plots, plot_index+1)

            # Create the full grid for this timestep
            full_grid = np.zeros((ENV_SIZE, ENV_SIZE))
            for x, y, xVel, yVel in all_pedestrian_states[0][step]:
                # Mark pedestrians on the full grid
                for di in range(-PEDESTRIAN_RADIUS, PEDESTRIAN_RADIUS + 1):
                    for dj in range(-PEDESTRIAN_RADIUS, PEDESTRIAN_RADIUS + 1):
                        wrapped_i = (x + di) % ENV_SIZE
                        wrapped_j = (y + dj) % ENV_SIZE
                        if di**2 + dj**2 <= PEDESTRIAN_RADIUS**2:
                            full_grid[wrapped_i, wrapped_j] = 1

            # Calculate the offset to determine captured area
            offset = (ENV_SIZE - CAPTURED_GRID_SIZE) // 2

            # Create a mask for highlighting the captured area
            mask = np.ones_like(full_grid) * 0.4  # Dim factor for non-captured area
            mask[offset:offset+CAPTURED_GRID_SIZE, offset:offset+CAPTURED_GRID_SIZE] = 1.0  # Normal brightness for captured area

            # Display the full grid with the mask applied
            ax.imshow(full_grid * mask, cmap='Greys', interpolation='none')

            # Draw a border around the captured area
            rect = patches.Rectangle((offset-0.5, offset-0.5), 
                                    CAPTURED_GRID_SIZE, 
                                    CAPTURED_GRID_SIZE,
                                    linewidth=2, 
                                    edgecolor='red', 
                                    facecolor='none')
            ax.add_patch(rect)

            # Add a black border around the entire environment
            outer_border = patches.Rectangle((-0.5, -0.5),
                                            ENV_SIZE,
                                            ENV_SIZE,
                                            linewidth=2,
                                            edgecolor='black',
                                            facecolor='none')
            ax.add_patch(outer_border)

            # Draw velocity arrows for all pedestrians
            for x, y, xVel, yVel in all_pedestrian_states[0][step]:
                ax.arrow(y, x, yVel * 3, xVel * 3, 
                        color='#33d1ff', width=0.5, head_width=1.5, head_length=1.5,
                        length_includes_head=True)

            ax.set_title(f"Step {step}")
            ax.set_xlim(-0.5, ENV_SIZE - 0.5)
            ax.set_ylim(ENV_SIZE - 0.5, -0.5)  # Reverse y-axis to match imshow orientation
            ax.axis('off')

        plt.tight_layout()
        plt.show()