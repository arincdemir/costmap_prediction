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
import argparse
from visualization_utils import log_visualizations_to_wandb



def train(config=None):
    """
    Training function that can be used with wandb sweeps
    """
    # Initialize wandb with either the sweep config or default config
    with wandb.init(project="ped_forecasting", config=config) as run:
        # Access all hyperparameters through wandb.config
        t_dim = wandb.config.t_dim
        grid_size = wandb.config.grid_size
        cnn_channels = wandb.config.cnn_channels
        encoder_hidden_dims = wandb.config.encoder_hidden_dims
        latent_dim = wandb.config.latent_dim
        decoder_hidden_dims = wandb.config.decoder_hidden_dims
        dropout_rate = wandb.config.dropout_rate
        batch_size = wandb.config.batch_size
        num_epochs = wandb.config.num_epochs
        learning_rate = wandb.config.learning_rate
        dataset_size = wandb.config.dataset_size
        dataset_population_factor = wandb.config.dataset_population_factor
        # Removed early stopping parameters:
        # early_stopping_patience = wandb.config.early_stopping_patience
        # early_stopping_min_delta = wandb.config.early_stopping_min_delta
        
        # Use run.id as model output name if not provided in config
        model_output_name_addition = wandb.config.get('model_output_name_addition', run.id)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data generated earlier
        data_path = "grids_tensor.pt"
        grids_tensor = torch.load(data_path)

        # Create the dataset and split into train and validation sets
        dataset = GridDataset(grids_tensor, max_encodings=5, max_queries=5, populate_factor=dataset_population_factor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset = torch.utils.data.Subset(dataset, range(val_size, len(dataset)))
        val_dataset = torch.utils.data.Subset(dataset, range(val_size))


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f"Device: {device}")
        print(f"Learning rate: {learning_rate}")
        print(f"Run ID: {run.id}")

        # Initialize the model
        model = CNN_CNMP(
            t_dim=t_dim,
            grid_size=grid_size,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            latent_dim=latent_dim,
            cnn_channels=cnn_channels,
            dropout_rate=dropout_rate
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        start_time = time.time()
        
        # Initialize best validation loss tracker
        best_val_loss = float('inf')
        best_model_path = f"best_model_{model_output_name_addition}.pth"
        

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
                output = model(padded_time_indices, padded_grids, encodings_mask, padded_query_indices, queries_mask)
                loss = model.loss(output, padded_query_targets, queries_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            wandb.log({"train_loss": avg_train_loss}, step=epoch)

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
                wandb.log({"validation_loss": avg_val_loss}, step=epoch)
                print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.8f}")
                
                # Save best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # Save the best model locally
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved with validation loss: {best_val_loss:.8f}")


            # Estimate remaining time
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (epoch + 1) * num_epochs
            estimated_time_left = estimated_total_time - elapsed_time
            
            if epoch > 20:
                wandb.log({"time_left": round(estimated_time_left)}, step=epoch)
        
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.8f} Time Left: {estimated_time_left:.2f} seconds")

        # Save final model
        torch.save(model.state_dict(), f"trained_model_{model_output_name_addition}.pth")
        print(f"Training complete. Model saved to trained_model_{model_output_name_addition}.pth")
        artifact = wandb.Artifact(f"model-{run.id}", type="model")
        artifact.add_file(best_model_path)
        run.log_artifact(artifact)
        print(f"Best model (validation loss: {best_val_loss:.8f}) uploaded to wandb")
    
        print("Generating model prediction visualizations...")
        log_visualizations_to_wandb(
            model_path=best_model_path,
            data_path="grids_tensor.pt",
            wandb_run=run,
            num_samples=3  # Visualize first 3 samples
        )

        return avg_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model or run sweep")
    parser.add_argument('--sweep', action='store_true', help='Run as a sweep agent')
    parser.add_argument('--sweep_id', type=str, help='Existing sweep ID to join')
    parser.add_argument('--create_sweep', action='store_true', help='Create a new sweep')
    parser.add_argument('--count', type=int, default=1, help='Number of sweep runs to perform')
    args = parser.parse_args()

    # Import sweep configuration
    from sweep_config import sweep_config, default_params

    # Merge default parameters into sweep config
    for param, value in default_params.items():
        if param not in sweep_config['parameters']:
            sweep_config['parameters'][param] = {'value': value}

    if args.create_sweep:
        # Create a new sweep
        sweep_id = wandb.sweep(sweep_config, project="ped_forecasting")
        print(f"Created sweep with ID: {sweep_id}")
        if args.sweep:
            wandb.agent(sweep_id, function=train, count=args.count)
    elif args.sweep:
        if not args.sweep_id:
            print("Error: Please provide a sweep ID with --sweep_id or use --create_sweep to create a new sweep")
            sys.exit(1)
        # Run as an agent for an existing sweep
        wandb.agent(args.sweep_id, function=train, count=args.count)
    else:
        # Regular training run with default params
        config = {}
        # Use default parameters from sweep_config
        for param, value in default_params.items():
            config[param] = value
        
        # Add other necessary default values (from original training script)
        config.update({
            'cnn_channels': [32, 64, 128],
            'encoder_hidden_dims': [256, 256],
            'latent_dim': 256,
            'decoder_hidden_dims': [256, 512, 512],
            'dropout_rate': 0.1,
            'batch_size': 128,
            'learning_rate': 0.0013,
            'model_output_name_addition': 'standard_run'
        })
        
        train(config)