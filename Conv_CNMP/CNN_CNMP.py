import torch
import torch.nn as nn
import torch.optim as optim

class CNN_CNMP(nn.Module):
    def __init__(self, t_dim: int, grid_size: int, encoder_hidden_dims: list[int], 
                 decoder_hidden_dims: list[int], latent_dim: int, cnn_channels: list[int]):
        super(CNN_CNMP, self).__init__()
        
        self.t_dim = t_dim
        self.grid_size = grid_size
        
        # CNN to process the grid
        cnn_layers = []
        in_channels = 1 
        for out_channels in cnn_channels:
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate the feature map size after CNN
        # Each maxpool with stride 2 halves the dimensions
        reduced_size = grid_size // (2 ** len(cnn_channels))
        self.SM_dim = reduced_size * reduced_size * cnn_channels[-1]
        
        # Encoder
        encoder_layers = []
        input_dim = t_dim + self.SM_dim
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            input_dim = hidden_dim
        encoder_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        input_dim = latent_dim + t_dim
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            input_dim = hidden_dim
        decoder_layers.append(nn.Linear(input_dim, self.grid_size**2 * 2)) 
        self.decoder = nn.Sequential(*decoder_layers)

    def preprocess_grid(self, grid):
        # grid shape: (batch_size, max_encode_size, grid_size, grid_size)
        batch_size, max_encode_size = grid.shape[0], grid.shape[1]
        
        # Reshape for CNN processing
        # for CNN, we merge batch_size and max_encode_size, we will split the two after processing
        grid_reshaped = grid.view(-1, 1, self.grid_size, self.grid_size)  # (-1, 1, grid_size, grid_size)
        
        # Process through CNN
        grid_features = self.cnn(grid_reshaped)  # (-1, last_channel, reduced_size, reduced_size)
        
        # Flatten the spatial dimensions
        grid_features = grid_features.view(batch_size, max_encode_size, -1)  # (batch_size, max_encode_size, SM_dim)
        
        return grid_features

    def forward(self, padded_time_indices, padded_grids, encodings_mask, padded_query_indices, queries_mask):
        # padded_time_indices: (batch_size, max_encode_size, t_dim)
        # padded_grids: (batch_size, max_encode_size, grid_size, grid_size)
        # encodings_mask: (batch_size, max_encode_size)
        # padded_query_indices: (batch_size, max_query_size, t_dim)
        # queries_mask: (batch_size, max_query_size)
        
        # Process grids through CNN
        grid_features = self.preprocess_grid(padded_grids)  # (batch_size, max_encode_size, SM_dim)
        
        # Concatenate time indices with grid features
        padded_encodings = torch.cat([padded_time_indices, grid_features], dim=-1)  # (batch_size, max_encode_size, t_dim + SM_dim)
        
        # Encoder
        latent = self.encoder(padded_encodings)  # (batch_size, max_encode_size, latent_dim)
        latent_sum = torch.sum(latent * encodings_mask.unsqueeze(-1), dim=1)  # Sum over valid encodings
        count = torch.clamp(encodings_mask.sum(dim=1, keepdim=True), min=1.0)  # Avoid division by zero
        latent_mean = latent_sum / count  # (batch_size, latent_dim)
        
        # Expand latent_mean to match the number of queries
        latent_mean_expanded = latent_mean.unsqueeze(1).expand(-1, padded_query_indices.size(1), -1)  # (batch_size, max_query_size, latent_dim)
        
        # Concatenate latent_mean with each query index
        decoder_input = torch.cat((latent_mean_expanded, padded_query_indices), dim=-1)  # (batch_size, max_query_size, latent_dim + t_dim)
        
        # Pass through the decoder
        decoder_output = self.decoder(decoder_input)  # (batch_size, max_query_size, SM_dim*2)
        
        # Split into mean and std, apply softplus to std
        mean_raw = decoder_output[:, :, :self.grid_size**2]
        mean = torch.nn.functional.sigmoid(mean_raw)  # guide the model to output mean between 0 and 1
        std_raw = decoder_output[:, :, self.grid_size**2:]
        std = torch.nn.functional.softplus(std_raw) + 1e-6  # Ensure positivity
        
        return torch.cat([mean, std], dim=-1)  # (batch_size, max_query_size, SM_dim*2)

    def loss(self, output, padded_query_targets, queries_mask):
        # Split the preprocessed output
        mean = output[:, :, :self.grid_size**2]          # Modified line
        std = output[:, :, self.grid_size**2:]           # Modified line
        
        distribution = torch.distributions.Normal(mean, std)
        
        # Compute negative log likelihood
        nll = -distribution.log_prob(padded_query_targets)
        nll = nll * queries_mask.unsqueeze(-1)
        valid_elements = queries_mask.sum() * mean.size(-1)
        return nll.sum() / valid_elements.clamp(min=1e-6)