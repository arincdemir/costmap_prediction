import torch
import torch.nn as nn
import torch.optim as optim

class CNMP(nn.Module):
    def __init__(self, t_dim: int, SM_dim: int, encoder_hidden_dims: list[int], decoder_hidden_dims: list[int], latent_dim: int):
        super(CNMP, self).__init__()

        self.t_dim = t_dim
        self.SM_dim = SM_dim

        # Encoder
        encoder_layers = []
        input_dim = t_dim + SM_dim
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
        decoder_layers.append(nn.Linear(input_dim, SM_dim * 2))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, padded_encodings, encodings_mask, padded_query_indices, queries_mask): 
        # padded_encodings: (batch_size, max_encode_size, t_dim + SM_dim)
        # encodings_mask: (batch_size, max_encode_size) 
        # padded_query_indices: (batch_size, max_query_size, t_dim)
        # queries_mask: (batch_size, max_query_size)

        latent = self.encoder(padded_encodings)  # (batch_size, max_encode_size, latent_dim)
        latent_sum = torch.sum(latent * encodings_mask.unsqueeze(-1), dim=1)  # Sum over valid encodings
        count = torch.clamp(encodings_mask.sum(dim=1, keepdim=True), min=1.0)   # Avoid division by zero
        latent_mean = latent_sum / count   # (batch_size, latent_dim)
        
        # Expand latent_mean to match the number of queries
        latent_mean_expanded = latent_mean.unsqueeze(1).expand(-1, padded_query_indices.size(1), -1)  # (batch_size, max_query_size, latent_dim)
        
        # Concatenate latent_mean with each query index
        decoder_input = torch.cat((latent_mean_expanded, padded_query_indices), dim=-1)  # (batch_size, max_query_size, latent_dim + t_dim)
        
        # Pass through the decoder
        decoder_output = self.decoder(decoder_input)  # (batch_size, max_query_size, SM_dim*2)
        
        # Split into mean and std, apply softplus to std
        mean_raw = decoder_output[:, :, :self.SM_dim]
        mean = torch.nn.functional.sigmoid(mean_raw)    # guide the model to output mean between 0 and 1
        #mean = decoder_output[:, :, :self.SM_dim]
        std_raw = decoder_output[:, :, self.SM_dim:]
        std = torch.nn.functional.softplus(std_raw) + 1e-6  # Ensure positivity
        
        return torch.cat([mean, std], dim=-1)  # Concatenate to match original output shape

    def loss(self, output, padded_query_targets, queries_mask):
        # Split the preprocessed output
        mean = output[:, :, :self.SM_dim]
        std = output[:, :, self.SM_dim:]  # Already processed with softplus + eps
        
        distribution = torch.distributions.Normal(mean, std)
        
        # Compute negative log likelihood (rest unchanged)
        nll = -distribution.log_prob(padded_query_targets)
        nll = nll * queries_mask.unsqueeze(-1)
        valid_elements = queries_mask.sum() * mean.size(-1)
        return nll.sum() / valid_elements.clamp(min=1e-6)