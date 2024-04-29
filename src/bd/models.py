import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch_geometric.nn import GCNConv

from torch_geometric.utils import to_dense_adj
class GCNAutoencoder(pl.LightningModule):
    def __init__(self, in_channels, hidden_dim=64, lr=0.001, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = None
        gcnconv = GCNConv(in_channels, hidden_dim)
        self.encoder = gcnconv
        self.decoder = GCNConv(hidden_dim, in_channels)  # Simplified decoder

        self.train_losses = []
        self.val_losses = []

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = nn.functional.relu(self.encoder(x, edge_index))  # Latent space
        x_reconstructed = self.decoder(z, edge_index)
        return z, x_reconstructed

    def training_step(self, batch, batch_idx):
        _, x_reconstructed = self(batch)
        adj = to_dense_adj(batch.edge_index, batch_size=batch.num_graphs, max_num_nodes=batch.num_nodes)[0]
        adj_reconstructed = torch.sigmoid(torch.matmul(x_reconstructed, x_reconstructed.transpose(-1, -2)))
        loss = nn.functional.mse_loss(adj_reconstructed, adj)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        _, x_reconstructed = self(batch)
        adj = to_dense_adj(batch.edge_index, batch_size=batch.num_graphs, max_num_nodes=batch.num_nodes)[0]
        adj_reconstructed = torch.sigmoid(torch.matmul(x_reconstructed, x_reconstructed.transpose(-1, -2)))
        loss = nn.functional.mse_loss(adj_reconstructed, adj)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def on_train_epoch_end(self):
        # Log the average training loss of the epoch
        train_loss_avg = self.trainer.callback_metrics['train_loss'].item()
        self.train_losses.append(train_loss_avg)
        
    def on_validation_epoch_end(self):
        # Log the average validation loss of the epoch
        val_loss_avg = self.trainer.callback_metrics['val_loss'].item()
        self.val_losses.append(val_loss_avg)

    def save_model(self, filename="../../models/gnn_model.pth"):
        # model.save_model()
        # Save the model's state_dict and optionally other components like optimizer
        torch.save({
            'model_state_dict': self.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="../../models/gnn_model.pth"):
        # model = GraphAutoencoder(in_channels=dataset.num_features)
        # model.load_model()
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filename}")

class VAE(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=128, latent_dim=64):
        # existing initialization
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim

        # The encoder now needs to produce two things: a mean and a log-variance
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim, padding_idx=0),
            nn.Linear(embedding_dim, 2 * latent_dim),  # for both mean and log-variance
            nn.ReLU(True) # xxx???
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embedding_dim),
            nn.ReLU(True),
            nn.Linear(embedding_dim, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

        
        self.train_losses = []
        self.val_losses = []
        
    def encode(self, x):
        encoded = self.encoder(x)
        mean, log_var = encoded.chunk(2, dim=-1)  # split the encoder output into mean and log-var
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), mean, log_var

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat, mean, log_var = self(x)
        # Reconstruction loss
        recon_loss = nn.functional.nll_loss(x_hat.transpose(1, 2), x, ignore_index=0)
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # Total loss
        loss = recon_loss + kl_div
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat, mean, log_var = self(x)
        recon_loss = nn.functional.nll_loss(x_hat.transpose(1, 2), x, ignore_index=0)
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        val_loss = recon_loss + kl_div
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_train_epoch_end(self):
        # Log the average training loss of the epoch
        train_loss_avg = self.trainer.callback_metrics['train_loss'].item()
        self.train_losses.append(train_loss_avg)
        
    def on_validation_epoch_end(self):
        # Log the average validation loss of the epoch
        val_loss_avg = self.trainer.callback_metrics['val_loss'].item()
        self.val_losses.append(val_loss_avg)

    def get_latent_vector(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return z

    def generate_selfies(self, max_length, start_token='<start>', pad_token='<pad>', z=None, num_samples=1):
        if z is None:
            # Sample from the standard normal distribution, which is what the VAE's latent space assumes
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
        else:
            # Optionally, a deterministic z can be passed in, but it would usually be sampled
            pass
        
        generated = torch.zeros(num_samples, max_length, dtype=torch.long, device=self.device)

        # Decode the sampled z
        logits = self.decoder(z)
        probs = torch.exp(logits)

        for i in range(max_length):
            # Sample from the probability distribution for the next token
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            if next_token == pad_token:
                break
            generated[:, i] = next_token
        return generated, z

    
