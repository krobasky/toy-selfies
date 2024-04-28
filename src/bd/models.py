import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import torch
from torch.nn.utils.rnn import pad_sequence

from pytorch_lightning.callbacks import RichProgressBar


class CustomRichProgressBar(RichProgressBar):
    def get_metrics(self, trainer, pl_module):
        # Fetch the default metrics (might include average loss, etc.)
        items = super().get_metrics(trainer, pl_module)
        
        # Check if 'train_loss' is in the logged metrics and add it to the items dict
        if trainer.logged_metrics:
            train_loss = trainer.logged_metrics.get('train_loss', None)
            val_loss = trainer.logged_metrics.get('val_loss', None)  # Get validation loss if available
            if train_loss is not None:
                items['loss'] = f"{train_loss:.8f}"  # Format the loss to two decimal places
            if val_loss is not None:
                items['val_loss'] = f"{val_loss:.8f}"
        
        return items


class SELFIESDataset(Dataset):
    def __init__(self, dataset_selfies, max_selfies_tokens, start_token_index):
        self.dataset_selfies = dataset_selfies
        self.max_selfies_tokens = max_selfies_tokens
        self.start_token_index = start_token_index

    def __len__(self):
        return len(self.dataset_selfies)

    def __getitem__(self, idx):
        # add the start token
        encoded_selfies_list = [self.start_token_index] + self.dataset_selfies[idx].encoded_selfies_list
        encoded_selfies = torch.tensor(encoded_selfies_list, dtype=torch.long)
        # Pad the sequence to the maximum tokens length
        padded_selfies = torch.nn.functional.pad(encoded_selfies, (0, self.max_selfies_tokens - len(encoded_selfies)), value=0)
        return padded_selfies

class SELFIESDataModule(LightningDataModule):
    def __init__(self, dataset_selfies, max_selfies_tokens, start_token_index, train_batch_size=32, val_batch_size=32, train_val_split=0.8, drop_last=True):
        super().__init__()
        self.dataset_selfies = dataset_selfies
        self.max_selfies_tokens = max_selfies_tokens
        self.start_token_index = start_token_index
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_val_split = train_val_split
        self.drop_last = drop_last


    def setup(self, stage=None):
        train_size = int(self.train_val_split * len(self.dataset_selfies))
        val_size = len(self.dataset_selfies) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset_selfies, [train_size, val_size])
        self.train_dataset = SELFIESDataset(dataset_selfies = self.train_dataset, max_selfies_tokens = self.max_selfies_tokens, start_token_index=self.start_token_index)
        self.val_dataset = SELFIESDataset(dataset_selfies = self.val_dataset, max_selfies_tokens = self.max_selfies_tokens, start_token_index=self.start_token_index)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, pin_memory=True, drop_last = self.drop_last) 
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, pin_memory=True, drop_last = self.drop_last)

class VAEConv1D(pl.LightningModule):
    def __init__(self, vocab_size=56, max_length=21, embedding_dim=64, feature_channels=32, latent_dims=128):
        super(VAEConv1D, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(feature_channels, feature_channels*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(feature_channels*2*max_length, latent_dims)
        self.fc_var = nn.Linear(feature_channels*2*max_length, latent_dims)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dims, feature_channels*2*max_length)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(feature_channels*2, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(feature_channels, self.vocab_size, kernel_size=3, stride=1, padding=1),  # Output channels = vocab_size
            nn.LogSoftmax(dim=1)  # Optional: Apply LogSoftmax if using NLLLoss instead of CrossEntropyLoss
        )

        '''
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(feature_channels*2, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(feature_channels, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        '''

    def encode(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # Change (batch, length, channels) to (batch, channels, length)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps

    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(-1, 32*2, 21)  # Adjust shape to match the output of encoder
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def training_step(self, batch, batch_idx):
        x = batch  # Updated to handle your DataLoader output
        x_recon, mu, log_var = self.forward(x)
        # Ensure dimensions match: logits (N, C, L) and targets (N, L)
        recon_loss = nn.functional.cross_entropy(x_recon.permute(0, 2, 1), x, ignore_index=0)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_div
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch  # Updated to handle your DataLoader output
        x_recon, mu, log_var = self.forward(x)
        recon_loss = nn.functional.cross_entropy(x_recon.permute(0, 2, 1), x, ignore_index=0)
        self.log('val_loss', recon_loss)
        return loss

    '''
    def training_step(self, batch, batch_idx):
        x = batch
        x_recon, mu, log_var = self.forward(x)
        recon_loss = nn.functional.cross_entropy(x_recon.permute(0, 2, 1), x, ignore_index=0)  # Adjust dimensions for CrossEntropy
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_div
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon, mu, log_var = self.forward(x)
        recon_loss = nn.functional.cross_entropy(x_recon.permute(0, 2, 1), x, ignore_index=0)
        self.log('val_loss', recon_loss)
    '''        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)



class VAELSTM(pl.LightningModule):
    def __init__(self, vocab_size=53, embedding_dim=128, hidden_dim=256, latent_dim=64, learning_rate=0.001):
        super(VAELSTM, self).__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Encoder
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.hidden_to_mean = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.outputs_to_vocab = nn.Linear(hidden_dim, vocab_size)
        
        self.loss_method = nn.CrossEntropyLoss(ignore_index=0)  # assuming 0 is your padding index

        self.train_losses = []
        self.val_losses = []

    def encode(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.encoder(embedded)
        mean = self.hidden_to_mean(hidden.squeeze(0))
        logvar = self.hidden_to_logvar(hidden.squeeze(0))
        return mean, logvar, hidden

    def decode(self, z, input_seq, hidden=None):
        if hidden == None:
            hidden = self.latent_to_hidden(z).unsqueeze(0)
        hidden = self.latent_to_hidden(z).unsqueeze(0)
        embedded_input = self.embedding(input_seq)
        output, hidden = self.decoder(embedded_input, (hidden, torch.zeros_like(hidden)))
        return self.outputs_to_vocab(output), hidden

    def forward(self, x):
        mean, logvar, hidden = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x, hidden = self.decode(z,x, hidden)
        return recon_x, mean, logvar

    def loss_function(self, x, recon_x, mean, logvar):
        recon_loss = self.loss_method(recon_x.view(-1, self.hparams.vocab_size), x.view(-1))
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def training_step(self, batch, batch_idx):
        x = batch
        recon_x, mean, logvar = self.forward(x)
        loss = self.loss_function(x, recon_x, mean, logvar)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        recon_x, mean, logvar = self.forward(x)
        loss = self.loss_function(x, recon_x, mean, logvar)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def on_train_epoch_end(self):
        # Log the average training loss of the epoch
        train_loss_avg = self.trainer.callback_metrics['train_loss'].item()
        self.train_losses.append(train_loss_avg)
        
    def on_validation_epoch_end(self):
        # Log the average validation loss of the epoch
        val_loss_avg = self.trainer.callback_metrics['val_loss'].item()
        self.val_losses.append(val_loss_avg)

    def get_latent_vector(self, x):
        mean, log_var, hidden = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return z


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

    
# Atom list
g_atom_list = ['C', 'H', 'O', 'N', 'F', 'Cl', 'Br', 'I']

class SimpleTransformer(pl.LightningModule):
    def __init__(self, lr, scheduler_patience, emb_size, num_heads, num_layers, target_features, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Assuming g_atom_list is predefined somewhere in the script
        self.emb = nn.Embedding(num_embeddings=len(g_atom_list), embedding_dim=emb_size)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_size, target_features)
        self.loss_fn = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        features, targets = batch
        outputs = self(features)
        loss = self.loss_fn(outputs, targets)
        self.log('trn_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        outputs = self(features)
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def on_train_epoch_end(self, unused=None):
        trn_loss = self.trainer.callback_metrics.get('trn_loss')
        if trn_loss is not None:
            self.train_losses.append(trn_loss.cpu().item())

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.hparams.scheduler_patience)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def plot_losses(self):
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.show()


# Define the Graph Autoencoder
class GraphAutoencoder(pl.LightningModule):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.encoder = GCNConv(in_channels, hidden_dim)
        self.decoder = GCNConv(hidden_dim, in_channels)  # Simplified decoder

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = F.relu(self.encoder(x, edge_index))  # Latent space
        x_reconstructed = self.decoder(z, edge_index)
        return z, x_reconstructed

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        _, x_reconstructed = self(batch)
        adj = to_dense_adj(batch.edge_index, batch_size=batch.num_graphs, max_num_nodes=batch.num_nodes)[0]
        adj_reconstructed = torch.sigmoid(torch.matmul(x_reconstructed, x_reconstructed.transpose(-1, -2)))
        loss = F.mse_loss(adj_reconstructed, adj)
        return loss

    def save_model(self, filename="/content/drive/My Drive/selfies/gnn_model.pth"):
        # model.save_model()
        # Save the model's state_dict and optionally other components like optimizer
        torch.save({
            'model_state_dict': self.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="/content/drive/My Drive/selfies/gnn_model.pth"):
        # model = GraphAutoencoder(in_channels=dataset.num_features)
        # model.load_model()
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filename}")

    def generate_random_graph(self, num_nodes, hidden_dim=64):
        # Randomly generate a latent space vector
        # hidden_dim must be the original used in the model
        z = torch.randn((num_nodes, hidden_dim))
        # Generate a fully-connected edge_index
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # Decode the latent vector to an adjacency matrix
        x_reconstructed = self.forward(z, edge_index)
        adj_reconstructed = torch.sigmoid(torch.matmul(x_reconstructed, x_reconstructed.transpose(-1, -2)))
        # Threshold to get a binary adjacency matrix
        adj_reconstructed = (adj_reconstructed > 0.5).float()
        return adj_reconstructed


    # def generate_graph(self, num_nodes):
    #     # Randomly generate a latent space vector
    #     z = torch.randn((num_nodes, self.hidden_dim))
    #    # Generate a random but valid edge_index (simple chain for demonstration)
    #     edge_index = torch.tensor([[i, i+1] for i in range(num_nodes - 1)], dtype=torch.long).t().contiguous()
    #     # Decode the latent vector to an adjacency matrix
    #     x_reconstructed = self.forward(z, edge_index)
    #     adj_reconstructed = torch.sigmoid(torch.matmul(x_reconstructed, x_reconstructed.transpose(-1, -2)))
    #     # Threshold to get a binary adjacency matrix
    #     adj_reconstructed = (adj_reconstructed > 0.5).float()
    #     return adj_reconstructed

