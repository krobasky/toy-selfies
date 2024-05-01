import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.utils import to_dense_adj

from einops import rearrange


import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import random
from rdkit import Chem
from rdkit.Chem import AllChem


class MolecularEnv:
    def __init__(self):
        self.state = None
        self.current_molecule = None  # Keep track of the current molecule

    def reset(self):
        # Start with a simple molecule, e.g., benzene
        self.current_molecule = Chem.MolFromSmiles('c1ccccc1')
        fp = AllChem.GetMorganFingerprintAsBitVect(self.current_molecule, 2, nBits=2048)
        self.state = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)
        return self.state

    def step(self, action):
        # Apply the action to the current molecule to create a new molecule
        # For simplicity, let's say each action corresponds to adding a different functional group
        
        # Convert the current molecule to a RWMol for editing
        rw_mol = Chem.RWMol(self.current_molecule)

        if action == 0:
            # Example action: Add an OH group to a random carbon
            carbon_indices = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 6]
            if carbon_indices:
                chosen_carbon = random.choice(carbon_indices)
                rw_mol.AddAtom(Chem.Atom(8))  # Add oxygen
                rw_mol.AddBond(chosen_carbon, rw_mol.GetNumAtoms() - 1, order=Chem.BondType.SINGLE)
        
        # Other actions can correspond to other modifications, e.g., adding different functional groups
        
        # Update the current molecule
        tmp_mol = rw_mol.GetMol()

        # After modifying the molecule:
        try:
            # Sanitize the molecule to ensure its consistency and to compute properties
            Chem.SanitizeMol(tmp_mol)
            
            # Update the state to reflect the changes
            fp = AllChem.GetMorganFingerprintAsBitVect(tmp_mol, 2, nBits=2048)
            self.current_molecule = tmp_mol
            self.state = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)
        except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException):
            # Handle cases where the molecule cannot be kekulized (usually due to valence issues)
            pass  # Here you can decide what to do with such molecules, e.g., discard or replace


        # For now, return a random reward and termination condition
        reward = torch.randn(1)
        done = torch.rand(1) > 0.95

        return self.state, reward, done

    def orig_step(self, action):
        # This should modify the molecule based on the action; simplified here
        reward = torch.randn(1)
        done = torch.rand(1) > 0.95
        # Simulate a change in the molecule (this should be more meaningful)
        self.current_molecule = Chem.MolFromSmiles('c1ccncc1')  # Example change
        return self.state, reward, done

    def get_current_molecule(self):
        return self.current_molecule

# Define the policy model that takes a fingerprint and predicts a probability vector for "actions"
class Policy(nn.Module):
    def __init__(self, input_size=2048, output_size=10):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)
        )
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        return self.layers(x)


class TransformerVAE(pl.LightningModule):
    def __init__(self, vocab_size, max_length, d_model=512, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model

        # Transformer Encoder settings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer Decoder settings
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, max_length, d_model))

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Latent space and reconstruction layers
        self.to_latent = nn.Linear(d_model, 2 * d_model)  # Output both mu and logvar
        self.from_latent = nn.Linear(d_model, d_model)

        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

        self.train_losses = []
        self.val_losses = []
        

    def encode(self, src):
        src = self.embedding(src) + self.pos_encoder[:, :src.size(1)]
        encoded_src = self.encoder(src)
        mu_logvar = self.to_latent(encoded_src.mean(dim=1))
        mu, log_var = mu_logvar.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_length):
        # Process the latent vector z to shape it for the transformer decoder input
        z = self.from_latent(z).unsqueeze(1).repeat(1, target_length, 1)
        z += self.pos_encoder[:, :target_length]
        # Since the decoder in Transformer expects a 'memory' (outputs from the encoder),
        # we will use the processed latent vector as a pseudo-memory.
        # It is necessary to duplicate the latent representation to fulfill the API requirements
        # although semantically it does not function as typical memory does in Transformer.
        output = self.decoder(z, z)
        return self.output_layer(output)

    def forward(self, src, src_mask=None):
        mu, log_var = self.encode(src)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, src.size(1)), mu, log_var

    def training_step(self, batch, batch_idx):
        x = batch
        output, mu, log_var = self.forward(x)
        recon_loss = F.cross_entropy(output.view(-1, self.vocab_size), x.view(-1))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        output, mu, log_var = self.forward(x)
        recon_loss = F.cross_entropy(output.view(-1, self.vocab_size), x.view(-1))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def on_train_epoch_end(self):
        # Log the average training loss of the epoch
        train_loss_avg = self.trainer.callback_metrics['train_loss'].item()
        self.train_losses.append(train_loss_avg)
        
    def on_validation_epoch_end(self):
        # Log the average validation loss of the epoch
        val_loss_avg = self.trainer.callback_metrics['val_loss'].item()
        self.val_losses.append(val_loss_avg)




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

    
