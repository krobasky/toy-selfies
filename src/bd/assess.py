import torch
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw

from bd import log as bdl


def selfies_to_image(selfies_string):
    # Convert SELFIES to SMILES
    smiles = sf.decoder(selfies_string)
    
    if smiles is None:
        print("Invalid SELFIES string.")
        return

    # Generate a molecule object from SMILES
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print("Could not parse SMILES string.")
        return

    # Use RDKit to draw the molecule
    return Draw.MolToImage(mol)

def jitter_latent_vector(z, std=0.05):
    noise = torch.randn_like(z) * std
    return z # + noise

def decode_indices_to_selfies(indices, vocab):
    ''' Convert a list of indices to a SELFIES string using the vocab. '''
    idx2vocab = {vocab[token]: token for token in vocab.keys() }
    return ''.join(idx2vocab[idx] for idx in indices if idx != 0)  # Assuming PAD_TOKEN is 0

def generate_and_decode(model, vocab, max_tokens, start_token = '<start>', pad_token = '<pad>', z=None, latent_dim=None ):
    model.eval()
    with torch.no_grad():
        generated_indices, z = generate_selfies(model, max_length=max_tokens, 
                                                start_token_index=vocab[start_token], 
                                                pad_token_index=vocab[pad_token], 
                                                z=z,
                                                latent_dim=latent_dim)
    selfies_string = decode_indices_to_selfies(generated_indices.squeeze().tolist(), vocab)
    return selfies_string, z, generated_indices

def generate_selfies(model, max_length, start_token_index=55, pad_token_index=0, z=None, latent_dim=None, num_samples=1):
    if z is None:
        # Sample from the standard normal distribution, which is what the VAE's latent space assumes
        if latent_dim is None:
            z = torch.randn(num_samples, model.latent_dim)
            logits = model.decoder(z)
        else:
            # transformer VAE decoder is peculiar
            z = torch.randn(num_samples, latent_dim)
            logits = model.decoder(z,z)

    else:
        # Optionally, a deterministic z can be passed in, but it would usually be sampled
        pass

    generated = torch.zeros(num_samples, max_length, dtype=torch.long)

    probs = torch.exp(logits)

    for i in range(max_length):
        # Sample from the probability distribution for the next token
        next_token = torch.multinomial(probs, 1).squeeze(-1)
        bdl.info(f"token:{next_token}")
        if next_token == start_token_index:
            bdl.info(f"---SKIP:{next_token}")
            # skip this one
            i = i - 1
            continue
        if next_token == pad_token_index:
            break
        generated[:, i] = next_token
    return generated, z
