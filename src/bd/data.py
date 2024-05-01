import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import RichProgressBar
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.datasets import QM9

import selfies as sf
from rdkit import Chem


from bd import log as bdl

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



def smiles_to_selfies(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:  # Ensure the molecule is valid
        smiles = Chem.MolToSmiles(mol)  # Sanitize SMILES
        return sf.encoder(smiles)
    return None

VOCAB_FILE = 'config/vocab.json'
# see notes in _vocab_from_file on how this file was created
def _build_vocab(dataset, start_token, pad_token, vocab_file=VOCAB_FILE):
    from os.path import exists

    if exists(vocab_file):
        # see notes in _vocab_from_file on how this file was created
        return(_vocab_from_file())
    else:
        vocab = set()
        for data in dataset:
            if data.get('selfies',False):
                tokens = sf.split_selfies(data.selfies)
                vocab.update(tokens)
        vocab = sorted(vocab)  # Sort for consistency
        token_to_idx = {token: i for i, token in enumerate(vocab, start=1)}  # Start indexing from 1
        token_to_idx[pad_token] = 0 # Add a padding token
        token_to_idx[start_token] = len(token_to_idx) # Add a start token
        return start_token, pad_token, token_to_idx

def _vocab_from_file(vocab_file=VOCAB_FILE):
    '''Notes on how vocab_file was created:

 1. _build_vocab is run once to get a list of all the tokens present in the QM9 dataset

 2. Add a padding token at index 0

 3. Build on from there with chemically similar
    tokens clustered together into a list as follows:

    + focusing first on simplicity and then moving to complexity within each atom type:

    + This arrangement starts with simple elements and progressively
      introduces more complex features such as stereochemistry,
      charges, and multiple bonds, reflecting both the structural and
      functional hierarchy in organic chemistry.

 4. The list, ordered with nearest neighbors having the least affect
    on compound chemistry (drug-likeness, toxicity) if switched;
    caveat, this should be verified by a chemist:

        ordered_tokens = [
            # Hydrogen
            '[H]', '[\\H]', '[/H]',

            # Carbon Tokens
            '[C]', '[\\C]', '[/C]',                                              # Simple carbon
            '[C@@]', '[C@]',                                                     # Carbons with stereochemistry
            '[C@@H1]', '[C@H1]', '[\\C@@H1]', '[\\C@H1]', '[/C@@H1]', '[/C@H1]', # Carbons in functional groups
            '[C@H1+1]', '[CH1+1]', '[CH1-1]',                                    # Charged carbons or variations
            '[=C]',                                                              # Double bond
            '[Ring1]', '[Ring2]', '[-/Ring1]', '[-\\Ring1]', '[=Ring1]',         # Rings
            '[Branch1]', '[Branch2]',                                            # Branches
            '[=Branch1]', '[=Branch2]',                                          # Double-bonded branches
            '[#C]',                                                              # Triple bonded C
            '[#Branch1]', '[#Branch2]',                                          # Triple bonded branches

            # Nitrogen Tokens
            '[N]', '[\\N]', '[/N]',                                              # Simple nitrogens
            '[N+1]', '[/N+1]',                                                   # Charged nitrogens
            '[NH1]', '[NH1-1]', '[NH1+1]', '[/NH1+1]',
            '[NH2+1]', '[NH3+1]', 
            '[N@@H1+1]', '[N@H1+1]',                                             # Nitrogens with hydrogen or stereochemistry
            '[=N]', '[=N+1]', '[=NH1+1]', '[=NH2+1]',                            # Double-bonded nitrogens
            '[#N]',                                                              # Triple-bonded nitrogens

            # Oxygen and Complex Functional Groups
            '[O]','[\\O]', '[/O]',                                               # Simple oxygens
            '[O-1]',                                                             # Charged oxygen
            '[=O]',                                                              # Double-bonded oxygen

            '[F]', 
        ]


    + Generally, this list:

      a. starts with the most basic elements (hydrogen and a simple
         halogen).

      b. Progresses through carbons, starting from the simplest carbon
         representations to more complex ones involving
         stereochemistry and functional groups.

      c. Follows with nitrogen tokens, arranged from simple elemental
         representations to more complex ions and functional groups.

      d. Includes oxygen and ends with complex oxygen-related groups.

      e. This arrangement should help anyone working with these
         SELFIE tokens to find related structural representations
         quickly and intuitively.

5. Next: The list is made into a vocabulary dictionary

    >>> unordered_tokens = vocab.keys()
    >>> # ensure nothing is missing
    >>> for token in unordered_tokens:
    >>>     if token not in ordered_tokens:
    >>>         print(token)
    >>> # if anything besides "PAD_TOKEN" shows up, insert it to ordered_tokens
    >>> # then save ordered_tokens to file:
    >>> import json
    >>> {idx: token for idx, token in enumerate(['PAD_TOKEN'] + ordered_tokens)}
    >>> with open('vocab.json', 'w') as json_file:
    >>>     json.dump(vocab, json_file, indent=4)


6. It would be better to create the list in a data-driven fashion,
   e.g., via computational clustering

'''
    import json
    # Open the JSON file and load it into a dictionary
    with open(vocab_file, 'r') as json_file:
        vocab = json.load(json_file)
    idx_to_token = {vocab[token]: token for token in vocab.keys()}
    pad_token = idx_to_token[0] # its convention to put pad at 0...
    start_token = idx_to_token[len(idx_to_token) - 1] #...and start at len
    return start_token, pad_token, vocab

def setup(data_path='data/QM9', start_token='<start>', pad_token='<pad>'):
    # Load the dataset
    dataset = QM9(root=data_path)

    bdl.info("+ Add SELFIES to the dataset entries...")
    bdl.info("outputting 'x' for each record skipped due to unusable SMILES string.")
    import sys
    import os
    dataset_selfies_tmp = []
    # -- Redirect stderr to null
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    for data in dataset:
        data.selfies = smiles_to_selfies(data.smiles)
        if data.get('selfies',None) is not None:
            dataset_selfies_tmp.append(data)
        else:
            bdl.info("x",endl="")
    sys.stderr.flush()
    # -- Restore stderr
    sys.stderr = stderr

    bdl.info("+ Creating vocabulary...")
    start_token, pad_token, vocab = _build_vocab(dataset_selfies_tmp, start_token=start_token, pad_token=pad_token)

    bdl.info("+ Finding stats for number of tokens in the selfies:...")
    import numpy as np
    token_lengths=[]
    for data in dataset_selfies_tmp:
        tokens = sf.split_selfies(data.selfies)
        token_lengths.append(len(list(tokens)))
    token_lengths = np.array(token_lengths)
    token_stats = {
        "median": np.median(token_lengths),
        "average":np.average(token_lengths),
        "std": np.std(token_lengths),
        "max": np.max(token_lengths),
        "min": np.min(token_lengths),
    }
    max_selfies_tokens = np.max(token_lengths)
                        
    bdl.info("+ Adding SELFIES to the dataset entries...")
    dataset_selfies = []
    for data in dataset_selfies_tmp:
        data.encoded_selfies_list = [vocab[token] for token in list(sf.split_selfies(data.selfies))]
        dataset_selfies.append(data)

    bdl.info("+ Done.")
    return dataset_selfies, max_selfies_tokens, token_stats, vocab, start_token, pad_token
