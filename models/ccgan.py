import torch
import torch.nn as nn
import utils
import metrics
import progressive_blocks
import plotting
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam



class CCGAN:
    def __new__(self,
            seq_length,
            batch_size = None,
            embedding_dim = None,
            value_features = None,
            key_features = None,
            num_features = 11,
            sa  = None,
            gpu = None,
            **kwargs):
    
        device =utils.assign_device(gpu)
        self.DiscriminatorModel =progressive_blocks.Discriminator(embedding_dim,
                                           seq_length,
                                           num_features,
                                           batch_size,
                                           value_features,
                                           key_features,sa,
                                           device)

        self.GeneratorModel=progressive_blocks.Generator(embedding_dim,
                                       seq_length,
                                       num_features,
                                       batch_size,
                                       value_features,
                                       key_features,sa,device)
        
        return {"D":self.DiscriminatorModel, "G":self.GeneratorModel}