from torch import nn
import torch.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirichletMixture(nn.Module):
    """_summary_
    The `DirichletMixture` class represents a mixture model with a specified number of components, each
    modeled as a Dirichlet distribution with a shared concentration parameter alpha.  

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_components, alpha):
        super(DirichletMixture, self).__init__()
        self.num_components = num_components
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(num_components) / num_components)
        self.dirichlets = nn.ModuleList([dist.Dirichlet(alpha) for _ in range(num_components)])

    def forward(self):
        samples = [dirichlet.sample() for dirichlet in self.dirichlets]
        mixed_sample = sum(weight * sample for weight, sample in zip(self.weights, samples))
        return mixed_sample
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out



class AbundanceResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AbundanceResNet, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(64, 64, blocks=1)
        self.layer3 = self.make_layer(64, 256, blocks=1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        #x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
class DirichletImplicit(nn.Module):
    def __init__(self, abundance, channel, alpha_min, dir_prior):
        super(DirichletImplicit, self).__init__()
        self.P = abundance
        self.Channel = channel
        self.alpha_min = torch.tensor([alpha_min])

        self.abundance_enc_resnet = AbundanceResNet(self.Channel,self.P)
        # encoder a
        self.fc9 = nn.Linear(self.Channel, 32 * self.P)
        self.bn9 = nn.BatchNorm1d(32 * self.P)
        self.fc10 = nn.Linear(32 * self.P, 16 * self.P)
        self.bn10 = nn.BatchNorm1d(16 * self.P)
        self.fc11 = nn.Linear(16 * self.P, 4 * self.P)
        self.bn11 = nn.BatchNorm1d(4 * self.P)
        self.fc12 = nn.Linear(4 * self.P, 4 * self.P)
        self.bn12 = nn.BatchNorm1d(4 * self.P)
        self.fc13 = nn.Linear(4 * self.P, 1 * self.P)  # get abundance
        ## decoder
        self.fc6 = nn.Linear(self.P, self.P * 4)
        self.bn6 = nn.BatchNorm1d(self.P * 4)
        self.fc7 = nn.Linear(self.P * 4, self.P * 64)
        self.bn7 = nn.BatchNorm1d(self.P * 64)
        self.fc8 = nn.Linear(self.P * 64, self.Channel)
        self.min_alpha = torch.tensor(self.alpha_min, dtype=torch.float32)
        self.dir_prior = dir_prior
        self.n_componenet = abundance
        
    def encoder(self, x):
        alphas = self.abundance_enc_resnet(x)
        return alphas

    def decoder(self, z):
        h1 = self.fc6(z)
        h1 = self.bn6(h1)
        h1 = F.leaky_relu(h1, 0.00)
        h1 = self.fc7(h1)
        h1 = self.bn7(h1)
        h1 = F.leaky_relu(h1, 0.00)
        h1 = self.fc8(h1).unsqueeze(-1)
        return h1

    def encoder_a(self, x):
        h1 = self.fc9(x)
        h1 = self.bn9(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc10(h1)
        h1 = self.bn10(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc11(h1)
        h1 = self.bn11(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc12(h1)
        h1 = self.bn12(h1)

        h1 = F.leaky_relu(h1, 0.00)
        h1 = self.fc13(h1)

        a = F.softmax(h1, dim=1)
        return a

    def forward(self,
                inputs,
                endmemebers = None,
                self_supervised = True):
        
        if self_supervised:
            self.mean = self.encoder(inputs)
            # Assuming self.mean and self.min_alpha are tensors
            # Compute self.alpha
            self.alpha = torch.max(self.alpha_min.to(self.mean.device), torch.log(1. + torch.exp(self.mean)))
            # Dirichlet prior alpha0
            self.prior = torch.ones(inputs.shape[0], self.n_componenet, dtype=torch.float32) * self.dir_prior
            # Assuming self.alpha is a tensor representing the alpha values
            # Create a Dirichlet distribution and sample from it
            dirichlet_dist = dist.Dirichlet(self.alpha)
            sampled_abundance = dirichlet_dist.rsample()
            #print("rsample", sampled_abundance.shape)
            # Squeeze the sampled tensor to remove singleton dimensions
            #sampled_abundance = torch.squeeze(sampled_abundance)
            # Assuming self.prior and self.alpha are tensors representing the prior and alpha values respectively
            # Assuming self.doc_vec is the document vector
            # Create Dirichlet distributions
            dirichlet_prior = dist.Dirichlet(self.prior.to(self.mean.device))
            dirichlet_alpha = dist.Dirichlet(self.alpha.to(self.mean.device))
            # Compute KL divergence Implicit 
            kld = dirichlet_alpha.log_prob(sampled_abundance).to(self.mean.device)
            kld =  kld- dirichlet_prior.log_prob(sampled_abundance).to(self.mean.device)
            # Find the index of the maximum KLD value
            max_kld_sampled = torch.argmax(kld, dim=0)
            y_rec = self.decoder(sampled_abundance)
        else:
            self.mean = self.encoder(inputs)
            self.alpha = torch.max(self.alpha_min.to(self.mean.device), torch.log(1. + torch.exp(self.mean)))
            self.prior = torch.ones(inputs.shape[0], self.n_componenet, dtype=torch.float32) * self.dir_prior
            dirichlet_dist = dist.Dirichlet(self.alpha)
            sampled_abundance = dirichlet_dist.sample()
            sampled_abundance = torch.squeeze(sampled_abundance)
            dirichlet_prior = dist.Dirichlet(self.prior.to(self.mean.device))
            dirichlet_alpha = dist.Dirichlet(self.alpha)
            kld = dirichlet_alpha.log_prob(sampled_abundance) - dirichlet_prior.log_prob(sampled_abundance)
            max_kld_sampled = torch.argmax(kld, dim=0)
            y_rec = sampled_abundance.view([-1, 1, self.P]) @ endmemebers

        y_rec = torch.squeeze(y_rec, dim=1)
        return (y_rec, sampled_abundance, kld.mean(), max_kld_sampled) #(y_rec_dirichlet, sampled_abundance, kld, max_kld_sampled)


# The code snippet you provided is a typical Python script structure. The `if __name__ == '__main__':`
# block is a common Python idiom used to make the code inside it only run if the script is executed
# directly, not if it is imported as a module in another script.
# if __name__ == '__main__':
#     device = 'cpu'
#     model = DirichletImplicit(abundance=5, channel=162, alpha_min=0.5, dir_prior=0.1)
#     input = torch.randn(32, 162, 128) #.squeeze()
#     print('input',input.shape)
#     y_hat, sampled_abundance, kld, max_kld_sampled = model(input, self_supervised=True)
#     print("#"*100)
#     print(' shape of y_hat: ', y_hat.shape)
#     print(' shape of sampled_abundance: ',sampled_abundance.shape)

