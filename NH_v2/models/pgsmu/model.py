
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PGMSU(nn.Module):
    def __init__(self, P, Channel, z_dim):
        super(PGMSU, self).__init__()
        self.P = P
        self.Channel = Channel
        # encoder z  fc1 -->fc5
        self.fc1 = nn.Linear(Channel, 32 * P)
        self.bn1 = nn.BatchNorm1d(32 * P)

        self.fc2 = nn.Linear(32 * P, 16 * P)
        self.bn2 = nn.BatchNorm1d(16 * P)
        
        self.fc3 = nn.Linear(16 * P, 4 * P)
        self.bn3 = nn.BatchNorm1d(4 * P)

        self.fc4 = nn.Linear(4 * P, z_dim)
        self.fc5 = nn.Linear(4 * P, z_dim)

        # encoder a
        self.fc9 = nn.Linear(Channel, 32 * P)
        self.bn9 = nn.BatchNorm1d(32 * P)

        self.fc10 = nn.Linear(32 * P, 16 * P)
        self.bn10 = nn.BatchNorm1d(16 * P)

        self.fc11 = nn.Linear(16 * P, 4 * P)
        self.bn11 = nn.BatchNorm1d(4 * P)

        self.fc12 = nn.Linear(4 * P, 4 * P)
        self.bn12 = nn.BatchNorm1d(4 * P)

        self.fc13 = nn.Linear(4 * P, 1 * P)  # get abundance

        ### decoder
        self.fc6 = nn.Linear(z_dim, P * 4)
        self.bn6 = nn.BatchNorm1d(P * 4)

        self.fc7 = nn.Linear(P * 4, P * 64)
        self.bn7 = nn.BatchNorm1d(P * 64)

        self.fc8 = nn.Linear(P * 64, Channel * P)

    def encoder_z(self, x):
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc2(h1)
        h1 = self.bn2(h1)
        h11 = F.leaky_relu(h1, 0.00)

        h1 = self.fc3(h11)
        h1 = self.bn3(h1)
        h1 = F.leaky_relu(h1, 0.00)

        mu = self.fc4(h1)
        log_var = self.fc5(h1)
        return mu, log_var

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

    def reparameterize(self, mu, log_var):
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=mu.device)
        return mu + eps * std

    def decoder(self, z):
        h1 = self.fc6(z)
        h1 = self.bn6(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc7(h1)
        h1 = self.bn7(h1)
        h1 = F.leaky_relu(h1, 0.00)
        #print("h1", h1.shape)
        h1 = self.fc8(h1)
        em = torch.sigmoid(h1)
        #print("em", em.shape)
        return em

    def forward(self, inputs):
        mu, log_var = self.encoder_z(inputs)
        a = self.encoder_a(inputs)

        # reparameterization trick
        z = self.reparameterize(mu, log_var)
        em = self.decoder(z)

        em_tensor = em.view([-1, self.P, self.Channel])
        a_tensor = a.view([-1, 1, self.P])
        y_hat = a_tensor @ em_tensor
        y_hat = torch.squeeze(y_hat, dim=1)

        return y_hat, mu, log_var, a, em_tensor
    