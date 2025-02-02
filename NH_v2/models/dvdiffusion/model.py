
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .src.resnet import Res12_Quadratic
from .src.embedding import DataEmbedding
from .src.l_variational import L_VariationalAutoencoder
from .src.diffusion_process import Coupled_Diffusion_L_Var
from .src.implicit_dirichlet import DirichletImplicit # We can use both L_VariationalAutoencoder with dirichlet prior see comment in .l_variational.py


class TestDVD(nn.Module):
    def __init__(self, P, Channel, z_dim):
        super(TestDVD, self).__init__()
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


class DVDiffusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        """
        The whole model architecture consists of three main parts, the coupled diffusion process and the generative model are 
         included in diffusion_generate module, an resnet is used to calculate the score. 
        """
        # ResNet that used to calculate the scores.
        # Res12_Quadratic(self,inchan, channel, L, dim,hw,normalize=False,AF=None)

        self.score_net = Res12_Quadratic(inchan=1, channel=args.channels, L=args.abundance_dim, dim=args.score_hdim, hw=args.score_hw, normalize=False, AF=nn.ELU())

        self.without_diffusion = args.without_diffusion
        # Generate the diffusion schedule.
        sigmas = self.get_beta_schedule(args.beta_schedule, args.beta_start, args.beta_end, args.diff_steps)
        alphas = 1.0 - sigmas*0.5
        self.alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0))
        self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(np.cumprod(alphas, axis=0)))
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1-np.cumprod(alphas, axis=0)))
        self.sigmas = torch.tensor(1. - self.alphas_cumprod)

        # Data embedding module.
        self.embedding = DataEmbedding(args.input_dim, args.embedding_dimension,
                                           args.dropout_rate)
        
        #DirichletImplicit(abundance=5, channel=162, alpha_min=0.5, dir_prior=0.1)
        self.dirichlet_implicit = DirichletImplicit(abundance=args.abundance_dim,
                                                    channel=args.channels,
                                                    alpha_min=args.alpha_min,
                                                    dir_prior=args.dir_prior)
                                                    
        # self.dirichlet_implicit = DirichletImplicitMixture(abundance=args.abundance,
        #                                             channel=args.channels,
        #                                             alpha_min=0.5,
        #                                             dir_prior=0.5,
        #                                             n_c = args.n_componenet) # 3 Dirichlet dist 

        # The generative bvae model.
        self.diffusion_gen = Coupled_Diffusion_L_Var(args)

        self.model = TestDVD(P=args.abundance_dim, Channel=args.channels, z_dim=args.abundance_dim)


    def get_beta_schedule(self, beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
        if beta_schedule == 'quad':
            betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
        elif beta_schedule == 'linear':
            betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == 'const':
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_diffusion_timesteps,)
        return betas
    
    def extract(self, a, t, x_shape):
        """ extract the t-th element from a"""
        b, *_ = t.shape
        a = a.to(t.device)
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    

    def inference(self, y_unsup):
        """
        This class extends a DVDiffusion model and defines a forward method that processes input data
        through embedding, projection, concatenation, and variational inference to generate an output.

        Return:
            y: The noisy generative results
            out: Denoised results, remove the noise from y through score matching.
            tc: Total correlations, indicator of extent of disentangling.
        """
        """NEW"""
        # pixel_embedding = self.embedding(y_unsup)
        # proj_feat = self.diffusion_gen.proj_feat(pixel_embedding)
        # y_embedding = torch.cat([proj_feat, pixel_embedding], dim=-1)
        # y_embedding = y_embedding.unsqueeze(1)
        # outputs_dirichlet_unsup = self.dirichlet_implicit(pixel_embedding)
        # y_rec_dirichlet, sampled_abundance, _ , _ = outputs_dirichlet_unsup

        # ### Learning variability with Diffusion + Generative L-VAE
        # logits, kl_diag, all_z= self.diffusion_gen.l_variational(y_embedding) # logits, kl_diag, all_z

        # output = self.diffusion_gen.l_variational.decoder_output(logits)
        
        # # endmemebers_noisy = output.mu.float().requires_grad_()
        # # print("output", output.mu.float().shape)
        # # E = self.score_net(endmemebers_noisy).sum().requires_grad_()
        # # grad_x = torch.autograd.grad(E, endmemebers_noisy, create_graph=True, allow_unused=True)[0]
        # # endmemebers_pred = endmemebers_noisy - grad_x*0.001


        # endmemebers_noisy = output.mu.float().requires_grad_()
        # E = self.score_net(endmemebers_noisy).sum().requires_grad_()
        # endmemebers_pred = endmemebers_noisy #- grad_x*0.001

        y_hat, mu, log_var, abundance_pred, endmemebers_pred = self.model(y_unsup)

        return y_hat, mu, log_var, abundance_pred, endmemebers_pred


    def forward_1(self,
            y_unsupervised,
            y_supervised = None,
            endmembers = None,
            abbandance = None,
            t = None,
            current_epoch=0):
        
        # supervised q(M|a,Z) ~ q(Z|y_supervised)*q(a_sup)
        #pixel_embedding = self.embedding(y_supervised) # whitout this we observed vanishing issues !! talk to khalid for more infos

        # --------------------------SELF SUPERVISED ------------------------------------
        y_hat, mu, log_var, abundance_pred, endmemebers_pred = self.model(y_unsupervised)
        # --------------------------END SELF SUPERVISED --------------------------------

        # -------------------------- SUPERVISED ----------------------------------------
        #y_hat, mu, log_var, abundance_pred, endmemebers_pred = self.model(y_unsupervised.sequeze(-1))
        # -------------------------- END SUPERVISED --------------------------------------
        return y_hat, mu, log_var, abundance_pred, endmemebers_pred #(y_hat_sup, y_hat_unsup), (outputs_dirichlet_sup, outputs_genearitv_diffusion_sup, loss_diffusion_sup,outputs_dirichlet_unsup, outputs_genearitv_diffusion_unsup, loss_diffusion_unsup)
    

    def forward(self,
            y_unsupervised,
            y_supervised,
            endmembers,
            abbandance,
            t, current_epoch=0):
        # supervised q(M|a,Z) ~ q(Z|y_supervised)*q(a_sup)
        pixel_embedding = self.embedding(y_supervised) # whitout this we observed vanishing issues !! talk to khalid for more infos

        # --------------------------SELF SUPERVISED ------------------------------------
        # 1st model
        pixel_embedding = self.embedding(y_unsupervised)
        outputs_dirichlet_unsup = self.dirichlet_implicit(pixel_embedding)
        y_rec_dirichlet, sampled_abundance, kld, max_kld_sampled = outputs_dirichlet_unsup
        # 2nd model 
        outputs_genearitv_diffusion_unsup = self.diffusion_gen(pixel_embedding, endmembers, t, self_supervised=False)
        end_memebers_sample, endmembers_diffused, total_correlation, l_variational_latents = outputs_genearitv_diffusion_unsup

        y_hat_unsup = torch.bmm(end_memebers_sample.sample().float().requires_grad_().squeeze(dim=1), sampled_abundance.unsqueeze(2))


        if self.without_diffusion:
            end_memebers_diffused = endmembers
        # Score matching.
        loss_diffusion_unsup = self.score_diffusion(end_memebers_diffused, end_memebers_sample, t, self_supervised=True)
        # --------------------------END SELF SUPERVISED --------------------------------

        # -------------------------- SUPERVISED ----------------------------------------
        # 1st model
        pixel_embedding = self.embedding(y_supervised)
        outputs_dirichlet_sup = self.dirichlet_implicit(pixel_embedding, endmembers)
        y_rec_dirichlet, sampled_abundance, kld, max_kld_sampled = outputs_dirichlet_sup
        # 2nd model 
        outputs_genearitv_diffusion_sup = self.diffusion_gen(pixel_embedding, endmembers, t, self_supervised=False)
        end_memebers_sample, endmembers_diffused, total_correlation, l_variational_latents = outputs_genearitv_diffusion_sup
        

        #print("end_memebers_sample.sample().float().requires_grad_()", end_memebers_sample.sample().float().requires_grad_().squeeze(dim=1).shape)
        #print('abbandance.unsqueeze(1)', abbandance.permute(0,2,1).shape)

        # linear part #abbandance @ end_memebers_sample
        y_hat_sup = torch.bmm(end_memebers_sample.sample().float().requires_grad_().squeeze(dim=1), abbandance)

        # Score matching.
        loss_diffusion_sup = self.score_diffusion(end_memebers_diffused, end_memebers_sample, t, endmembers=endmembers, self_supervised=False)


        # -------------------------- END SUPERVISED --------------------------------------
        return (y_hat_sup, y_hat_unsup), (outputs_dirichlet_sup, outputs_genearitv_diffusion_sup, loss_diffusion_sup,outputs_dirichlet_unsup, outputs_genearitv_diffusion_unsup, loss_diffusion_unsup)
    

    def score_diffusion(self, end_memebers_diffused, end_memebers_sample, t, endmembers =None, self_supervised=True):
        sigmas_t = self.extract(self.sigmas.to(end_memebers_diffused.device),
                                t.to(end_memebers_diffused.device),
                                end_memebers_diffused.shape)
        if self_supervised:
            endmembers = end_memebers_sample.sample().float().requires_grad_()
        else:
            endmembers = endmembers.unsqueeze(1).float()

        endmembers_diffused = end_memebers_sample.sample().float().requires_grad_()

        ## DIFFUSION UP TO LATENT SAMPLES
        Score_matching_net = self.score_net(endmembers_diffused).sum()
        # The Loss of multiscale score matching.
        grad_x = torch.autograd.grad(Score_matching_net, endmembers_diffused, create_graph=True)[0].to(endmembers_diffused.device)
        loss_diffusion = torch.mean(torch.sum(((endmembers-endmembers_diffused.detach())+grad_x*1e-9)**2*sigmas_t, [1,2,3])
                                    ).float()
        
        return loss_diffusion 
