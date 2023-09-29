from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import plotting
import lstm
import metrics
import train
import torch.nn as nn
import progressive_blocks
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

class TrainingPipeline:
    def __init__(self,
                training_config = None,
                model= None, **kwargs):
        self.model = model
        self.batch_size = training_config["batch_size"]
        self.discriminator_lr = training_config["discriminator_lr"]
        self.generator_lr = training_config["generator_lr"]
        self.num_epochs = training_config["num_epochs"]
        self.blocks_to_add = training_config["blocks_to_add"]
        self.timestamp = training_config["timestamp"]
        self.ml = training_config["ml"]
        self.fade_in = training_config["fade_in"]
        self.sa = training_config["sa"]
        self.save = training_config["save"]
        self.name = training_config["name"]
        self.gpu = training_config["gpu"]
        self.path = training_config["path"]
        self.device=utils.assign_device(self.gpu)
        
    def __call__(self,
                train_data = None,
                eval_data = None,
                plot_history : bool = False,
                *kargs):
        
        self.pipline(train_data=train_data, eval_data=eval_data, plot_history=plot_history)
        
    def pipline(self,
                train_data = None,
                eval_data = None,
                plot_history : bool = False,
                *kargs):
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)

        D = self.model["D"]
        G = self.model["G"]
        
        criterion = nn.MSELoss()
        self.device=utils.assign_device(self.gpu)

        optimD = Adam(D.parameters(), lr=self.discriminator_lr, betas=(0.9, 0.999))
        optimG = Adam(G.parameters(), lr=self.generator_lr, betas=(0.9, 0.999))
        #embedder=lstm.LSTMEncoder().to(device)
        #path="Models/M4/"
        embedder=torch.load("Models/Embedder/embedder_model.pt", map_location=torch.device('cpu')).to(self.device)

        activeG=(G.step-1)-self.blocks_to_add
        activeD=self.blocks_to_add

        utils.create_folder(self.path+self.name+'/')

        #Training
        g_losses = []
        d_losses = []
        fids = []
        G.to(self.device)
        D.to(self.device)
        fade=1
        sum_fade=0
        g_loss_min=1000000
        d_loss_min=1000000

        print()
        print("Starting training:",self.name)
        print("Total Epochs: %d \nBlocks to add with fade: %d\nTimestamp to add blocks: %d" % 
                            (self.num_epochs, self.blocks_to_add, self.timestamp))
        print("Fade-in",self.fade_in)
        print("ML", self.ml)
        print("SA", self.sa)
        
        
        
        
        for epoch in range(1,self.num_epochs+1):
                pbar_training = tqdm(total=len(train_loader),
                                     position=0,
                                     desc='Training, Epoch:{}/{}'.format(epoch,self.num_epochs))
                g_losses_temp=[]
                d_losses_temp=[]
                fids_temp=[]
                if (epoch%self.timestamp==0 and epoch!=0 and activeG!=G.step-1 and activeD!=0 and self.fade_in==True):
                    activeD-=1
                    activeG+=1
                    fade=0
                    sum_fade=1/((self.timestamp)/2)
                    print("Block added")

                elif(fade+sum_fade<=1 and self.fade_in==True):
                    fade+=sum_fade

                else:
                    fade=1

                for i, (X, Y) in enumerate((train_loader)):
                    X=X.to(self.device)
                    Y=Y.to(self.device)

                    # Generate fake data
                    fake_data = G(X,fade,activeG)
                    #fake_label = torch.zeros(Y.size(0))

                    # Train the discriminator
                    Y=Y[:,:,:fake_data.size(2)]  #we use this to adapt real sequences length to fake sequences length

                    D.zero_grad()
                    d_real_loss = criterion(D(Y,X,fade,activeD), torch.ones_like(D(Y,X,fade,activeD)))
                    d_fake_loss = criterion(D(fake_data.detach(),X,fade,activeD), torch.zeros_like(D(fake_data.detach(),X,fade,activeD)))
                    d_loss = d_real_loss + d_fake_loss
                    d_losses_temp.append(d_loss.item())
                    d_loss.backward(retain_graph=False)
                    optimD.step()

                    # Train the generator
                    G.zero_grad()
                    g_loss = criterion(D(fake_data,X,fade,activeD), torch.ones_like(D(fake_data,X,fade,activeD)))

                    if(self.ml==True):
                        # Add the moment loss
                        g_loss += utils.moment_loss(fake_data, Y)

                    g_losses_temp.append(g_loss.item())

                    g_loss.backward()
                    optimG.step()

                    #Compute FID
                    with torch.no_grad():
                        fake_embedding=embedder(fake_data)
                        real_embedding=embedder(Y) 
                        fid = metrics.calculate_fid(fake_embedding.to("cpu").detach().numpy(),real_embedding.to("cpu").detach().numpy())

                    fids_temp.append(fid)    

                    # Print the losses
                    if (i+1) % 1 == 0:
                        
                        pbar_training.set_postfix({"Epoch": epoch,
                                                   "D loss":d_loss.item(),
                                                   'G loss': d_loss.item(),
                                                   "Fade-in:": fade,
                                                   "FID": fid})
                        pbar_training.update(1)
                        
                        #print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Fade-in: %f] [FID: %f]" % 
                        #    (epoch, self.num_epochs, i+1, len(train_loader), d_loss.item(), g_loss.item(), fade, fid))
                    '''
                    if(g_loss<g_loss_min and d_loss<d_loss_min and save):
                            g_loss_min = g_loss
                            d_loss_min = d_loss
                            torch.save(G, path+name+'/'+name+'_generator.pt')
                            torch.save(D, path+name+'/'+name+'_discriminator.pt')
                            print('Improvement-Detected, model saved')
                    '''

                g_losses.append(torch.mean(torch.Tensor(g_losses_temp)))
                d_losses.append(torch.mean(torch.Tensor(d_losses_temp)))
                fids.append(torch.mean(torch.Tensor(fids_temp)))

        values=['Last G loss: '+str(g_losses[-1].item()), 
                'Last D loss: '+str(d_losses[-1].item()),
                'Last FID: '+str(fids[-1].item()),
                'epochs: '+str(self.num_epochs),
                'ML: '+str(self.ml),
                'SA: '+str(self.sa),
                'Fade-in: '+str(self.fade_in),
                'Blocks to add: '+str(self.blocks_to_add),
                'Timestamp: '+str(self.timestamp),
                ]
        torch.save(G, self.path+self.name+'/'+self.name+'_generator.pt')
        torch.save(D, self.path+self.name+'/'+self.name+'_discriminator.pt')
        
        if plot_history:
            plotting.plot_training_history('PSA-GAN - M4 - '+self.name,d_losses, g_losses)
            plotting.plot_fid_history('PSA-GAN - M4 - '+self.name, fids)
        location=self.path+'/'+self.name+'/'+self.name
        utils.write_file(location, values)
        
        return  {"d_losses": d_losses, "g_losses": g_losses, "fids": fids}