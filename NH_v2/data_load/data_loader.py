# -*-Encoding: utf-8 -*-
"""
Authors:
    Khalid OUBLAL, PhD IPP/ OneTech 
NB: Part of this code is froked from @Borsoi repo IDNet
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
import torch.distributions as td
import scipy
import numpy as np
from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt
import utils

import warnings
warnings.filterwarnings('ignore')


class StandardScaler(object):
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean



class dataNonlinear_synth():
    def __init__(self, args):
        '''data from synthetic example with nonlinear mixtures (BLMM)'''
        mat_contents1 = loadmat(args.data_path) # load image
        mat_contents2 = loadmat(args.data_path_extract) # load spectral libraries!!! 

        self.L = mat_contents1['Mth'].shape[0]
        self.P = mat_contents1['Mth'].shape[1]
        self.N = mat_contents1['Y'].shape[1]
        self.Nlib = mat_contents2['bundleLibs'][0,0].shape[1]
        self.Y = torch.from_numpy(mat_contents1['Y']).type(torch.float32)
        
        SNR = 40 
        ssigma = (self.Y.mean())*(10**(-SNR/10))
        
        self.data_sup = []
        for i in range(self.Nlib):
            M_s = torch.zeros((self.L,self.P))
            for j in range(self.P):
                m_ij = mat_contents2['bundleLibs'][0,j][:,i]                
                M_s[:,j] = torch.from_numpy(m_ij)            
            for k in range(self.P):
                a_s = torch.zeros((self.P,))
                a_s[k] = 1.0
                y_s = torch.mv(M_s,a_s) + ssigma * torch.randn((self.L,))
                self.data_sup.append((y_s,M_s,a_s))
                
        self.data_unsup = []
        for i in range(self.N):
            self.data_unsup.append(torch.from_numpy(mat_contents1['Y'][:,i]).type(torch.float32))
        self.A_cube_gt = torch.from_numpy(mat_contents1['A_cube'])
        self.A_gt = self.A_cube_gt.permute(1,0,2).reshape((self.N,self.P)).T
        self.Mavg_th = torch.from_numpy(mat_contents1['Mth'])
        self.Mn_th = -0.5*torch.ones(self.L,self.P,self.N) 
        
    def getdata(self):
        return self.Y, self.data_sup, self.data_unsup
    
    def get_gt(self):
        return self.A_gt, self.A_cube_gt, self.Mavg_th, self.Mn_th

        


        

class dataVariability_synth():
    def __init__(self, args):
        '''data from synthetic example with spectral variability'''
        mat_contents1 = loadmat(args.data_path) # load image
        mat_contents2 = loadmat(args.data_path_extract) # load spectral libraries!!!         
        self.L = mat_contents1['Mth'].shape[0]
        self.P = mat_contents1['Mth'].shape[1]
        self.N = mat_contents1['Mth'].shape[2]
        self.Nlib = mat_contents2['bundleLibs'][0,0].shape[1]
        self.Y = torch.from_numpy(mat_contents1['Y']).type(torch.float32)
        
        SNR = 40 
        ssigma = (self.Y.mean())*(10**(-SNR/10))
        
        self.data_sup = []
        for i in range(self.Nlib):
            M_s = torch.zeros((self.L,self.P))
            for j in range(self.P):
                m_ij = mat_contents2['bundleLibs'][0,j][:,i]                
                M_s[:,j] = torch.from_numpy(m_ij)            
            for k in range(self.P):
                a_s = torch.zeros((self.P,))
                a_s[k] = 1.0
                y_s = torch.mv(M_s,a_s) + ssigma * torch.randn((self.L,))
                self.data_sup.append((y_s,M_s,a_s))
                
        self.data_unsup = []
        for i in range(self.N):
            self.data_unsup.append(torch.from_numpy(mat_contents1['Y'][:,i]).type(torch.float32))
        self.A_gt = torch.from_numpy(mat_contents1['A'])
        self.A_cube_gt = torch.from_numpy(mat_contents1['A_cube'])
        self.Mavg_th = torch.from_numpy(mat_contents1['M_avg'])
        self.Mn_th = torch.from_numpy(mat_contents1['Mth'])
        
    def getdata(self):
        return self.Y, self.data_sup, self.data_unsup
    
    def get_gt(self):
        return self.A_gt, self.A_cube_gt, self.Mavg_th, self.Mn_th




class dataReal_real:
    def __init__(self,args):
        """
        Initialize data from real examples.

        :param example_name: Name of the example dataset. Options: "Samson", "Jasper Ridge", "Cuprite"
        """
        if args.data_option in ["Samson", "Jasper Ridge", "Cuprite", "Houston"]:
            mat_contents1 = loadmat(args.data_path) # load image
            mat_contents2 = loadmat(args.data_path_extract) # load spectral libraries!!! 
        else:
            raise ValueError(f"Invalid {args.data_option}. Please choose from 'Samson', 'Jasper Ridge', or 'Cuprite'")

        self.L = mat_contents1['M0'].shape[0]
        self.P = mat_contents1['M0'].shape[1]
        self.N = mat_contents1['Y'].shape[1]
        self.Nlib = mat_contents2['bundleLibs'][0,0].shape[1]
        self.nr, self.nc = mat_contents1['Yim'].shape[0], mat_contents1['Yim'].shape[1]
        self.Y = torch.from_numpy(mat_contents1['Y']).type(torch.float32)
        
        SNR = 40 
        ssigma = (self.Y.mean())*(10**(-SNR/10))
        
        self.data_sup = []
        for i in range(self.Nlib):
            M_s = torch.zeros((self.L,self.P))
            for j in range(self.P):
                # m_ij = mat_contents2['bundleLibs'][0,j][:,i]  
                m_ij = mat_contents2['bundleLibs'][0,j][:,i%mat_contents2['bundleLibs'][0,j].shape[1]] # circular shift
                # m_ij = mat_contents1['M0'][:,j]
                M_s[:,j] = torch.from_numpy(m_ij)
            for k in range(self.P):
                a_s = torch.zeros((self.P,))
                a_s[k] = 1.0
                y_s = torch.mv(M_s,a_s) + ssigma * torch.randn((self.L,))
                self.data_sup.append((y_s,M_s,a_s))
                
        self.data_unsup = []
        for i in range(self.N):
            self.data_unsup.append(torch.from_numpy(mat_contents1['Y'][:,i]).type(torch.float32))
        self.A_gt = -torch.ones((self.P,self.N))
        self.A_cube_gt = -torch.ones((self.nr,self.nc,self.P))
        self.Mavg_th = torch.from_numpy(mat_contents1['M0'])
        self.Mn_th = -torch.ones((self.L,self.P,self.N))
        
    def getdata(self):
        return self.Y, self.data_sup, self.data_unsup
    
    def get_gt(self):
        return self.A_gt, self.A_cube_gt, self.Mavg_th, self.Mn_th
    


class HyperspectralDataset(Dataset):
    def __init__(self, args, validation_flag=False):
        """
        Initialize variables and select which data to load.

        :param data_option: Option to select the type of data.
                            Options:
                            - "Synthetic Nonlinear Mixture" (DC1, with the BLMM)
                            - "Synthetic Example with Variability" (DC2)
                            - "Real Data Examples: Samson", "Jasper", "Cuprite"
        :param validation_flag: Flag indicating whether the dataset is for validation purposes.
        """
        self.data_sup = []
        self.data_unsup = []
        # The following is the data ground truth:
        self.A_u = []  # Ground truth abundance matrix (P * N)
        self.A_u_cube = []  # Ground truth abundance cube (nr * nc * P)
        self.M_u_avg = []  # Ground truth 'average' or 'reference' EM matrix
        self.M_u_ppx = []  # Ground truth EM matrices for each pixel
        self.Y = []  # Observed hyperspectral image (L * N)
        self.validation = validation_flag
        self.data_option = args.data_option  # Store data index for later access

        if args.data_option == "Synthetic_Nonlinear_Mixture":
            self.NonlinearData_synth(args)
        elif args.data_option == "Synthetic_Variability":
            self.VariabilityData_synth(args)
        elif args.data_option in ["Samson", "Jasper", "Cuprite", "Houston"]:
            self.RealData_real(args)
        else:
            raise ValueError("Invalid data_option. Please choose from available options.")
        
        if len(self.data_sup) < len(self.data_unsup):
            self.flag_unsup_is_bigger = True
        else:
            self.flag_unsup_is_bigger = False

        #self._spliter_data() # later
            
    def NonlinearData_synth(self, args):
        myDatator = dataNonlinear_synth(args)
        self.Y, self.data_sup, self.data_unsup = myDatator.getdata()
        self.A_u, self.A_u_cube, self.M_u_avg, self.M_u_ppx = myDatator.get_gt()
    
    def VariabilityData_synth(self, args):
        myDatator = dataVariability_synth(args)
        self.Y, self.data_sup, self.data_unsup = myDatator.getdata()
        self.A_u, self.A_u_cube, self.M_u_avg, self.M_u_ppx = myDatator.get_gt()
    
    def RealData_real(self, args):
        myDatator = dataReal_real(args)
        self.Y, self.data_sup, self.data_unsup = myDatator.getdata()
        self.A_u, self.A_u_cube, self.M_u_avg, self.M_u_ppx = myDatator.get_gt()
        
    def _spliter_data(self, size_val_percent=20):
        if self.validation:
            size_val = len(self.A_u_cube) //2
            for k in range(len(self.Y)):
                size_val = len(self.Y[k])//2
                self.Y[k] = self.Y[k][:size_val]   

            for p in range(len(self.data_sup)):
                size_val = len(self.Y[k])//2
                self.data_sup[p] = self.data_sup[p][:size_val]

            for p in range(len(self.data_unsup)):
                size_val = len(self.Y[k])//2
                self.data_unsup[p] =  self.data_unsup[p][:size_val]

            size_val = len(self.A_u_cube) //2
            self.A_u, self.A_u_cube, self.M_u_avg, self.M_u_ppx = self.A_u[:size_val], self.A_u_cube[:size_val], self.M_u_avg[:size_val], self.M_u_ppx[:size_val]
            
        
    def __len__(self):
        # take the maximum length between the supervised and unsupervised datasets
        return max(len(self.data_sup),len(self.data_unsup))

    def __getitem__(self, idx):
        # now, idx corresponds to the index among the largest (sup or unsup) dataset.
        # We can multiply it by the ratio between the smallest and the largest datset
        # and round down to an integer, to obtain the corresponding index for the 
        # smaller dataset
        if self.flag_unsup_is_bigger:
            idx_sup = int(np.floor(idx*len(self.data_sup)/len(self.data_unsup)))
            idx_unsup = idx
        else:
            idx_sup = idx
            idx_unsup = int(np.floor(idx*len(self.data_unsup)/len(self.data_sup)))
        # return tuples? (y) and (y,M,a), we add all values also

        x_data = [self.data_unsup[idx_unsup], self.data_sup[idx_sup], (self.A_u.T[idx_unsup], self.M_u_ppx.transpose(0,2)[idx_unsup], self.Y.T[idx_unsup])]
        y_unsup = x_data[0].unsqueeze(-1)
        y_sup = x_data[1][0].unsqueeze(-1) # batch * L
        M_sup = x_data[1][1]
        a_sup = x_data[1][2].unsqueeze(-1)
        return y_unsup, y_sup, M_sup, a_sup
    

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=1, freq='t', cols=None, percentage=0.05):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.percentage = percentage
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        length = len( df_raw)*self.percentage
        num_train = int(length*0.1)
        num_test = int(length*0.3)
        num_vali = int(length*0.3)
        # num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, num_train+num_vali-self.seq_len]
        border2s = [num_train, num_train+num_vali, num_train+num_vali+num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            
            for i in range(len(self.scaler.std)):
                if self.scaler.std[i] == 0:
                    print(i)
            # print(len(self.scaler.std))
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp = pd.date_range(start='4/1/2018',periods=border2-border1, freq='H')
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end, -1:]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
        
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
