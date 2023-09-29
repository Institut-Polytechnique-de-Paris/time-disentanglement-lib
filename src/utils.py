import numpy as np
import torch
import os
import gc
from sklearn.preprocessing import MinMaxScaler

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]    #the next has to be predicted
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


def real_seq(data,seq_length):

    y=[]
    for i in range(seq_length,len(data)-seq_length-1):
        _y = data[i:(i+seq_length)]
        y.append(_y)
    return np.array(y)

    
def noise(Xt, device):
    bs = Xt.size(0)
    target_len = Xt.size(2)
    noise = torch.randn((bs, 1, target_len))
    noise=noise.to(device)
    Xt=Xt.to(device)
    Xt = torch.cat((Xt, noise), dim=1)
    return Xt


def create_folder(path):
    path = path
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print("New models directory created!:",path)
    else:
        print("Directory already exists:",path)
    return

def load_model(path,name):
    G = torch.load(path+'/'+name+'/'+name+'_generator.pt')
    D = torch.load(path+'/'+name+'/'+name+'_discriminator.pt')
    print("Model Loaded succesfully: ",name)
    return G,D


def scale(generated):
    sc = MinMaxScaler()
    generated = sc.fit_transform(generated)
    return generated


def assign_device(gpu):
    if (torch.cuda.is_available() and gpu==True):
        device = "cuda"
        print("Cuda enabled: using GPU")
    else:
        device = "cpu"
        print("Cuda not available: using CPU")
    return device


def moment_loss(fake_data, real_data):
    fake_mean = fake_data.mean()
    real_mean = real_data.mean()
    fake_std = fake_data.std()
    real_std = real_data.std()
    return abs(fake_mean - real_mean) + abs(fake_std - real_std)


def write_file(path, values):
    with open(path+'.txt', 'w') as file:
        for value in values:
            file.write(str(value) + '\n')

def free_gpu(G):
    
    G.cpu()
    del G
    gc.collect()
    torch.cuda.empty_cache()
    return
