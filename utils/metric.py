import numpy as np
import torch
import scipy
from tqdm import tqdm

def mse(fake_data, real_data):
    return np.mean(np.square(real_data-fake_data))


def abs_error(fake_data, real_data):
    return np.sum(np.abs(real_data-fake_data))


def nrmse(fake_data, real_data):
    return torch.sqrt(torch.mean((real_data-fake_data)**2)) / (torch.max(real_data)-torch.min((real_data)))
    #return np.sqrt(np.mean((real_data-fake_data)**2)) / (np.max(real_data)-np.min((real_data)))



def evaluation(eval, test_loader, G, embedder=None):
    results=[]
    with torch.no_grad():
        device='cuda'
        G.to(device)
        embedder=embedder.to(device)
        for i, (X, Y) in tqdm(enumerate((test_loader)), total=len(test_loader)):

            X=X.to(device)
            Y=Y.to(device)
            
            fake_data = G(X,1,(G.step-1))
            real_data=Y[:,:,:fake_data.size(2)]

            if(eval=="FID"):
                fake_embedding=embedder(fake_data)
                real_embedding=embedder(real_data) 

                result= calculate_fid(real_embedding.to("cpu").detach().numpy(), fake_embedding.to("cpu").detach().numpy())
                results.append(result)                  
            
            else:
                fake_data=fake_data.permute(0,2,1)
                fake_data = fake_data.to("cpu").detach().numpy()
                real_data = real_data.to("cpu").detach().numpy()

                #fake_data=utils.scale(fake_data)
                #real_data=utils.scale(real_data)

                if(eval=="NRMSE"):
                    result=nrmse(fake_data, real_data)
                    results.append(result)
                elif(eval=="MSE"):
                    result=mse(fake_data, real_data)
                    results.append(result)
                elif(eval=="ABS"):
                    result=abs_error(fake_data, real_data)
                    results.append(result)                
                else:
                    print("error")
                    break

    res_mean=np.mean(results)
    res_stdev=np.std(results)

    return res_mean, res_stdev, results


def calculate_fid(generated_embeddings,real_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
    

def evaluate_model(model, test_loader,eval, embedder=None):
    device="cuda"
    model.eval()
    model.to(device)

    if(eval=="NRMSE"):
        total_nrmse = 0
        with torch.no_grad():
            for i,(X, Y) in tqdm(enumerate(test_loader), total=len(test_loader)):

                X=X.to(device)
                Y=Y.to(device)
                #fake_data = model(X,1,model.step-1).to("cpu")
                fake_data = model(X,1,(model.step-1))
                batch_nrmse = nrmse(fake_data, Y)
                #total_nrmse += batch_nrmse * Y.shape[0]
                total_nrmse += batch_nrmse

        avg_nrmse = total_nrmse / len(test_loader.dataset)
        return avg_nrmse.item()

    elif(eval=="FID"):
        embedder.to(device)
        total_fid=0
        with torch.no_grad():
            for i,(X, Y) in tqdm(enumerate(test_loader), total=len(test_loader)):
                
                X=X.to(device)
                Y=Y.to(device)
                fake_data = model(X,1,model.step-1)
                fake_embedding=embedder(fake_data)
                real_embedding=embedder(Y)
                fid = calculate_fid(real_embedding.to("cpu").detach().numpy(), fake_embedding.to("cpu").detach().numpy())
                    
                #total_fid += fid * Y.shape[0]
                total_fid += fid

        avg_fid = total_fid / len(test_loader.dataset)
        return avg_fid.item()

    else:
        print("ERROR")
        return 0


def run_evaluation(G1,G2,G3, test_loader,eval, confidence=95, embedder=None):
    results = []
    
    res1=evaluate_model(G1, test_loader,eval, embedder)
    res2=evaluate_model(G2, test_loader,eval, embedder)
    res3=evaluate_model(G3, test_loader,eval, embedder)
    results.append(res1)
    results.append(res2)
    results.append(res3)

    res_mean = np.mean(results)
    res_stdev = np.std(results)

    alpha=1.0-confidence
    p = ((alpha)/2.0)*100
    lower=max(0.0, np.percentile(res_mean,p))
    p = ((alpha)/2.0)*100
    upper=min(1.0, np.percentile(res_mean,p))
    return res_mean, res_stdev, lower, upper


def run_evaluation1(G1, test_loader,eval, confidence=95, embedder=None):
    results = []
    
    res1=evaluate_model(G1, test_loader,eval, embedder)
    results.append(res1)

    res_mean = np.mean(results)
    res_stdev = np.std(results)

    alpha=1.0-confidence
    p = ((alpha)/2.0)*100
    lower=max(0.0, np.percentile(res_mean,p))
    p = ((alpha)/2.0)*100
    upper=min(1.0, np.percentile(res_mean,p))
    return res_mean, res_stdev, lower, upper
