

<h1 align="left">
<!--
    <a href="https://oublalkhalid.github.io/time-disentanglement-lib/">
        <img src="docs/img/hugging_time_disentanglement_lib.png" width="80" height="80" alt="Time Disentanglement-lib">
    </a>
-->
    Time-Disentanglement-Lib
</h1>

##  Disentangling Time Series Representations via Contrastive Independence-of-Support on $l$-Variational Inference
📣 Published as a conference paper at ICLR 2024 
![An overview](docs/img/model.png)

**Note ⚠️**
- Currently, we updated some classes of our  framework "DIoSC" Time Series Disentangling for correlated data.

At present, this repository remains anonymous as a paper based on its content is under review. It provides procedures to enhance the disentanglement of time series data, offering both configurations and the necessary code to reproduce our results.

| Loss               | Source                                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| Standard VAE Loss  | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)                             |
| β-VAE<sub>H</sub>  | [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl)   |
| β-VAE<sub>B</sub>  | [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599)                       |
| S3VAE          | [Disentangling by Factorising](https://arxiv.org/abs/1802.05983)                                |
| C-DSVAE          | [Disentangling by Factorising](https://arxiv.org/abs/1802.05983)                                |
| CoST          | [Disentangling by Factorising](https://arxiv.org/abs/1802.05983)                                |
| Probabilistic Transformer          | [Disentangling by Factorising](https://arxiv.org/abs/1802.05983)                                |
| Autoformers          | [Disentangling by Factorising](https://arxiv.org/abs/1802.05983)                                |
| β-TCVAE            | [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942) |
|  HFS  | [https://openreview.net/forum?id=OKcJhpQiGiX](https://openreview.net/forum?id=OKcJhpQiGiX)                       |
|     RNN-VAE       | [A Recurrent Latent Variable Model for Sequential Data](https://proceedings.neurips.cc/paper_files/paper/2015/file/b618c3210e934362ac261db280128c22-Paper.pdf)                                |
| D3VAE            | [Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement](https://arxiv.org/abs/2301.03028) |
| DIoSC Time Series (ours)  | [Disentangling Time Series Representations via Contrastive based L-Variational Inference](#)                        |
| MSA-Conv for Time Series  | [Disentangling Time Series Representations via Contrastive based L-Variational Inference](#)                        |


## Installation

To get started with the Disentangling Time Series Energy codebase, follow these steps:

```shell
# Clone the repository
pip install -r requirements.txt
```

### With pip
```shell
pip install time-disentnaglement
```

## Run
Use `python main.py <model-name> <param>` to train and/or evaluate a model. For example:

```
python main.py DIoSC_ukdal_mini -d ukdal -l DIoSC --lr 0.001 -b 256 -e 5
```

Predefined experiments with associated hyperparameters can be executed using the -x <experiment> flag. The hyperparameters can be found in the hyperparam.ini file, and pretrained models for each experiment are located in the results/<experiment> directory (created using ./bin/train_all.sh).

Output
Running experiments will create a directory results/<saving-name>/, which includes the following:

### use Checkpoint 

We offer checkpoints for each model, and we are actively working on providing access to other experiments using W&B as well.



### Help
```
usage: main.py ...

PyTorch implementation and evaluation of disentangled Variational AutoEncoders
and metrics.

optional arguments:
  -h, --help            show this help message and exit

General options:
  name                  Name of the model for storing or loading purposes.
  -L, --log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        Logging levels. (default: info)
  --no-progress-bar     Disables progress bar. (default: False)
  --no-cuda             Disables CUDA training, even when have one. (default:
                        False)
  -s, --seed SEED       Random seed. Can be `None` for stochastic behavior.
                        (default: 1234)

Training specific options:
  --checkpoint-every CHECKPOINT_EVERY
                        Save a checkpoint of the trained model every n epoch.
                        (default: 30)
  -d, --dataset {mnist,fashion,dsprites,celeba,chairs}
                        Path to training data. (default: mnist)
  -x, --experiment {custom,debug,best_celeba,VAE_mnist,VAE_fashion,VAE_dsprites,VAE_celeba,VAE_chairs,betaH_mnist,betaH_fashion,betaH_dsprites,betaH_celeba,betaH_chairs,betaB_mnist,betaB_fashion,betaB_dsprites,betaB_celeba,betaB_chairs,factor_mnist,factor_fashion,factor_dsprites,factor_celeba,factor_chairs,btcvae_mnist,btcvae_fashion,btcvae_dsprites,btcvae_celeba,btcvae_chairs}
                        Predefined experiments to run. If not `custom` this
                        will overwrite some other arguments. (default: custom)
  -e, --epochs EPOCHS   Maximum number of epochs to run for. (default: 100)
  -b, --batch-size BATCH_SIZE
                        Batch size for training. (default: 64)
  --lr LR               Learning rate. (default: 0.0005)

Model specfic options:
  -m, --model-type {Burgess}
                        Type of encoder and decoder to use. (default: Burgess)
  -z, --latent-dim LATENT_DIM
                        Dimension of the latent variable. (default: 10)
  -l, --loss {VAE,betaH,betaB,factor,btcvae}
                        Type of VAE loss function to use. (default: betaB)
  -r, --rec-dist {bernoulli,laplace,gaussian}
                        Form of the likelihood ot use for each pixel.
                        (default: bernoulli)
  -a, --reg-anneal REG_ANNEAL
                        Number of annealing steps where gradually adding the
                        regularisation. What is annealed is specific to each
                        loss. (default: 0)
```

