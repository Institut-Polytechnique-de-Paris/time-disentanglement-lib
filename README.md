#  Disentangling Time Series Representations via Contrastive based L-Variational Inference
![An overview](doc/img/overview.png)
**Note ⚠️**
- Currently, we updated some classes of our  framework, please use our last release (v0.2.1-alpha)


At present, this repository remains anonymous as a paper based on its content is under review. It provides procedures to enhance the disentanglement of time series data, offering both configurations and the necessary code to reproduce our results.

| Loss               | Source                                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| Standard VAE Loss  | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)                             |
| β-VAE<sub>H</sub>  | [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl)   |
| β-VAE<sub>B</sub>  | [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599)                       |
| FactorVAE          | [Disentangling by Factorising](https://arxiv.org/abs/1802.05983)                                |
| β-TCVAE            | [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942) |

|  HFS  | [https://openreview.net/forum?id=OKcJhpQiGiX](https://openreview.net/forum?id=OKcJhpQiGiX)                       |
|     RNN-VAE (VRNN)      | [A Recurrent Latent Variable Model for Sequential Data](https://proceedings.neurips.cc/paper_files/paper/2015/file/b618c3210e934362ac261db280128c22-Paper.pdf)                                |
| D3VAE            | [Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement](https://arxiv.org/abs/2301.03028) |
| DisCo Time Series (ours)  | [Disentangling Time Series Representations via Contrastive based L-Variational Inference](#)                        |


# Disentangling Time Series Energy

## Installation

To get started with the Disentangling Time Series Energy codebase, follow these steps:

```shell
# Clone the repository
pip install -r requirements.txt
```

## Run
Use `python main.py <model-name> <param>` to train and/or evaluate a model. For example:

```
python main.py disco_ukdal_mini -d ukdal -l disco --lr 0.001 -b 256 -e 5
```

Predefined experiments with associated hyperparameters can be executed using the -x <experiment> flag. The hyperparameters can be found in the hyperparam.ini file, and pretrained models for each experiment are located in the results/<experiment> directory (created using ./bin/train_all.sh).

Output
Running experiments will create a directory results/<saving-name>/, which includes the following:

model.pt: The model at the end of training.
model-i.pt: Model checkpoint after i iterations (default: every 10).
specs.json: Parameters used to run the program (default and modified with CLI).
training.gif: A GIF showing latent traversals of the latent dimensions Z at every epoch of training.
train_losses.log: Log of all (sub-)losses computed during training.
test_losses.log: Log of all (sub-)losses computed at the end of training with the model in evaluate mode (no sampling).
metrics.log: Includes the Mutual Information Gap metric and the Axis Alignment Metric. This is included only if --is-metric is enabled (but it may run slowly).

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

BetaH specific parameters:
  --betaH-B BETAH_B     Weight of the KL (beta in the paper). (default: 4)

BetaB specific parameters:
  --betaB-initC BETAB_INITC
                        Starting annealed capacity. (default: 0)
  --betaB-finC BETAB_FINC
                        Final annealed capacity. (default: 25)
  --betaB-G BETAB_G     Weight of the KL divergence term (gamma in the paper).
                        (default: 1000)

factor VAE specific parameters:
  --factor-G FACTOR_G   Weight of the TC term (gamma in the paper). (default:
                        6)
  --lr-disc LR_DISC     Learning rate of the discriminator. (default: 5e-05)

beta-tcvae specific parameters:
  --btcvae-A BTCVAE_A   Weight of the MI term (alpha in the paper). (default:
                        1)
  --btcvae-G BTCVAE_G   Weight of the dim-wise KL term (gamma in the paper).
                        (default: 1)
  --btcvae-B BTCVAE_B   Weight of the TC term (beta in the paper). (default:
                        6)

Evaluation specific options:
  --is-eval-only        Whether to only evaluate using precomputed model
                        `name`. (default: False)
  --is-metrics          Whether to compute the disentangled metrcics.
                        Currently only possible with `dsprites` as it is the
                        only dataset with known true factors of variations.
                        (default: False)
  --no-test             Whether not to compute the test losses.` (default:
                        False)
  --eval-batchsize EVAL_BATCHSIZE
                        Batch size for evaluation. (default: 1000)
```


Here are examples of plots you can generate:

* `python main_viz.py <model> reconstruct-traverse --is-show-loss --is-posterior` first row are originals, second are reconstructions, rest are traversals. Shown for `btcvae_dsprites`:

    ![btcvae_dsprites reconstruct-traverse](results/btcvae_dsprites/reconstruct_traverse.png)

* `python main_viz.py <model> gif-traversals` grid of gifs where rows are latent dimensions, columns are examples, each gif shows posterior traversals. Shown for `btcvae_celeba`:

    ![btcvae_celeba gif-traversals](results/btcvae_celeba/posterior_traversals.gif)

* Grid of gifs generated using code in `bin/plot_all.sh`. The columns of the grid correspond to the datasets (besides FashionMNIST), the rows correspond to the models (in order: Standard VAE, β-VAE<sub>H</sub>, β-VAE<sub>B</sub>, FactorVAE, β-TCVAE):

    ![grid_posteriors](results/grid_posteriors.gif)

For more examples, all of the plots for the predefined experiments are found in their respective directories (created using `./bin/plot_all.sh`).

## Data

Current datasets that can be used:
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
- [3D Chairs](https://www.di.ens.fr/willow/research/seeing3Dchairs)
- [Celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [2D Shapes / Dsprites](https://github.com/deepmind/dsprites-dataset/)

The dataset will be downloaded the first time you run it and will be stored in `data` for future uses. The download will take time and might not work anymore if the download links change. In this case either:

1. Open an issue
2. Change the URLs (`urls["train"]`) for the dataset you want in `utils/datasets.py` (please open a PR in this case :) )
3. Download by hand the data and save it with the same names (not recommended)

## Our Contributions

In addition to replicating the aforementioned papers, we also propose and investigate the following:

### Axis Alignment Metric

Qualitative inspections are unsuitable to compare models reliably due to their subjective and time consuming nature. Recent papers use quantitative measures of disentanglement based on the ground truth factors of variation **v** and the latent dimensions **z**. The [Mutual Information Gap (MIG)](https://arxiv.org/abs/1802.04942) metric is an appealing information theoretic metric which is appealing as it does not use any classifier. To get a MIG of 1 in the dSprites case where we have 10 latent dimensions and 5 generative factors, 5 of the latent dimensions should exactly encode the true factors of variations, and the rest should be independent of these 5.

Although a metric like MIG is what we would like to use in the long term, current models do not get good scores and it is hard to understand what they should improve. We thus propose an axis alignment metric AAM, which does not focus on how much information of **v** is encoded by **z**, but rather if each v<sub>k</sub> is only encoded in a single z<sub>j</sub>. For example in the dSprites dataset, it is possible to get an AAM of 1 if **z** encodes only 90% of the variance in the x position of the shapes as long as this 90% is only encoded by a single latent dimension z<sub>j</sub>. This is a useful metric to have a better understanding of what each model is good and bad at. Formally:

![Axis Alignment Metric](doc/imgs/aam.png)

Where the subscript *(d)* denotes the *d*<sup>th</sup> order statistic and *I*<sub>x</sub> is estimated using empirical distributions and stratified sampling (like with MIG):

![Mutual Information for AAM](doc/imgs/aam_helper.png)


### Single Model Comparison

The model is decoupled from all the losses and it should thus be very easy to modify the encoder / decoder without modifying the losses. We only used a single model in order to have more objective comparisons of the different losses. The model used is the one from [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599), which is summarized below:

![Model Architecture](doc/imgs/architecture.png)


# Loss Terms and Hyperparameters

1. **Index-code mutual information**: This measures the mutual information between latent variables **z** and data variable **x**. There is debate in the literature on how to treat this term. According to the [information bottleneck perspective](https://arxiv.org/abs/1706.01350), it should be penalized. Conversely, [InfoGAN](https://arxiv.org/abs/1606.03657) achieves good results by increasing mutual information (with a negative α), while [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558) omits this term.

2. **Total Correlation (TC)**: TC represents the Kullback-Leibler (KL) divergence between the joint distribution of latent variables and the product of their marginals. It quantifies the dependence between latent dimensions. Increasing the hyperparameter β encourages the model to discover statistically independent factors in the data distribution.