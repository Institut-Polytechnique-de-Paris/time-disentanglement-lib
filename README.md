<h1 align="center"\>
Time-Disentanglement-Lib
</h1\>

<div align="center">

[](https://iclr.cc/)
[](https://pytorch.org/)
[](https://www.google.com/search?q=LICENSE)

**A comprehensive PyTorch library for Disentangled Representation Learning in Time Series.**

</div>



> **News 📣:** Our paper *"Disentangling Time Series Representations via Contrastive Independence-of-Support on $l$-Variational Inference"* has been published at **ICLR**.

## Overview

**Time-Disentanglement-Lib** is a modular framework designed to facilitate research in disentangled representation learning for sequential data. Unlike standard static disentanglement libraries, this repository focuses on the temporal dynamics, offering state-of-the-art (SOTA) models, including our proposed **DIoSC** framework.

This library provides:

1.  **SOTA Baselines:** Implementations of leading time-series and static disentanglement models.
2.  **Evaluation Metrics:** A suite of quantitative metrics to measure disentanglement quality.
3.  **Reproducibility:** Pre-configured experiments for datasets like UK-DALE, MNIST, dSprites, and more.
4.  
-----

## 🏗️ Supported Models

We support a wide range of models, categorized into Time-Series specific architectures and General VAE-based frameworks.

### Time-Series & Sequential Models

| Model | Type | Paper / Citation |
| :--- | :--- | :--- |
| **DIoSC (Ours)** | Contrastive/HVAE | [Disentangling Time Series Representations via Contrastive Independence-of-Support](https://www.google.com/search?q=%23) |
| **S3VAE** | Sequential VAE | [Self-Supervised Sequential VAE (Li et al., 2018)](https://arxiv.org/abs/1808.01955) |
| **CoST** | Contrastive | [Contrastive Seasonal-Trend Decomposition (Woo et al., ICLR 2022)](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3DCi3amM3H56d) |
| **D3VAE** | Diffusion/VAE | [Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement (Yang et al., 2023)](https://arxiv.org/abs/2301.03028) |
| **RNN-VAE** | Recurrent | [A Recurrent Latent Variable Model for Sequential Data (Chung et al., NeurIPS 2015)](https://proceedings.neurips.cc/paper/2015/hash/b618c3210e934362ac261db280128c22-Abstract.html) |
| **Autoformer** | Transformer | [Autoformer: Decomposition Transformers for Long-Term Series Forecasting (Wu et al., NeurIPS 2021)](https://arxiv.org/abs/2106.13008) |
| **Probabilistic Transformer** | Transformer | [Deep Transformer Models for Time Series Forecasting (Wen et al., 2020)](https://arxiv.org/abs/2002.06103) |

### General Disentanglement Baselines

| Model | Paper / Citation |
| :--- | :--- |
| **Standard VAE** | [Auto-Encoding Variational Bayes (Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114) |
| **β-VAE (H)** | [β-VAE: Learning Basic Visual Concepts (Higgins et al., ICLR 2017)](https://openreview.net/pdf?id=Sy2fzU9gl) |
| **β-VAE (B)** | [Understanding disentangling in β-VAE (Burgess et al., 2018)](https://arxiv.org/abs/1804.03599) |
| **β-TCVAE** | [Isolating Sources of Disentanglement (Chen et al., NeurIPS 2018)](https://arxiv.org/abs/1802.04942) |
| **FactorVAE** | [Disentangling by Factorising (Kim & Mnih, ICML 2018)](https://arxiv.org/abs/1802.05983) |

-----

## ebox 📏 Evaluation Metrics

To rigorously evaluate the quality of the learned representations, we implement the following standard disentanglement metrics:

| Metric | Description | Citation |
| :--- | :--- | :--- |
| **Beta-VAE Score** | Measures accuracy of a linear classifier predicting the fixed factor of variation. | [Higgins et al., 2017](https://openreview.net/forum?id=Sy2fzU9gl) |
| **FactorVAE Score** | Majority vote classifier accuracy on the index of the fixed generative factor. | [Kim & Mnih, 2018](https://arxiv.org/abs/1802.05983) |
| **MIG** | **Mutual Information Gap**: The difference in mutual information between the top two latent variables sharing info with a factor. | [Chen et al., 2018](https://arxiv.org/abs/1802.04942) |
| **DCI** | **Disentanglement, Completeness, Informativeness**: Uses importance weights from a regressor to quantify disentanglement. | [Eastwood & Williams, 2018](https://openreview.net/forum?id=By-7dz-AZ) |
| **SAP Score** | **Separated Attribute Predictability**: The difference in prediction error between the top two most predictive latent dimensions. | [Kumar et al., 2017](https://arxiv.org/abs/1711.00848) |
| **Modularity** | Measures if each latent dimension depends on at most one factor of variation. | [Ridgeway & Mozer, 2018](https://arxiv.org/abs/1802.05312) |
| **UDR** | **Unsupervised Disentanglement Ranking**: A correlation-based metric for model selection without ground truth. | [Duan et al., 2020](https://arxiv.org/abs/1907.02544) |

-----

## 💻 Installation

To get started with the codebase, clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/time-disentanglement-lib.git
cd time-disentanglement-lib

# Install requirements
pip install -r requirements.txt
```

Alternatively, if you are installing as a package (coming soon to PyPI):

```bash
pip install time-disentanglement
```

-----

## 🚀 Quick Start

You can train and evaluate models using the `main.py` entry point.

### Basic Training

Train the DIoSC model on the UK-DALE dataset:

```bash
python main.py DIoSC_ukdal_mini -d ukdal -l DIoSC --lr 0.001 -b 256 -e 50
```

### Running Predefined Experiments

We provide configuration files for reproducibility. Use the `-x` flag to run specific benchmarks:

```bash
# Run Beta-TCVAE on Temporal CausalIden
python main.py -x btcvae_causalIden

# Run CoST on a time series dataset
python main.py -x cost_ukdal
```

*Note: Hyperparameters are stored in `hyperparam.ini`. Pretrained models will be saved in `results/<experiment_name>/`.*

### Command Line Arguments

```text
usage: main.py [-h] [-d DATASET] [-x EXPERIMENT] [-l LOSS] ...

Time-Disentanglement-Lib: A library for sequential representation learning.

Options:
  -h, --help            Show this help message.
  -d, --dataset         Dataset to use (e.g., mnist, dsprites, ukdal).
  -x, --experiment      Predefined experiment name (loads config from .ini).
  -l, --loss            Loss/Model type (e.g., DIoSC, CoST, betaH, btcvae).
  --lr LR               Learning rate.
  -b, --batch-size      Batch size.
  -e, --epochs          Number of training epochs.
  -s, --seed            Random seed for reproducibility.
  --no-cuda             Force CPU execution.
```

-----

## 📜 Citation

If you use **DIoSC** or this library ``time-disentnaglement-lib`` in your research, please cite our ICLR 2024 paper:

```bibtex
@inproceedings{oublal2024disentangling,
  title={Disentangling time series representations via contrastive independence-of-support on l-variational inference},
  author={Oublal, Khalid and Ladjal, Said and Benhaiem, David and LE BORGNE, Emmanuel and Roueff, Fran{\c{c}}ois},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
---
