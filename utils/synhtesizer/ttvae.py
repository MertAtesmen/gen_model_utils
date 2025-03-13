from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Parameter
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .gmm_data_transformer import GMMDATATransformer


class Encoder_T(nn.Module):
    def __init__(self, input_dim, latent_dim, embedding_dim, nhead, dim_feedforward=2048, dropout=0.1):
      super(Encoder_T, self).__init__()
      # Input data to Transformer
      self.linear = nn.Linear(input_dim,embedding_dim)
      # Transformer Encoder
      self.transformerencoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
      self.encoder = nn.TransformerEncoder(self.transformerencoder_layer, num_layers=2)
      # Latent Space Representation
      self.fc_mu = nn.Linear(embedding_dim, latent_dim)
      self.fc_log_var = nn.Linear(embedding_dim, latent_dim)

    def forward(self, x):
      # Encoder
      x = self.linear(x)
      enc_output = self.encoder(x)
      # Latent Space Representation
      mu = self.fc_mu(enc_output)
      logvar = self.fc_log_var(enc_output)
      std = torch.exp(0.5 * logvar)
      return mu, std, logvar, enc_output


class Decoder_T(nn.Module):
    def __init__(self, input_dim, latent_dim, embedding_dim, nhead, dim_feedforward=2048, dropout=0.1):
      super(Decoder_T, self).__init__()
      # Linear layer for mapping latent space to decoder input size
      self.latent_to_decoder_input = nn.Linear(latent_dim, embedding_dim)
      # Transformer Decoder
      self.transformerdecoder_layer = nn.TransformerDecoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
      self.decoder = nn.TransformerDecoder(self.transformerdecoder_layer, num_layers=2)
      # Transformer Embedding to input
      self.linear = nn.Linear(embedding_dim,input_dim)
      self.sigma = Parameter(torch.ones(input_dim) * 0.1)

    def forward(self, z, enc_output):
      # Encoder
      z_decoder_input = self.latent_to_decoder_input(z)
      # Decoder
      # Note: Pass enc_output (memory) to the decoder
      dec_output = self.decoder(z_decoder_input, enc_output)

      return self.linear(dec_output), self.sigma


class TTVAE():
    """TTVAE."""

    def __init__(
        self,
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        latent_dim =32,# Example latent dimension
        embedding_dim=128,# Transformer embedding dimension
        nhead=8,# Number of attention heads
        dim_feedforward=1028,# Feedforward layer dimension
        dropout=0.1,
        cuda=True,
        verbose=False,
        device='cuda'
    ):
        self.latent_dim=latent_dim
        self.embedding_dim = embedding_dim
        self.nhead=nhead
        self.dim_feedforward=dim_feedforward
        self.dropout=dropout
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self._device = torch.device(device)

    # @random_state
    def fit(self, train_data, discrete_columns=(),
            # save_path=''
        ):
        self.transformer = GMMDATATransformer()
        self.transformer.fit(train_data, discrete_columns)

        train_data = self.transformer.transform(train_data).astype('float32')
        dataset = TensorDataset(torch.from_numpy(train_data).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions

        print(data_dim, self.latent_dim, self.embedding_dim, self.nhead, self.dim_feedforward, self.dropout)

        self.encoder = Encoder_T(data_dim, self.latent_dim, self.embedding_dim, self.nhead, self.dim_feedforward, self.dropout).to(self._device)
        self.decoder = Decoder_T(data_dim, self.latent_dim, self.embedding_dim, self.nhead, self.dim_feedforward, self.dropout).to(self._device)

        optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

        self.encoder.train()
        self.decoder.train()

        best_loss = float('inf')
        patience = 0
        start_time = time.time()

        for epoch in range(self.epochs):
        
            batch_loss = 0.0
            len_input = 0

            for id_, data in enumerate(loader):
                optimizer.zero_grad()
                real_x = data[0].to(self._device)
                mean, std, logvar, enc_output = self.encoder(real_x)
                z = reparameterize(mean, logvar)
                recon_x, sigmas = self.decoder(z,enc_output)
                loss = _loss_function_MMD(recon_x, real_x, sigmas, mean, logvar, self.transformer.output_info_list, self.loss_factor)

                batch_loss += loss.item() * len(real_x)
                len_input += len(real_x)

                loss.backward()
                optimizer.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)

        with torch.no_grad():
            mean, std, logvar, enc_embed= self.encoder(torch.Tensor(train_data).to(self._device))
        
        self.mean = mean
        self.std = std
        self.logvar = logvar
        self.enc_embed = enc_embed
                        
    # @random_state
    def sample(self, n_samples=100):
        """Sample data similar to the training data.

        """
        embeddings = torch.normal(mean=self.mean, std=self.std).cpu().detach().numpy()
        synthetic_embeddings=z_gen(embeddings,n_to_sample=n_samples,metric='minkowski',interpolation_method='SMOTE')
        noise = torch.Tensor(synthetic_embeddings).to(self._device)

        self.decoder.eval()
        with torch.no_grad():
          fake, sigmas = self.decoder(noise,self.enc_embed)
          fake = torch.tanh(fake).cpu().detach().numpy()

        return self.transformer.inverse_transform(fake)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)
        self.encoder.to(self._device)
        
    def save(self, path):
        """Save the model in the passed `path`."""
        device_backup = self._device
        self.set_device(torch.device('cpu'))
        torch.save(
            self.save_dict(), 
            path
        )
        self.set_device(device_backup)
    

        

    @classmethod
    def load(cls, path):
        """Load the model stored in the passed `path`."""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = cls.load_dict(
            torch.load(path)
        )
        model.set_device(device)
        return model 
    
        
    def save_dict(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "embedding_dim": self.embedding_dim,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "l2scale": self.l2scale,
            "batch_size": self.batch_size,
            "loss_factor": self.loss_factor,
            "epochs": self.epochs,
            "device": self._device,
            "transformer_dict": self.transformer.__dict__,
            "encoder_dict": self.encoder.state_dict(),
            "decoder_dict": self.decoder.state_dict(),
            "mean": self.mean,
            "std": self.std,
            "logvar": self.logvar,
            "enc_embed": self.enc_embed,
        }
    
    @classmethod
    def load_dict(cls, state: dict):
        instance = cls()
        
        # Saving trivial state
        instance.latent_dim = state["latent_dim"]
        instance.embedding_dim = state["embedding_dim"]
        instance.nhead = state["nhead"]
        instance.dim_feedforward = state["dim_feedforward"]
        instance.dropout = state["dropout"]
        instance.l2scale = state["l2scale"]
        instance.batch_size = state["batch_size"]
        instance.loss_factor = state["loss_factor"]
        instance.epochs = state["epochs"]
        instance._device = state["device"]
        instance.mean = state["mean"]
        instance.std = state["std"]
        instance.logvar = state["logvar"]
        instance.enc_embed = state["enc_embed"]    
        
        # Loading the transformer
        transformer = GMMDATATransformer()
        transformer.__dict__ = state["transformer_dict"]
        instance.transformer  = transformer
        
        data_dim = instance.transformer.output_dimensions
        
        # Loading the encoder decoder architecture
        instance.encoder = Encoder_T(data_dim, instance.latent_dim, instance.embedding_dim, instance.nhead, instance.dim_feedforward, instance.dropout).to(instance._device)
        instance.encoder.load_state_dict(state["encoder_dict"])
        instance.decoder = Decoder_T(data_dim, instance.latent_dim, instance.embedding_dim, instance.nhead, instance.dim_feedforward, instance.dropout).to(instance._device)
        instance.decoder.load_state_dict(state["decoder_dict"])
        
        return instance
        

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def _loss_function_MMD(recon_x, x, sigmas, mean, std, output_info, factor,kernel_choice='rbf'):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            # if span_info.activation_fn != 'softmax':
            if span_info[1] != 'softmax':
                # ed = st + span_info.dim
                ed = st + span_info[0]
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                # ed = st + span_info.dim
                ed = st + span_info[0]
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]

    eps = torch.randn_like(std)
    z = eps * std + mean

    N = z.shape[0]

    z_prior = torch.randn_like(z)#.to(device)

    if kernel_choice == "rbf":
        k_z = rbf_kernel(z, z)
        k_z_prior = rbf_kernel(z_prior, z_prior)
        k_cross = rbf_kernel(z, z_prior)

    else:
        k_z = imq_kernel(z, z)
        k_z_prior = imq_kernel(z_prior, z_prior)
        k_cross = imq_kernel(z, z_prior)

    mmd_z = (k_z - k_z.diag().diag()).sum() / ((N - 1) * N)
    mmd_z_prior = (k_z_prior - k_z_prior.diag().diag()).sum() / ((N - 1) * N)
    mmd_cross = k_cross.sum() / (N ** 2)

    mmd_loss = mmd_z + mmd_z_prior - 2 * mmd_cross

    return sum(loss) * factor / x.size()[0] + mmd_loss

def imq_kernel(z1, z2):
    """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""
    kernel_bandwidth = 1.0
    scales = 1.0
    latent_dim=z1.shape[1]
    Cbase = (2.0 * latent_dim * kernel_bandwidth ** 2)

    k = 0

    for scale in scales:
        C = scale * Cbase
        k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)

    return k

def rbf_kernel(z1, z2):
    """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""
    kernel_bandwidth = 1.0
    latent_dim=z1.shape[1]
    C = 2.0 * latent_dim * kernel_bandwidth ** 2

    k = torch.exp(-torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C)

    return k


def z_gen(embeddings,n_to_sample,metric='minkowski',interpolation_method='SMOTE'):
    # fitting the model
    n_neighbors = 5 +1
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1,metric= metric)
    nn.fit(embeddings)
    dist, ind = nn.kneighbors(embeddings)

    # generating samples
    base_indices = np.random.choice(list(range(len(embeddings))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neighbors)),n_to_sample)

    embeddings_base = embeddings[base_indices]

    if interpolation_method =='SMOTE': ## randomly generate synthetic latent point between 2 real latent points
      embeddings_neighbor = embeddings[ind[base_indices, neighbor_indices]]
      deviations = np.multiply(np.random.rand(n_to_sample,1), embeddings_neighbor - embeddings_base)

      embeddings_samples = embeddings_base + deviations

    elif interpolation_method == 'rectangle':
      embeddings_neighbor = embeddings[ind[base_indices, neighbor_indices]]
      embeddings_samples = np.random.uniform()*embeddings_neighbor + (1-np.random.uniform())*embeddings_base

    elif interpolation_method == 'triangle': ## permutation all latent points in one neighborhood based on their inverse distance as weight
      deviations=0
      for i in range(1,n_neighbors):
        embeddings_neighbor = embeddings[ind[base_indices, i]]
        weight = (n_neighbors -i)/(n_neighbors*(n_neighbors-1)/2)
        deviation = np.multiply(np.random.rand(n_to_sample,1), embeddings_neighbor - embeddings_base)
        deviations += weight*deviation
        embeddings_samples = embeddings_base + deviations

    else:
      mean = torch.zeros(n_to_sample, embeddings.shape[1])
      std = mean + 1
      embeddings_samples = reparameterize(mean, std)

    return embeddings_samples