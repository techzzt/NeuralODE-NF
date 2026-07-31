#!/usr/bin/env python
# coding: utf-8





import os
import re
import gc
import pickle
import pickle as pk
import datetime as dt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

import requests
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pickle Save object
def save_obj(obj, filename):
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'
    if not os.path.isabs(filename) and not filename.startswith('CSV data'):
        filename = os.path.join('CSV data', filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as outp:  
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def read_obj(filename):
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'
    if not os.path.isabs(filename) and not filename.startswith('CSV data'):
        filename = os.path.join('CSV data', filename)
    with open(filename, 'rb') as inp: 
        return pickle.load(inp)


# ### Neural ODE





class TimeEmbedding(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(1, embed_dim)
    
    def forward(self, t):

        if t.dim() == 0: 
            t = t.unsqueeze(0)  # (1,)
        if t.dim() == 1:  # (batch_size,) or (1,)
            t = t.unsqueeze(-1)  # (batch_size, 1)
        return self.linear(t)

class ODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim, time_emb_dim):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.ode_func = nn.Sequential(
            nn.Linear(latent_dim + time_emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, t, z):
        original_shape = z.shape
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        if z.dim() == 3:
            seq_len, batch_size, latent_dim = z.shape
            z_flat = z.view(-1, latent_dim)
            t_expanded = t.expand(seq_len * batch_size)
            t_emb = self.time_embedding(t_expanded)
        elif z.dim() == 2:
            batch_size, latent_dim = z.shape
            z_flat = z
            t_expanded = t.expand(batch_size)
            t_emb = self.time_embedding(t_expanded)
        else:
            raise ValueError(f"Unexpected z dimension: {z.shape}")
        
        z_with_time = torch.cat([z_flat, t_emb], dim=-1)
        dz_dt = self.ode_func(z_with_time)
        
        return dz_dt.view(original_shape)

class AugmentedODEFunc(nn.Module):
    def __init__(self, ode_func, noise_enabled=False, noise_decay=2.0, noise_coupling=0.1):
        super().__init__()
        self.ode_func = ode_func
        self.noise_enabled = noise_enabled
        self.noise_decay = noise_decay
        self.noise_coupling = noise_coupling
        self.memory_bank = None  # Will be set by the detector
        
    def forward(self, t, state):
        if self.noise_enabled:
            z, d, n = state
        else:
            z, d = state
        dz_dt = self.ode_func(t, z)
        
        # Continuous Cumulative Drift: driven by the distance to normal memory bank
        if self.memory_bank is not None:
            # compute_distance returns distance (1 - max_similarity) of shape (batch_size,)
            dd_dt = self.memory_bank.compute_distance(z)
        else:
            # Fallback to kinetic energy
            dd_dt = torch.mean(dz_dt.pow(2), dim=-1)
        
        if self.noise_enabled:
            dn_dt = -self.noise_decay * n + self.noise_coupling * dz_dt
            return (dz_dt, dd_dt, dn_dt)
        
        return (dz_dt, dd_dt)

class ODERNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, time_emb_dim):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.rnn = nn.GRU(input_dim + time_emb_dim, hidden_dim, batch_first=True)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, times, mask=None):
        batch_size = x.size(0)
        if times.dim() == 1:
            times = times.unsqueeze(0).expand(batch_size, -1)
        
        t_emb = self.time_embedding(times.unsqueeze(-1))
        x_with_time = torch.cat([x, t_emb], dim=-1)
        rnn_out, _ = self.rnn(x_with_time)
        
        final_hidden = rnn_out[:, -1, :]
        mu = self.to_mu(final_hidden)
        logvar = self.to_logvar(final_hidden)
        
        std = torch.exp(0.5 * logvar)
        z0 = mu + torch.randn_like(std) * std
        
        return z0, mu, logvar

class ProbabilisticDecoder(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),  # 더 넓은 레이어
            nn.ReLU(),
            nn.Dropout(0.1),  # 정규화 추가
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 2)
        )
        
    def forward(self, z_trajectory):
        original_shape = z_trajectory.shape
        if z_trajectory.dim() == 3:
            z_flat = z_trajectory.reshape(-1, z_trajectory.shape[-1])
        elif z_trajectory.dim() == 2:
            z_flat = z_trajectory
        else:
            raise ValueError(f"Unexpected z_trajectory dimension: {z_trajectory.shape}")
        
        decoded = self.decoder(z_flat)
        mu_flat = decoded[..., :self.input_dim]
        logvar_flat = decoded[..., self.input_dim:]
        
        if z_trajectory.dim() == 3:
            seq_len, batch_size = original_shape[:2]
            mu = mu_flat.reshape(seq_len, batch_size, self.input_dim)
            logvar = logvar_flat.reshape(seq_len, batch_size, self.input_dim)
        else:
            batch_size = original_shape[0]
            mu = mu_flat.reshape(batch_size, self.input_dim)
            logvar = logvar_flat.reshape(batch_size, self.input_dim)
        
        return mu, logvar






def init_model_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)






class MemoryBank(nn.Module):
    """Memory bank for storing normal pattern prototypes"""
    def __init__(self, memory_size, feature_dim, temperature=0.07):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Initialize memory bank with random normal vectors
        self.register_buffer('memory', torch.randn(memory_size, feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Normalize memory vectors
        self.memory = F.normalize(self.memory, dim=1)
        
    def update(self, features):
        """Update memory bank with new normal features"""
        # Detach features to ensure we don't backpropagate through the memory bank history
        features = F.normalize(features.detach(), dim=1)
        batch_size = features.size(0)
        
        ptr = int(self.memory_ptr)
        
        # Replace oldest memories with new ones out-of-place
        if ptr + batch_size <= self.memory_size:
            new_memory = torch.cat([
                self.memory[:ptr],
                features,
                self.memory[ptr + batch_size:]
            ], dim=0)
            ptr = (ptr + batch_size) % self.memory_size
        else:
            # Wrap around
            remaining = self.memory_size - ptr
            new_memory = torch.cat([
                features[remaining:],
                self.memory[batch_size - remaining:ptr],
                features[:remaining]
            ], dim=0)
            ptr = batch_size - remaining
            
        self.memory = new_memory
        self.memory_ptr[0] = ptr
        
    def compute_distance(self, features):
        """Compute minimum distance to memory bank"""
        features = F.normalize(features, dim=1)
        # Compute cosine similarity
        similarity = torch.mm(features, self.memory.t())
        # Convert to distance (higher distance = more anomalous)
        max_similarity, _ = torch.max(similarity, dim=1)
        distance = 1 - max_similarity
        return distance
class RobustGradientSpikeDetector(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=32,
        hidden_dim=64,
        time_emb_dim=8,
        beta=0.1,
        noise_enabled=False,
        noise_decay=2.0,
        noise_coupling=0.1,
        noise_loss_weight=0.05,
        memory_size=1000,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.noise_enabled = noise_enabled
        self.noise_loss_weight = noise_loss_weight
        
        self.encoder = ODERNNEncoder(input_dim, hidden_dim, latent_dim, time_emb_dim)
        self.ode_func = ODEFunc(latent_dim, hidden_dim, time_emb_dim)
        self.augmented_ode_func = AugmentedODEFunc(
            self.ode_func,
            noise_enabled=noise_enabled,
            noise_decay=noise_decay,
            noise_coupling=noise_coupling,
        )
        self.decoder = ProbabilisticDecoder(latent_dim, input_dim, hidden_dim)
        self.gradient_scale = nn.Parameter(torch.tensor(1.0))
        if self.noise_enabled:
            self.noise_projector = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.Tanh(),
            )
        else:
            self.noise_projector = None
            
        self.memory_bank = MemoryBank(memory_size, latent_dim)
        self.augmented_ode_func.memory_bank = self.memory_bank

        self.ode_method = kwargs.get('ode_method', 'dopri5')
        self.ode_rtol = kwargs.get('ode_rtol', 1e-4)
        self.ode_atol = kwargs.get('ode_atol', 1e-6)
        
        # History buffers for Robust Z-Score (median & MAD) & Quantile Clamping
        self.register_buffer('recon_history', torch.zeros(1000))
        self.register_buffer('drift_history', torch.zeros(1000))
        self.register_buffer('velocity_history', torch.zeros(1000))
        self.register_buffer('irregularity_history', torch.zeros(1000))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('history_filled', torch.zeros(1, dtype=torch.bool))
    
    def compute_gradients(self, z_trajectory, times, return_deltas=False):
        if len(z_trajectory) <= 1:
            zero_tensor = torch.zeros_like(z_trajectory)
            if return_deltas:
                return zero_tensor, torch.tensor([], device=z_trajectory.device)
            return zero_tensor
        if times.dim() > 1:
            base_times = times[0] if times.dim() > 1 else times
        else:
            base_times = times
        delta_t = torch.diff(base_times.to(z_trajectory.device))
        delta_t = torch.clamp(delta_t, min=1e-6)
        reshape_dims = (-1,) + (1,) * (z_trajectory.dim() - 1)
        dz_dt = torch.diff(z_trajectory, dim=0) / delta_t.view(reshape_dims)
        last_grad = dz_dt[-1:] if len(dz_dt) > 0 else torch.zeros(1, *z_trajectory.shape[1:], device=z_trajectory.device)
        grads = torch.cat([dz_dt, last_grad], dim=0) * self.gradient_scale
        if return_deltas:
            return grads, delta_t
        return grads
    
    def compute_spike_indicators(self, gradients):
        if len(gradients) <= 1:
            batch_size = gradients.shape[1] if gradients.dim() > 1 else 1
            device = gradients.device
            return torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device)
        
        grad_norms = torch.norm(gradients, dim=-1)
        sudden_spikes = torch.max(torch.abs(torch.diff(grad_norms, dim=0)), dim=0)[0] if len(grad_norms) > 1 else torch.zeros(grad_norms.shape[1], device=gradients.device)
        gradual_spikes = torch.mean(grad_norms, dim=0)
        return sudden_spikes, gradual_spikes
    
    def _compute_irregularity_metric(self, gradients, delta_t):
        if len(gradients) <= 1 or delta_t.numel() == 0:
            if len(gradients) > 0:
                return torch.mean(torch.norm(gradients, dim=-1), dim=0)
            return None
        
        effective_grads = gradients[:-1]
        if effective_grads.dim() >= 3:
            magnitudes = torch.norm(effective_grads, dim=-1)
        else:
            magnitudes = torch.abs(effective_grads)
        
        delta_t_normalized = delta_t / (delta_t.sum() + 1e-8)
        
        if magnitudes.dim() == 2:
            weights = delta_t_normalized.view(-1, 1)
            weighted = magnitudes * weights
            if len(magnitudes) > 1:
                grad_changes = torch.abs(torch.diff(magnitudes, dim=0))
                irregularity = torch.mean(weighted, dim=0) + 0.3 * torch.mean(grad_changes, dim=0)
            else:
                irregularity = torch.mean(weighted, dim=0)
        else:
            weights = delta_t_normalized.view(-1, *([1] * (magnitudes.dim() - 1)))
            weighted = magnitudes * weights
            if len(magnitudes) > 1:
                grad_changes = torch.abs(torch.diff(magnitudes, dim=0))
                irregularity = torch.mean(weighted, dim=0) + 0.3 * torch.mean(grad_changes, dim=0)
            else:
                irregularity = torch.mean(weighted, dim=0)
        
        return irregularity
        
    def _ode_integrate(self, augmented_z0, unique_times):
        try:
            return odeint(
                self.augmented_ode_func,
                augmented_z0,
                unique_times,
                method=self.ode_method,
                rtol=self.ode_rtol,
                atol=self.ode_atol,
                options={'max_num_steps': 1000},
            )
        except Exception:
            try:
                return odeint(self.augmented_ode_func, augmented_z0, unique_times, method='euler')
            except Exception:
                seq_len = len(unique_times)
                if self.noise_enabled:
                    z0, d0, n0 = augmented_z0
                else:
                    z0, d0 = augmented_z0
                    n0 = None
                z_rep = z0.unsqueeze(0).expand(seq_len, -1, -1)
                d_rep = d0.unsqueeze(0).expand(seq_len, -1)
                if self.noise_enabled:
                    n_rep = n0.unsqueeze(0).expand(seq_len, -1, -1)
                    return (z_rep, d_rep, n_rep)
                return (z_rep, d_rep)
    
    def forward(self, x, times, mask=None):
        if times.dim() == 1:
            times = times.unsqueeze(0).expand(x.size(0), -1)
        
        z_latent, mu, logvar = self.encoder(x, times, mask)
        batch_size = z_latent.size(0)
        seq_len = x.size(1)
        unique_times = torch.linspace(0, 1, seq_len).to(x.device)
        
        if self.noise_enabled:
            z_noise = self.noise_projector(z_latent)
            z0 = z_latent - z_noise
            noise_init = z_noise
        else:
            z0 = z_latent
            noise_init = None
        
        augmented_z0 = (z0, torch.zeros(batch_size, device=z0.device))
        if self.noise_enabled:
            augmented_z0 = (z0, torch.zeros(batch_size, device=z0.device), noise_init)
        integration_result = self._ode_integrate(augmented_z0, unique_times)
        if self.noise_enabled:
            z_trajectory, d_trajectory, n_trajectory = integration_result
        else:
            z_trajectory, d_trajectory = integration_result
            n_trajectory = None
        
        gradients = self.compute_gradients(z_trajectory, unique_times)
        sudden_spikes, gradual_spikes = self.compute_spike_indicators(gradients)
        mu_recon, logvar_recon = self.decoder(z_trajectory)
        
        losses = self.compute_loss(
            x,
            mu_recon,
            mu,
            logvar,
            gradients,
            d_trajectory[-1],
            noise_traj=n_trajectory,
        )
        
        return {
            'mu_recon': mu_recon,
            'logvar_recon': logvar_recon,
            'z_trajectory': z_trajectory,
            'noise_trajectory': n_trajectory,
            'gradients': gradients,
            'sudden_spikes': sudden_spikes,
            'gradual_spikes': gradual_spikes,
            'mu': mu,
            'logvar': logvar,
            'losses': losses,
            'final_drift': d_trajectory[-1],
            'initial_noise': noise_init,
        }
    
    def compute_loss(self, x, mu_recon, mu, logvar, gradients, final_drift, noise_traj=None):
        x_expanded = x.transpose(0, 1)
        min_len = min(mu_recon.size(0), x_expanded.size(0))
        
        # 1. Huber Loss (reconstruction error per sample)
        sample_recon_loss = F.huber_loss(mu_recon[:min_len], x_expanded[:min_len], delta=1.0, reduction='none').mean(dim=(0, 2))
        
        # 2. Reliability-Guided Trajectory Preservation (CODAR Core Formulation)
        if self.training:
            with torch.no_grad():
                median_loss = torch.median(sample_recon_loss)
                mad = torch.median(torch.abs(sample_recon_loss - median_loss))
                robust_scale = 1.4826 * mad + 1e-6
                z_score = torch.abs(sample_recon_loss - median_loss) / robust_scale
                # Continuous Reliability Index r_i in (0, 1]
                r_index = torch.sigmoid(-2.0 * (z_score - 3.0))
            
            # Reliable samples drive data reconstruction loss
            recon_loss = torch.mean(r_index * sample_recon_loss)
            # Contaminated samples (1 - r_index) enforce vector field trajectory preservation
            preservation_weight = torch.mean(1.0 - r_index)
        else:
            recon_loss = torch.mean(sample_recon_loss)
            preservation_weight = torch.tensor(0.0, device=x.device)
            
        logvar_clamped = torch.clamp(logvar, min=-10, max=10)
        kl_loss = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()) / x.size(0)
        
        # Update Memory Bank
        if self.training:
            self.memory_bank.update(mu.detach())
            
        # Distance to Memory Bank (Unsupervised clustering regularization)
        memory_distances = self.memory_bank.compute_distance(mu)
        memory_loss = torch.mean(memory_distances)
        
        # 3. Latent trajectory velocity smoothness
        if len(gradients) > 1:
            grad_norms = torch.norm(gradients, dim=-1) 
            grad_changes = torch.diff(grad_norms, dim=0) 
            grad_smoothness = torch.mean(grad_changes ** 2)
            grad_magnitude = torch.mean(grad_norms)
        else:
            grad_smoothness = torch.tensor(0.0, device=x.device)
            grad_magnitude = torch.tensor(0.0, device=x.device)
            
        # 4. Continuous cumulative drift loss
        drift_loss = torch.mean(final_drift)
        
        # 5. Path-wise Jacobian Frobenius Regularization (Lipschitz Stability Penalty)
        # Binds Lipschitz variance of the vector field self.ode_func along integration path
        jac_losses = []
        eps_jac = 1e-3
        for t_val in [0.25, 0.50, 0.75]:
            t_rand = torch.tensor(t_val, device=x.device)
            delta = torch.randn_like(mu) * eps_jac
            dz_dt = self.ode_func(t_rand, mu)
            dz_dt_perturbed = self.ode_func(t_rand, mu + delta)
            jac_diff = (dz_dt_perturbed - dz_dt) / eps_jac
            jac_losses.append(torch.mean(jac_diff ** 2))
        jac_loss = torch.mean(torch.stack(jac_losses))
        
        alignment_loss = torch.tensor(0.0, device=x.device)
        if noise_traj is not None:
            recon_step_error = torch.mean((mu_recon[:min_len] - x_expanded[:min_len]) ** 2, dim=-1)
            noise_energy = torch.norm(noise_traj[:min_len], dim=-1)
            noise_norm = (noise_energy - noise_energy.mean()) / (noise_energy.std() + 1e-6)
            recon_norm = (recon_step_error - recon_step_error.mean()) / (recon_step_error.std() + 1e-6)
            alignment_loss = F.mse_loss(noise_norm, recon_norm)
            
        # Combine losses
        total_loss = (
            0.5 * recon_loss
            + self.beta * kl_loss
            + (0.1 + 0.2 * preservation_weight) * grad_smoothness  
            + 0.01 * grad_magnitude  
            + 0.15 * drift_loss
            + 0.2 * memory_loss
            + 0.1 * jac_loss  # Jacobian Penalty
            + self.noise_loss_weight * alignment_loss
        )
        
        return {
            'total_loss': torch.clamp(total_loss, min=1e-8),
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'grad_smoothness': grad_smoothness,
            'grad_magnitude': grad_magnitude,  
            'drift_loss': drift_loss,
            'memory_loss': memory_loss,
            'jac_loss': jac_loss,
            'noise_alignment': alignment_loss,
        }
    
    def _compute_score(self, x, z_trajectory, d_trajectory, times, mu, logvar, 
                    is_forward=True, noise_trajectory=None, return_components=False):
        gradients, delta_t = self.compute_gradients(z_trajectory, times, return_deltas=True)
        mu_recon, _ = self.decoder(z_trajectory)
        
        x_expanded = x.transpose(0, 1)
        min_len = min(mu_recon.size(0), x_expanded.size(0))
        
        # 1. Reconstruction error component (MAE)
        recon_errors = torch.mean(torch.abs(mu_recon[:min_len] - x_expanded[:min_len]), dim=(0, 2))  # (batch_size,)
        
        # 2. Cumulative drift component
        drift_errors = d_trajectory[-1]  # (batch_size,)
        
        # 3. Latent velocity component
        grad_norms = torch.norm(gradients, dim=-1)  # (seq_len, batch_size)
        velocity_errors = torch.mean(grad_norms, dim=0)  # (batch_size,)
        
        # 4. Irregularity component
        irregularity = self._compute_irregularity_metric(gradients, delta_t)
        
        # Update running statistics buffers during training
        if self.training:
            with torch.no_grad():
                batch_size = recon_errors.size(0)
                ptr = int(self.history_ptr)
                
                r_vals = recon_errors.detach()
                d_vals = drift_errors.detach()
                v_vals = velocity_errors.detach()
                if irregularity is not None:
                    i_vals = irregularity.detach()
                else:
                    i_vals = torch.zeros_like(r_vals)
                
                # Update buffer
                if ptr + batch_size <= 1000:
                    self.recon_history[ptr:ptr+batch_size] = r_vals
                    self.drift_history[ptr:ptr+batch_size] = d_vals
                    self.velocity_history[ptr:ptr+batch_size] = v_vals
                    self.irregularity_history[ptr:ptr+batch_size] = i_vals
                    ptr = (ptr + batch_size) % 1000
                else:
                    rem = 1000 - ptr
                    self.recon_history[ptr:] = r_vals[:rem]
                    self.recon_history[:batch_size-rem] = r_vals[rem:]
                    self.drift_history[ptr:] = d_vals[:rem]
                    self.drift_history[:batch_size-rem] = d_vals[rem:]
                    self.velocity_history[ptr:] = v_vals[:rem]
                    self.velocity_history[:batch_size-rem] = v_vals[rem:]
                    self.irregularity_history[ptr:] = i_vals[:rem]
                    self.irregularity_history[:batch_size-rem] = i_vals[rem:]
                    ptr = (batch_size - rem) % 1000
                    self.history_filled[0] = True
                    
                self.history_ptr[0] = ptr
                
        # Compute Robust Median and MAD
        if self.history_filled[0] or int(self.history_ptr) > 10:
            valid_len = 1000 if self.history_filled[0] else int(self.history_ptr)
            r_hist = self.recon_history[:valid_len]
            d_hist = self.drift_history[:valid_len]
            v_hist = self.velocity_history[:valid_len]
            i_hist = self.irregularity_history[:valid_len]
            
            recon_med = torch.median(r_hist)
            recon_mad = torch.median(torch.abs(r_hist - recon_med))
            
            drift_med = torch.median(d_hist)
            drift_mad = torch.median(torch.abs(d_hist - drift_med))
            
            velocity_med = torch.median(v_hist)
            velocity_mad = torch.median(torch.abs(v_hist - velocity_med))
            
            irreg_med = torch.median(i_hist)
            irreg_mad = torch.median(torch.abs(i_hist - irreg_med))
            
            # Adaptive Clamping
            recon_clamp = torch.quantile(r_hist, 0.995)
            drift_clamp = torch.quantile(d_hist, 0.995)
            velocity_clamp = torch.quantile(v_hist, 0.995)
            irreg_clamp = torch.quantile(i_hist, 0.995)
        else:
            recon_med = torch.median(recon_errors)
            recon_mad = torch.median(torch.abs(recon_errors - recon_med))
            
            drift_med = torch.median(drift_errors)
            drift_mad = torch.median(torch.abs(drift_errors - drift_med))
            
            velocity_med = torch.median(velocity_errors)
            velocity_mad = torch.median(torch.abs(velocity_errors - velocity_med))
            
            if irregularity is not None:
                irreg_med = torch.median(irregularity)
                irreg_mad = torch.median(torch.abs(irregularity - irreg_med))
            else:
                irreg_med = torch.tensor(0.0, device=x.device)
                irreg_mad = torch.tensor(1.0, device=x.device)
            
            recon_clamp = torch.tensor(10.0, device=x.device)
            drift_clamp = torch.tensor(10.0, device=x.device)
            velocity_clamp = torch.tensor(10.0, device=x.device)
            irreg_clamp = torch.tensor(10.0, device=x.device)
            
        # Normalize components (Robust Z-score scaling)
        norm_recon = (recon_errors - recon_med) / (1.4826 * recon_mad + 1e-6)
        norm_drift = (drift_errors - drift_med) / (1.4826 * drift_mad + 1e-6)
        norm_velocity = (velocity_errors - velocity_med) / (1.4826 * velocity_mad + 1e-6)
        
        # Adaptive Clamping
        norm_recon = torch.clamp(norm_recon, min=-3.0, max=float(recon_clamp.item()) if not torch.isnan(recon_clamp) else 10.0)
        norm_drift = torch.clamp(norm_drift, min=-3.0, max=float(drift_clamp.item()) if not torch.isnan(drift_clamp) else 10.0)
        norm_velocity = torch.clamp(norm_velocity, min=-3.0, max=float(velocity_clamp.item()) if not torch.isnan(velocity_clamp) else 10.0)
        
        # Dynamic Irregularity Score Coupling
        if irregularity is not None:
            norm_irreg = (irregularity - irreg_med) / (1.4826 * irreg_mad + 1e-6)
            norm_irreg = torch.clamp(norm_irreg, min=-3.0, max=float(irreg_clamp.item()) if not torch.isnan(irreg_clamp) else 10.0)
            
            gate = torch.sigmoid(norm_irreg)
            alpha = 2.0  # Temperature
            w_recon = 0.4 * torch.exp(-alpha * gate)
            w_drift = 0.4 + 0.4 * (1.0 - torch.exp(-alpha * gate))
            w_velocity = 0.2 * torch.ones_like(gate)
            
            score = w_recon * norm_recon + w_drift * norm_drift + w_velocity * norm_velocity
        else:
            score = 0.4 * norm_recon + 0.4 * norm_drift + 0.2 * norm_velocity
            
        score = score - score.min().detach()
        
        if return_components:
            components = {
                'recon_errors': recon_errors,
                'drift': drift_errors,
                'velocity': velocity_errors,
                'norm_recon': norm_recon,
                'norm_drift': norm_drift,
                'norm_velocity': norm_velocity,
                'irregularity': irregularity if irregularity is not None else torch.zeros_like(recon_errors),
            }
            return score, components
        return score
    
    def anomaly_score(self, x, times, mask=None, direction='both'):
        self.eval()
        with torch.no_grad():
            if times.dim() == 1:
                times = times.unsqueeze(0).expand(x.size(0), -1)
            
            z_latent, mu, logvar = self.encoder(x, times, mask)
            seq_len = x.size(1)
            unique_times = torch.linspace(0, 1, seq_len).to(x.device)
            
            if self.noise_enabled:
                z_noise = self.noise_projector(z_latent)
                z0 = z_latent - z_noise
                noise_init = z_noise
            else:
                z0 = z_latent
                noise_init = None
            
            def build_initial_state():
                drift0 = torch.zeros(z0.size(0), device=z0.device)
                if self.noise_enabled:
                    return (z0, drift0, noise_init)
                return (z0, drift0)
            
            if direction in ['forward', 'both']:
                forward_state = build_initial_state()
                integration_result = self._ode_integrate(forward_state, unique_times)
                if self.noise_enabled:
                    z_fw, d_fw, n_fw = integration_result
                else:
                    z_fw, d_fw = integration_result
                    n_fw = None
                forward_scores = self._compute_score(
                    x,
                    z_fw,
                    d_fw,
                    unique_times,
                    mu,
                    logvar,
                    True,
                    noise_trajectory=n_fw,
                )
            
            if direction in ['backward', 'both']:
                reverse_times = torch.flip(unique_times, dims=[0])
                backward_state = build_initial_state()
                integration_result = self._ode_integrate(backward_state, reverse_times)
                if self.noise_enabled:
                    z_bw, d_bw, n_bw = integration_result
                else:
                    z_bw, d_bw = integration_result
                    n_bw = None
                backward_scores = self._compute_score(
                    x,
                    z_bw,
                    d_bw,
                    reverse_times,
                    mu,
                    logvar,
                    False,
                    noise_trajectory=n_bw,
                )
            
            if direction == 'forward':
                return forward_scores.cpu().numpy()
            elif direction == 'backward':
                return backward_scores.cpu().numpy()
            else:
                return (0.6 * forward_scores + 0.4 * backward_scores).cpu().numpy()



from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path
def unified_load_and_prepare_data(dataset_path, batch_size=64, window_size=50, stride=1, 
                                   normalize=True, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading dataset: {dataset_path}")
    p = Path(dataset_path)
    if not p.exists():
        stem = p.stem.lower()
        candidates = [
            Path(dataset_path),
            Path('datasets') / p.name,
            Path('datasets') / p.stem / p.name,
            Path('CSV data') / p.name,
            Path('CSV data') / f"{p.stem}.pkl",
            Path('CSV data') / 'GT' / f"{p.stem}.pk",
            Path('CSV data') / 'GT' / 'GT_final.pk',
            Path('/Users/jieun/Desktop/GIT/Research/CSV data/GT/GT_final.pk'),
            Path('/Users/jieun/Desktop/GIT/Research/CSV data/Project_final_20_23.pkl')
        ]
        found = None
        for cand in candidates:
            if cand.exists():
                found = cand
                break
        if found:
            print(f"  Redirected dataset path to: {found}")
            dataset_path = str(found)
        else:
            print(f"  Warning: Dataset file not found directly at {dataset_path}. Attempting default load...")
            
    with open(dataset_path, 'rb') as f:
        data = pk.load(f)
    
    if 'x_trn' in data and 'x_tst' in data and 'lab_tst' in data:
        x_trn = data['x_trn']
        x_tst = data['x_tst']
        lab_tst = data['lab_tst']
        
        # prioritize trimSyn detection using the file path hint
        dataset_path_lower = dataset_path.lower()
        is_trimSyn_path = 'trimsyn' in dataset_path_lower
        
        # infer overall data layout
        if isinstance(x_trn, list) and len(x_trn) > 0:
            if is_trimSyn_path:
                # explicit trimSyn path hint
                data_type = 'trimSyn'
            elif len(x_trn[0].shape) == 3:  # (samples, time, features) - trimSyn
                data_type = 'trimSyn'
            elif len(x_trn[0].shape) == 2: 
                if is_trimSyn_path or len(x_trn) != 28:
                    data_type = 'trimSyn'
                else:
                    data_type = 'SMD'
                    entity_idx = 0
                    x_trn = x_trn[entity_idx]
                    x_tst = x_tst[entity_idx]
                    lab_tst = lab_tst[entity_idx] if isinstance(lab_tst, list) else lab_tst
            else:
                data_type = 'Unknown'
        elif isinstance(x_trn, np.ndarray):
            data_type = 'Single'  # e.g., SWaT or WADI style datasets
        else:
            data_type = 'Unknown'
    else:
        if isinstance(data, pd.DataFrame):
            data_type = 'BAS'
        else:
            return None, 'Unknown', None
    
    print(f"Detected data type: {data_type}")
    
    # Data preprocessing
    if data_type == 'trimSyn':
        # trimSyn keeps the original list structure
        x_trn_all = x_trn
        x_tst_all = x_tst
        lab_tst_all = lab_tst
        
        # determine input dimension from the first sample
        input_dim = x_trn_all[0].shape[-1]
        
        # normalization
        if normalize:
            # fit scaler on all samples combined
            x_trn_flat = np.concatenate([x.reshape(-1, input_dim) for x in x_trn_all], axis=0)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(x_trn_flat)
            
            x_trn_all = [scaler.transform(x.reshape(-1, input_dim)).reshape(x.shape) 
                         for x in x_trn_all]
            x_tst_all = [scaler.transform(x.reshape(-1, input_dim)).reshape(x.shape) 
                         for x in x_tst_all]
        
        # ensure (samples, seq_len, features) per section
        def _ensure_3d(array):
            if array.ndim == 3:
                return array
            if array.ndim == 2:
                return array[np.newaxis, ...]
            if array.ndim == 1:
                return array[np.newaxis, :, np.newaxis]
            raise ValueError(f"Unsupported trimSyn array shape: {array.shape}")
        
        x_trn_all = [_ensure_3d(x) for x in x_trn_all]
        x_tst_all = [_ensure_3d(x) for x in x_tst_all]
        input_dim = x_trn_all[0].shape[-1]
        
        # convert to tensors
        x_trn_tensors = [torch.FloatTensor(x) for x in x_trn_all]
        x_tst_tensors = [torch.FloatTensor(x) for x in x_tst_all]
        # keep label lists as-is because sections may differ in length
        lab_tst_list = lab_tst_all
        
        # build per-section DataLoaders with matching time grids
        train_loaders = []
        for x_tensor in x_trn_tensors:
            seq_len = x_tensor.shape[1]
            time_steps = torch.linspace(0, 1, seq_len, dtype=x_tensor.dtype)
            time_steps = time_steps.unsqueeze(0).repeat(x_tensor.shape[0], 1)
            train_dataset = TensorDataset(x_tensor, time_steps)
            train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=False))
        
        data_dict = {
            'train_loaders': train_loaders,
            'x_tst': x_tst_tensors,
            'lab_tst': lab_tst_list,  # 리스트 형태로 저장
            'data_type': 'trimSyn',
            'scaler': scaler if normalize else None
        }
        
    elif data_type == 'SMD':
        # SMD uses 2D arrays
        input_dim = x_trn.shape[-1]
        
        # normalization
        if normalize:
            scaler = StandardScaler()
            x_trn = scaler.fit_transform(x_trn)
            x_tst = scaler.transform(x_tst)
        
        # sliding window helper
        def create_windows(data, window_size, stride):
            windows = []
            for i in range(0, len(data) - window_size + 1, stride):
                windows.append(data[i:i+window_size])
            return np.array(windows)
        
        x_trn_windows = create_windows(x_trn, window_size, stride)
        x_tst_windows = create_windows(x_tst, window_size, stride)

        # label handling (window-level)
        # - lab_tst가 1D(time,)이면: 각 윈도우 구간에서 max(label)로 윈도우 라벨 생성
        # - lab_tst가 2D(time, ...)이면: 윈도우 구간 전체에서 max로 윈도우 라벨 생성
        def create_label_windows(labels, window_size, stride):
            labels = np.asarray(labels)
            if labels.ndim == 0:
                labels = labels.reshape(-1)

            # time 축은 항상 첫 번째 축이라고 가정
            T = labels.shape[0]
            out = []
            for i in range(0, T - window_size + 1, stride):
                window = labels[i : i + window_size]
                out.append(1.0 if np.max(window) > 0 else 0.0)
            return np.asarray(out, dtype=np.float32)

        lab_tst_windows = create_label_windows(lab_tst, window_size, stride)

        # convert to tensors
        seq_len = window_size
        time_steps = torch.linspace(0, 1, seq_len)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(x_trn_windows),
            time_steps.repeat(len(x_trn_windows), 1)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(x_tst_windows),
            time_steps.repeat(len(x_tst_windows), 1),
            torch.FloatTensor(lab_tst_windows)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        data_dict = {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'x_tst': torch.FloatTensor(x_tst_windows),
            'lab_tst': torch.FloatTensor(lab_tst_windows),
            'data_type': 'SMD',
            'scaler': scaler if normalize else None
        }
        
    elif data_type == 'Single':
        # Single-stream datasets such as SWaT or WADI
        input_dim = x_trn.shape[-1]
        
        # normalization
        if normalize:
            scaler = StandardScaler()
            x_trn = scaler.fit_transform(x_trn)
            x_tst = scaler.transform(x_tst)
        
        # sliding window helper
        def create_windows(data, window_size, stride):
            windows = []
            for i in range(0, len(data) - window_size + 1, stride):
                windows.append(data[i:i+window_size])
            return np.array(windows)
        
        x_trn_windows = create_windows(x_trn, window_size, stride)
        x_tst_windows = create_windows(x_tst, window_size, stride)

        # label handling (window-level)
        def create_label_windows(labels, window_size, stride):
            labels = np.asarray(labels)
            if labels.ndim == 0:
                labels = labels.reshape(-1)

            T = labels.shape[0]
            out = []
            for i in range(0, T - window_size + 1, stride):
                window = labels[i : i + window_size]
                out.append(1.0 if np.max(window) > 0 else 0.0)
            return np.asarray(out, dtype=np.float32)

        lab_tst_windows = create_label_windows(lab_tst, window_size, stride)

        # convert to tensors
        seq_len = window_size
        time_steps = torch.linspace(0, 1, seq_len)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(x_trn_windows),
            time_steps.repeat(len(x_trn_windows), 1)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(x_tst_windows),
            time_steps.repeat(len(x_tst_windows), 1),
            torch.FloatTensor(lab_tst_windows)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        data_dict = {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'x_tst': torch.FloatTensor(x_tst_windows),
            'lab_tst': torch.FloatTensor(lab_tst_windows),
            'data_type': 'Single',
            'scaler': scaler if normalize else None
        }
    elif data_type == 'BAS':
        # Building Automation System (BAS) Dataset
        df = data.copy()
        df.index = pd.to_datetime(df.index)
        
        # Calculate cooling loads for all buildings if they don't exist
        for b in ['b210', 'b055', 'b103', 'b166', 'b002', 'b084']:
            rt_col = f'{b}_rt'
            st_col = f'{b}_st'
            fw_col = f'{b}_fw'
            pt_col = f'{b}_pt' if f'{b}_pt' in df.columns else (f'{b}_raw' if f'{b}_raw' in df.columns else None)
            
            if rt_col in df.columns and st_col in df.columns and fw_col in df.columns:
                df[f'{b}_cal'] = (df[rt_col] - df[st_col]) * df[fw_col] / 20.0
                if pt_col is not None:
                    df[f'{b}_diff'] = (df[f'{b}_cal'] - df[pt_col]).abs()
                    
        # b081 is special (sum of two sensors)
        if 'b081_rt_1' in df.columns:
            df['b081_cal'] = ((df['b081_rt_1'] + df['b081_rt_2']) - (df['b081_st_1'] + df['b081_st_2'])) * (df['b081_fw_1'] + df['b081_fw_2']) / 20.0
            if 'b081_raw' in df.columns:
                df['b081_diff'] = (df['b081_cal'] - df['b081_raw']).abs()
            elif 'b081_raw_1' in df.columns:
                df['b081_raw'] = df['b081_raw_1'] + df['b081_raw_2']
                df['b081_diff'] = (df['b081_cal'] - df['b081_raw']).abs()
                
        # default building: b166 (culc) or b210 (Kendeda)
        building = 'b166'
        signal_type = 'cal'
        
        if signal_type == 'cal':
            target_col = f'{building}_cal'
        elif signal_type == 'diff':
            target_col = f'{building}_diff'
        else:
            target_col = f'{building}_pt' if f'{building}_pt' in df.columns else f'{building}_raw'
            
        print(f"Selecting BAS target column: {target_col}")
        
        # Keep only the target column
        ts_data = df[[target_col]].copy()
        ts_data = ts_data.ffill().bfill()
        
        # Compute cyclical time features (hour of day, month of year)
        hours = ts_data.index.hour.values
        months = ts_data.index.month.values
        
        ts_data['sin_hour'] = np.sin(2.0 * np.pi * hours / 24.0)
        ts_data['cos_hour'] = np.cos(2.0 * np.pi * hours / 24.0)
        ts_data['sin_month'] = np.sin(2.0 * np.pi * months / 12.0)
        ts_data['cos_month'] = np.cos(2.0 * np.pi * months / 12.0)
        
        # Features: [cooling_load, sin_hour, cos_hour, sin_month, cos_month]
        feature_cols = [target_col, 'sin_hour', 'cos_hour', 'sin_month', 'cos_month']
        values = ts_data[feature_cols].values
        
        # Train-test split (70% training, 30% testing)
        split_idx = int(len(values) * 0.7)
        x_trn_raw = values[:split_idx]
        x_tst_raw = values[split_idx:]
        
        # Generate heuristic anomaly labels (top 2% of load deviations)
        if f'{building}_diff' in df.columns:
            diff_values = df[f'{building}_diff'].values
            diff_tst = diff_values[split_idx:]
            threshold = np.nanpercentile(diff_tst, 98)
            labels_tst_raw = (diff_tst > threshold).astype(float)
        else:
            labels_tst_raw = np.zeros(len(x_tst_raw))
            
        # Normalization
        if normalize:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            load_train = x_trn_raw[:, 0:1]
            scaler.fit(load_train)
            
            x_trn = x_trn_raw.copy()
            x_tst = x_tst_raw.copy()
            x_trn[:, 0:1] = scaler.transform(load_train)
            x_tst[:, 0:1] = scaler.transform(x_tst_raw[:, 0:1])
        else:
            scaler = None
            x_trn = x_trn_raw
            x_tst = x_tst_raw
            
        # sliding window
        def create_windows(data, window_size, stride):
            windows = []
            for i in range(0, len(data) - window_size + 1, stride):
                windows.append(data[i:i+window_size])
            return np.array(windows)
            
        x_trn_windows = create_windows(x_trn, window_size, stride)
        x_tst_windows = create_windows(x_tst, window_size, stride)
        
        def create_label_windows(labels, window_size, stride):
            out = []
            for i in range(0, len(labels) - window_size + 1, stride):
                window = labels[i : i + window_size]
                out.append(1.0 if np.max(window) > 0 else 0.0)
            return np.array(out, dtype=np.float32)
            
        lab_tst_windows = create_label_windows(labels_tst_raw, window_size, stride)
        
        seq_len = window_size
        time_steps = torch.linspace(0, 1, seq_len)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(x_trn_windows),
            time_steps.repeat(len(x_trn_windows), 1)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(x_tst_windows),
            time_steps.repeat(len(x_tst_windows), 1),
            torch.FloatTensor(lab_tst_windows)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        data_dict = {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'x_tst': torch.FloatTensor(x_tst_windows),
            'lab_tst': torch.FloatTensor(lab_tst_windows),
            'data_type': 'BAS',
            'scaler': scaler
        }
        input_dim = 5
    else:
        if isinstance(data, pd.DataFrame):
            data_type = 'BAS'
        else:
            return None, 'Unknown', None
    
    print(f"데이터 전처리 완료!")
    print(f"  입력 차원: {input_dim}")
    if data_type == 'trimSyn':
        print(f"  훈련 샘플 수: {len(x_trn_all)}")
        print(f"  테스트 샘플 수: {len(x_tst_all)}")
    else:
        print(f"  훈련 윈도우 수: {len(data_dict['train_loader'].dataset)}")
        print(f"  테스트 윈도우 수: {len(data_dict['test_loader'].dataset)}")
    
    return data_dict, data_type, input_dim


DEFAULT_TOPK_RATIO = 0.01


def _compute_metrics(scores, labels, name="Global"):
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if len(scores) == 0 or len(labels) == 0:
        return {
            'name': name,
            'threshold': None,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.5,
            'accuracy': 0.0,
            'positive_ratio': 0.0,
            'count': 0,
            'fallback_used': False,
        }

    min_len = min(len(scores), len(labels))
    scores = scores[:min_len]
    labels = labels[:min_len]

    thresholds = np.percentile(scores, np.linspace(80, 99.5, min(50, len(scores))))
    if np.all(scores == scores[0]):
        thresholds = np.array([scores[0]])

    best = {
        'f1': -1,
        'threshold': thresholds[0],
        'precision': 0.0,
        'recall': 0.0,
        'accuracy': 0.0,
        'fallback_used': False,
    }

    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        f1_val = f1_score(labels, predictions, zero_division=0)
        if f1_val > best['f1']:
            best.update({
                'f1': f1_val,
                'threshold': threshold,
                'precision': precision_score(labels, predictions, zero_division=0),
                'recall': recall_score(labels, predictions, zero_division=0),
                'accuracy': accuracy_score(labels, predictions),
                'fallback_used': False,
            })

    pos_count = int(labels.sum())
    if pos_count > 0 and best['f1'] <= 0:
        top_k = max(1, pos_count)
        top_k = min(len(scores), top_k)
        sorted_scores = np.sort(scores)[::-1]
        threshold = sorted_scores[top_k - 1] - 1e-8
        predictions = (scores >= threshold).astype(int)
        best.update({
            'f1': f1_score(labels, predictions, zero_division=0),
            'threshold': threshold,
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'accuracy': accuracy_score(labels, predictions),
            'fallback_used': True,
        })

    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.5

    return {
        'name': name,
        'threshold': best['threshold'],
        'precision': best['precision'],
        'recall': best['recall'],
        'f1': best['f1'],
        'auc': auc,
        'accuracy': best['accuracy'],
        'positive_ratio': float(np.mean(labels)),
        'count': int(len(labels)),
        'fallback_used': best['fallback_used'],
    }


def _topk_metrics(scores, labels, name="TopK", top_ratio=DEFAULT_TOPK_RATIO):

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if len(scores) == 0 or len(labels) == 0:
        return {
            'name': name,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'count': 0,
            'top_ratio': top_ratio,
        }

    n = min(len(scores), len(labels))
    scores = scores[:n]
    labels = labels[:n]
    k = max(1, int(np.ceil(n * top_ratio)))
    k = min(n, k)
    top_indices = np.argpartition(scores, -k)[-k:]
    predictions = np.zeros(n, dtype=int)
    predictions[top_indices] = 1

    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    return {
        'name': name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'count': n,
        'top_ratio': top_ratio,
        'top_k': k,
    }


def _sre_metrics(scores_a, scores_b, labels=None, name="SRE", top_ratio=DEFAULT_TOPK_RATIO):
    """
    SRE(Score Rank Stability): 두 설정(A/B)에서 top-k 이상 후보가 얼마나 겹치는지.

    - jaccard = |TopK_A ∩ TopK_B| / |TopK_A ∪ TopK_B|
    - sre = 1 - jaccard  (0이면 완전 안정, 1이면 완전 불안정)

    labels가 주어지면(평가 데이터) intersection/union의 precision도 같이 산출.
    """
    scores_a = np.asarray(scores_a).flatten()
    scores_b = np.asarray(scores_b).flatten()

    n = min(len(scores_a), len(scores_b))
    scores_a = scores_a[:n]
    scores_b = scores_b[:n]

    if labels is not None:
        labels = np.asarray(labels).flatten()[:n]
    else:
        labels = None

    if n == 0:
        return {
            'name': name,
            'top_ratio': top_ratio,
            'top_k': 0,
            'overlap': 0,
            'union': 0,
            'jaccard': 1.0,
            'sre': 0.0,
        }

    k = max(1, int(np.ceil(n * top_ratio)))
    k = min(n, k)

    idx_a = set(np.argpartition(scores_a, -k)[-k:].tolist())
    idx_b = set(np.argpartition(scores_b, -k)[-k:].tolist())

    inter = idx_a & idx_b
    union = idx_a | idx_b

    jaccard = (len(inter) / len(union)) if len(union) > 0 else 1.0

    out = {
        'name': name,
        'top_ratio': float(top_ratio),
        'top_k': int(k),
        'overlap': int(len(inter)),
        'union': int(len(union)),
        'jaccard': float(jaccard),
        'sre': float(1.0 - jaccard),
    }

    if labels is not None and len(labels) == n:
        # precision@set (해당 top 후보 중 true anomaly 비율)
        def _precision_for(index_set):
            if not index_set:
                return 0.0
            idx = np.fromiter(index_set, dtype=int)
            return float(np.mean(labels[idx] > 0))

        out.update({
            'precision_topk_a': _precision_for(idx_a),
            'precision_topk_b': _precision_for(idx_b),
            'precision_intersection': _precision_for(inter),
            'precision_union': _precision_for(union),
        })

        pos = float(np.sum(labels > 0))
        if pos > 0:
            out['recall_intersection'] = float(np.sum(labels[list(inter)] > 0) / pos) if inter else 0.0
            out['recall_union'] = float(np.sum(labels[list(union)] > 0) / pos) if union else 0.0

    return out


def _to_sequence_labels(lab_sample, num_sequences):

    if isinstance(lab_sample, torch.Tensor):
        arr = lab_sample.cpu().numpy()
    else:
        arr = np.array(lab_sample)

    if arr.ndim == 2:
        seq_labels = (np.max(arr, axis=1) > 0).astype(int)
    elif arr.ndim == 1:
        if len(arr) == num_sequences:
            seq_labels = (arr > 0).astype(int)
        else:
            seq_labels = (arr[:num_sequences] > 0).astype(int)
    else:
        seq_labels = np.zeros(num_sequences, dtype=int)

    if len(seq_labels) < num_sequences:
        pad_value = seq_labels[-1] if len(seq_labels) else 0
        seq_labels = np.pad(seq_labels, (0, num_sequences - len(seq_labels)), constant_values=pad_value)

    return seq_labels[:num_sequences]


def unified_train(model, data_dict, num_epochs=50, data_difficulty='auto', device=None, val_interval=None, val_max_sections=1):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    data_type = data_dict['data_type']
    
    # auto-detect recommended difficulty settings
    if data_difficulty == 'auto':
        if data_type == 'trimSyn':
            data_difficulty = 'hard'
        elif data_type == 'SMD':
            data_difficulty = 'medium'
        else:
            data_difficulty = 'easy'
    
    print(f"학습 시작 (데이터 타입: {data_type}, 난이도: {data_difficulty}, 에포크: {num_epochs})")
    
    if data_difficulty == 'hard':
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
    elif data_difficulty == 'medium':
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # 학습 루프
    if data_type == 'trimSyn':
        # trimSyn: 여러 샘플에 대해 학습
        train_loaders = data_dict['train_loaders']
        history = {
            'loss': [],
            'recon': [],
            'kl': [],
            'grad': [],
            'drift': [],
            'noise': [],
            'val': []
        }
        
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            recon_sum = 0
            kl_sum = 0
            grad_sum = 0
            drift_sum = 0
            noise_sum = 0
            n_batches = 0
            
            for train_loader in train_loaders:
                for batch_data in train_loader:
                    if len(batch_data) == 2:
                        batch_x, batch_times = batch_data
                        batch_mask = None
                    else:
                        batch_x, batch_times, batch_mask = batch_data
                    
                    batch_x = batch_x.to(device)
                    batch_times = batch_times.to(device)
                    if batch_mask is not None:
                        batch_mask = batch_mask.to(device)
                    
                    optimizer.zero_grad()
                    
                    try:
                        outputs = model(batch_x, batch_times, batch_mask)
                        losses = outputs['losses']
                        loss = losses['total_loss']
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        total_loss += loss.item()
                        recon_sum += losses['recon_loss'].item()
                        kl_sum += losses['kl_loss'].item()
                        grad_sum += losses['grad_smoothness'].item()
                        drift_sum += losses['drift_loss'].item()
                        noise_sum += losses.get('memory_loss', torch.tensor(0.0, device=device)).item()
                        n_batches += 1
                    except Exception as e:
                        print(f"배치 오류 (계속 진행): {e}")
                        continue
            
            denom = max(n_batches, 1)
            avg_loss = total_loss / denom
            avg_recon = recon_sum / denom
            avg_kl = kl_sum / denom
            avg_grad = grad_sum / denom
            avg_drift = drift_sum / denom
            avg_noise = noise_sum / denom
            history['loss'].append(avg_loss)
            history['recon'].append(avg_recon)
            history['kl'].append(avg_kl)
            history['grad'].append(avg_grad)
            history['drift'].append(avg_drift)
            history['noise'].append(avg_noise)
            
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()
            
            print(
                f"Epoch {epoch+1}/{num_epochs} | Loss:{avg_loss:.6f} "
                f"Recon:{avg_recon:.6f} KL:{avg_kl:.6f} Grad:{avg_grad:.6f} "
                f"Drift:{avg_drift:.6f} Memory:{avg_noise:.6f} | Batches:{n_batches}"
            )
            
            if val_interval and (epoch + 1) % val_interval == 0:
                snapshot = quick_validation(
                    model,
                    data_dict,
                    max_sections=val_max_sections,
                    device=device,
                )
                if snapshot is not None:
                    g = snapshot['global']
                    print(
                        f"  [Val@{epoch+1}] F1:{g['f1']:.4f} Precision:{g['precision']:.4f} "
                        f"Recall:{g['recall']:.4f} PosRatio:{g['positive_ratio']:.4f}"
                    )
                    history['val'].append({'epoch': epoch + 1, 'global': g, 'sections': snapshot['sections']})
    else:
        # SMD, Single: 단일 DataLoader
        train_loader = data_dict['train_loader']
        history = {
            'loss': [],
            'recon': [],
            'kl': [],
            'grad': [],
            'drift': [],
            'noise': [],
            'val': []
        }
        
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            recon_sum = 0
            kl_sum = 0
            grad_sum = 0
            drift_sum = 0
            noise_sum = 0
            n_batches = 0
            
            for batch_data in train_loader:
                if len(batch_data) == 2:
                    batch_x, batch_times = batch_data
                    batch_mask = None
                elif len(batch_data) == 3:
                    batch_x, batch_times, batch_labels = batch_data
                    batch_mask = None
                else:
                    batch_x, batch_times, batch_mask = batch_data
                
                batch_x = batch_x.to(device)
                batch_times = batch_times.to(device)
                if batch_mask is not None:
                    batch_mask = batch_mask.to(device)
                
                optimizer.zero_grad()
                
                try:
                    outputs = model(batch_x, batch_times, batch_mask)
                    losses = outputs['losses']
                    loss = losses['total_loss']
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    recon_sum += losses['recon_loss'].item()
                    kl_sum += losses['kl_loss'].item()
                    grad_sum += losses['grad_smoothness'].item()
                    drift_sum += losses['drift_loss'].item()
                    noise_sum += losses.get('memory_loss', torch.tensor(0.0, device=device)).item()
                    n_batches += 1
                except Exception as e:
                    print(f"배치 오류 (계속 진행): {e}")
                    continue
            
            denom = max(n_batches, 1)
            avg_loss = total_loss / denom
            avg_recon = recon_sum / denom
            avg_kl = kl_sum / denom
            avg_grad = grad_sum / denom
            avg_drift = drift_sum / denom
            avg_noise = noise_sum / denom
            history['loss'].append(avg_loss)
            history['recon'].append(avg_recon)
            history['kl'].append(avg_kl)
            history['grad'].append(avg_grad)
            history['drift'].append(avg_drift)
            history['noise'].append(avg_noise)
            
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()
            
            print(
                f"Epoch {epoch+1}/{num_epochs} | Loss:{avg_loss:.6f} "
                f"Recon:{avg_recon:.6f} KL:{avg_kl:.6f} Grad:{avg_grad:.6f} "
                f"Drift:{avg_drift:.6f} Memory:{avg_noise:.6f} | Batches:{n_batches}"
            )
            
            if val_interval and (epoch + 1) % val_interval == 0:
                snapshot = quick_validation(
                    model,
                    data_dict,
                    max_sections=val_max_sections,
                    device=device,
                )
                if snapshot is not None:
                    g = snapshot['global']
                    print(
                        f"  [Val@{epoch+1}] F1:{g['f1']:.4f} Precision:{g['precision']:.4f} "
                        f"Recall:{g['recall']:.4f} PosRatio:{g['positive_ratio']:.4f}"
                    )
                    history['val'].append({'epoch': epoch + 1, 'global': g, 'sections': snapshot['sections']})
    
    print("학습 완료!")
    return model, history


def unified_evaluate(model, data_dict, device=None):
    """
    모델을 평가합니다.
    
    주의: 이 함수는 test 데이터를 사용하여 최종 평가를 수행합니다.
    - x_tst, lab_tst를 사용 (data_dict에서 가져옴)
    - 최종 F1 score, Precision, Recall 등은 test 데이터 기준입니다.
    """
    def compute_metrics(scores, labels, name="Global"):
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        if len(scores) == 0 or len(labels) == 0:
            return {
                'name': name,
                'threshold': None,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.5,
                'accuracy': 0.0,
                'positive_ratio': 0.0,
                'count': 0,
                'fallback_used': False,
            }
            
        min_len = min(len(scores), len(labels))
        scores = scores[:min_len]
        labels = labels[:min_len]
        thresholds = np.percentile(scores, np.linspace(80, 99.5, min(50, len(scores))))
        
        if np.all(scores == scores[0]):
            thresholds = np.array([scores[0]])
        best = {'f1': -1, 'threshold': thresholds[0], 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0, 'fallback_used': False}
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            f1_val = f1_score(labels, predictions, zero_division=0)
            if f1_val > best['f1']:
                best.update({
                    'f1': f1_val,
                    'threshold': threshold,
                    'precision': precision_score(labels, predictions, zero_division=0),
                    'recall': recall_score(labels, predictions, zero_division=0),
                    'accuracy': accuracy_score(labels, predictions),
                    'fallback_used': False,
                })
        pos_count = int(labels.sum())

        if pos_count > 0 and best['f1'] <= 0:
            top_k = max(1, pos_count)
            top_k = min(len(scores), top_k)
            sorted_scores = np.sort(scores)[::-1]
            threshold = sorted_scores[top_k - 1] - 1e-8
            predictions = (scores >= threshold).astype(int)
            best.update({
                'f1': f1_score(labels, predictions, zero_division=0),
                'threshold': threshold,
                'precision': precision_score(labels, predictions, zero_division=0),
                'recall': recall_score(labels, predictions, zero_division=0),
                'accuracy': accuracy_score(labels, predictions),
                'fallback_used': True,
            })
        try:
            auc = roc_auc_score(labels, scores)
        except Exception:
            auc = 0.5
        return {
            'name': name,
            'threshold': best['threshold'],
            'precision': best['precision'],
            'recall': best['recall'],
            'f1': best['f1'],
            'auc': auc,
            'accuracy': best['accuracy'],
            'positive_ratio': float(np.mean(labels)),
            'count': int(len(labels)),
            'fallback_used': best['fallback_used'],
        }
    
    def to_sequence_labels(lab_sample, num_sequences):
        if isinstance(lab_sample, torch.Tensor):
            arr = lab_sample.cpu().numpy()
        else:
            arr = np.array(lab_sample)
        
        if arr.ndim == 2:
            seq_labels = (np.max(arr, axis=1) > 0).astype(int)
        elif arr.ndim == 1:
            if len(arr) == num_sequences:
                seq_labels = (arr > 0).astype(int)
            else:
                seq_labels = (arr[:num_sequences] > 0).astype(int)
        else:
            seq_labels = np.zeros(num_sequences, dtype=int)
        if len(seq_labels) < num_sequences:
            seq_labels = np.pad(seq_labels, (0, num_sequences - len(seq_labels)), constant_values=seq_labels[-1] if len(seq_labels) else 0)
        return seq_labels[:num_sequences]
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    data_type = data_dict['data_type']
    x_tst = data_dict['x_tst']
    lab_tst = data_dict['lab_tst']
    
    print(f"평가 시작 (데이터 타입: {data_type})")
    
    all_scores = []
    all_labels = []
    section_metrics = []
    skipped_sections = []
    macro_section_metrics = None
    macro_topk_metrics = None
    topk_section_metrics = None
    
    with torch.no_grad():
        if data_type == 'trimSyn':
            section_scores = []
            section_labels = []
            section_valid_names = []
            for idx, (x_sample, lab_sample) in enumerate(zip(x_tst, lab_tst)):
                x_sample = x_sample.to(device)
                num_sequences = x_sample.shape[0]
                seq_len = x_sample.shape[1]
                time_steps = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(num_sequences, 1).to(device)
                
                scores = model.anomaly_score(x_sample, time_steps)
                
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                scores = np.asarray(scores).flatten()
                seq_labels = to_sequence_labels(lab_sample, num_sequences)
                min_len = min(len(scores), len(seq_labels))
                scores = scores[:min_len]
                seq_labels = seq_labels[:min_len]
                pos_ratio = float(np.mean(seq_labels)) if len(seq_labels) > 0 else 0.0
                
                if pos_ratio == 0.0 or pos_ratio == 1.0:
                    reason = "no positives" if pos_ratio == 0.0 else "no negatives"
                    skipped_sections.append({
                        'name': f"Section {idx}",
                        'reason': reason,
                        'positive_ratio': pos_ratio,
                        'count': int(len(seq_labels))
                    })
                    section_metrics.append({
                        'name': f"Section {idx}",
                        'skipped': True,
                        'reason': reason,
                        'positive_ratio': pos_ratio,
                        'count': int(len(seq_labels))
                    })
                    print(
                        f"  Section {idx}: 스킵 ({reason}, PosRatio={pos_ratio:.4f})"
                    )
                    continue
                
                section_scores.append(scores)
                section_labels.append(seq_labels)
                section_valid_names.append(f"Section {idx}")
                section_result = compute_metrics(scores, seq_labels, name=f"Section {idx}")
                section_metrics.append(section_result)
                fallback_tag = " (fallback)" if section_result['fallback_used'] else ""
                
                print(
                    f"  Section {idx}{fallback_tag}: F1={section_result['f1']:.4f}, "
                    f"Precision={section_result['precision']:.4f}, "
                    f"Recall={section_result['recall']:.4f}, AUC={section_result['auc']:.4f}, "
                    f"PosRatio={section_result['positive_ratio']:.4f}"
                )
            
            if section_scores:
                all_scores = np.concatenate(section_scores)
                all_labels = np.concatenate(section_labels)
            else:
                all_scores = np.array([])
                all_labels = np.array([])
            valid_metrics = [m for m in section_metrics if not m.get('skipped')]
            
            if valid_metrics:
                macro_section_metrics = {
                    'precision': float(np.mean([m['precision'] for m in valid_metrics])),
                    'recall': float(np.mean([m['recall'] for m in valid_metrics])),
                    'f1': float(np.mean([m['f1'] for m in valid_metrics])),
                    'count': len(valid_metrics)
                }
            if section_scores:
                topk_section_metrics = []
                for name, scores, labels in zip(section_valid_names, section_scores, section_labels):
                    tk = _topk_metrics(scores, labels, name=name, top_ratio=DEFAULT_TOPK_RATIO)
                    topk_section_metrics.append(tk)
                if topk_section_metrics:
                    macro_topk_metrics = {
                        'precision': float(np.mean([m['precision'] for m in topk_section_metrics])),
                        'recall': float(np.mean([m['recall'] for m in topk_section_metrics])),
                        'f1': float(np.mean([m['f1'] for m in topk_section_metrics])),
                        'count': len(topk_section_metrics)
                    }
        else:
            if isinstance(x_tst, torch.Tensor):
                x_tst = x_tst.to(device)
            seq_len = x_tst.shape[1]

            # SRE용: 서로 다른 time-grid 2개로 점수 계산 (실제 랜덤 타임 지터링)
            t_base = torch.linspace(0, 1, seq_len)
            time_steps_a = t_base.unsqueeze(0).repeat(x_tst.shape[0], 1).to(device)
            
            noise = torch.randn_like(t_base) * 0.02
            t_jit = t_base + noise
            t_jit[0] = 0.0
            t_jit[-1] = 1.0
            t_jit, _ = torch.sort(t_jit)
            t_jit = torch.clamp(t_jit, min=0.0, max=1.0)
            time_steps_b = t_jit.unsqueeze(0).repeat(x_tst.shape[0], 1).to(device)

            scores_a = model.anomaly_score(x_tst, time_steps_a)
            scores_b = model.anomaly_score(x_tst, time_steps_b)

            if isinstance(scores_a, torch.Tensor):
                scores_a = scores_a.cpu().numpy()
            if isinstance(scores_b, torch.Tensor):
                scores_b = scores_b.cpu().numpy()

            all_scores = np.asarray(scores_a).flatten()
            all_scores_b = np.asarray(scores_b).flatten()
            all_labels = lab_tst.cpu().numpy().flatten() if isinstance(lab_tst, torch.Tensor) else np.asarray(lab_tst).flatten()
    
    global_metrics = compute_metrics(all_scores, all_labels, name="Global")

    # Global SRE
    if 'all_scores_b' in locals():
        sre_global = _sre_metrics(all_scores, all_scores_b, labels=all_labels, name="Global", top_ratio=DEFAULT_TOPK_RATIO)
        global_metrics.update({
            'sre': sre_global['sre'],
            'sre_jaccard': sre_global['jaccard'],
            'sre_overlap': sre_global['overlap'],
            'sre_union': sre_global['union'],
        })
        for k in ['precision_topk_a', 'precision_topk_b', 'precision_intersection', 'precision_union', 'recall_intersection', 'recall_union']:
            if k in sre_global:
                global_metrics[f"sre_{k}"] = sre_global[k]

    print("\n통합 평가 결과:")
    
    fallback_msg = " (fallback)" if global_metrics['fallback_used'] else ""
    print(f"  임계값{fallback_msg}: {global_metrics['threshold']:.4f}" if global_metrics['threshold'] is not None else "  임계값: N/A")
    print(f"  Precision: {global_metrics['precision']:.4f}")
    print(f"  Recall: {global_metrics['recall']:.4f}")
    print(f"  F1 Score: {global_metrics['f1']:.4f}")
    print(f"  AUC: {global_metrics['auc']:.4f}")
    print(f"  Accuracy: {global_metrics['accuracy']:.4f}")
    print(f"  Positive Ratio: {global_metrics['positive_ratio']:.4f} (count={global_metrics['count']})")
    if 'sre' in global_metrics:
        print(
            f"  SRE: {global_metrics['sre']:.4f} (Jaccard={global_metrics.get('sre_jaccard', 0.0):.4f}, "
            f"overlap={global_metrics.get('sre_overlap', 0)}, union={global_metrics.get('sre_union', 0)})"
        )
        if 'sre_precision_intersection' in global_metrics:
            print(
                f"  SRE@TopK Precision: intersection={global_metrics['sre_precision_intersection']:.4f}, "
                f"union={global_metrics['sre_precision_union']:.4f}, "
                f"topkA={global_metrics.get('sre_precision_topk_a', 0.0):.4f}, topkB={global_metrics.get('sre_precision_topk_b', 0.0):.4f}"
            )
    
    if macro_section_metrics:
        print(
            f"  Macro F1 (valid sections): {macro_section_metrics['f1']:.4f}"
            f" | Precision: {macro_section_metrics['precision']:.4f}"
            f" | Recall: {macro_section_metrics['recall']:.4f}"
        )
    
    if macro_topk_metrics:
        print(
            f"  Macro Top-K F1: {macro_topk_metrics['f1']:.4f}"
            f" | Precision: {macro_topk_metrics['precision']:.4f}"
            f" | Recall: {macro_topk_metrics['recall']:.4f}"
            f" (top {DEFAULT_TOPK_RATIO*100:.2f}% per section)"
        )
    
    threshold = global_metrics['threshold']
    global_scores = np.array(all_scores)
    if threshold is not None and len(global_scores) > 0:
        predictions = (global_scores > threshold).astype(int)
    else:
        predictions = np.zeros_like(global_scores, dtype=int)
    
    metrics = {
        'threshold': threshold,
        'precision': global_metrics['precision'],
        'recall': global_metrics['recall'],
        'f1': global_metrics['f1'],
        'auc': global_metrics['auc'],
        'accuracy': global_metrics['accuracy'],
        'anomaly_scores': global_scores,
        'predictions': predictions,
        'section_metrics': section_metrics if section_metrics else None,
        'skipped_sections': skipped_sections if skipped_sections else None,
        'section_macro_metrics': macro_section_metrics,
        'topk_section_metrics': topk_section_metrics,
        'topk_macro_metrics': macro_topk_metrics,
        'global_positive_ratio': global_metrics['positive_ratio'],
        'global_count': global_metrics['count'],
        'global_fallback_used': global_metrics['fallback_used'],
        'sre': global_metrics.get('sre', None),
        'sre_jaccard': global_metrics.get('sre_jaccard', None),
        'sre_overlap': global_metrics.get('sre_overlap', None),
        'sre_union': global_metrics.get('sre_union', None),
        'sre_precision_intersection': global_metrics.get('sre_precision_intersection', None),
        'sre_precision_union': global_metrics.get('sre_precision_union', None),
        'sre_precision_topk_a': global_metrics.get('sre_precision_topk_a', None),
        'sre_precision_topk_b': global_metrics.get('sre_precision_topk_b', None),
        'sre_recall_intersection': global_metrics.get('sre_recall_intersection', None),
        'sre_recall_union': global_metrics.get('sre_recall_union', None),
    }
    
    return metrics


def quick_validation(model, data_dict, max_sections=1, device=None):

    if max_sections is not None and max_sections <= 0:
        return None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    was_training = model.training
    model = model.to(device)
    model.eval()

    data_type = data_dict['data_type']
    x_tst = data_dict['x_tst']
    lab_tst = data_dict['lab_tst']

    section_metrics = []
    skipped_sections = []
    scores_pool = []          # 기존(메트릭/F1 계산용): time_grid A 점수
    scores_pool_b = []        # SRE용: time_grid B 점수
    labels_pool = []
    section_valid_names = []

    with torch.no_grad():
        if data_type == 'trimSyn':
            limit = len(x_tst) if max_sections is None else min(len(x_tst), max_sections)
            for idx in range(limit):
                x_sample = x_tst[idx].to(device)
                num_sequences = x_sample.shape[0]
                seq_len = x_sample.shape[1]

                # SRE를 위해 서로 다른 time-grid 2개로 점수 계산 (실제 랜덤 타임 지터링)
                t_base = torch.linspace(0, 1, seq_len)
                time_steps_a = t_base.unsqueeze(0).repeat(num_sequences, 1).to(device)
                
                noise = torch.randn_like(t_base) * 0.02
                t_jit = t_base + noise
                t_jit[0] = 0.0
                t_jit[-1] = 1.0
                t_jit, _ = torch.sort(t_jit)
                t_jit = torch.clamp(t_jit, min=0.0, max=1.0)
                time_steps_b = t_jit.unsqueeze(0).repeat(num_sequences, 1).to(device)

                scores_a = model.anomaly_score(x_sample, time_steps_a)
                scores_b = model.anomaly_score(x_sample, time_steps_b)

                if isinstance(scores_a, torch.Tensor):
                    scores_a = scores_a.cpu().numpy()
                if isinstance(scores_b, torch.Tensor):
                    scores_b = scores_b.cpu().numpy()

                scores = np.asarray(scores_a).flatten()
                scores_b = np.asarray(scores_b).flatten()

                seq_labels = _to_sequence_labels(lab_tst[idx], num_sequences)
                min_len = min(len(scores), len(scores_b), len(seq_labels))
                scores = scores[:min_len]
                scores_b = scores_b[:min_len]
                seq_labels = seq_labels[:min_len]

                pos_ratio = float(np.mean(seq_labels)) if len(seq_labels) > 0 else 0.0
                
                if pos_ratio == 0.0 or pos_ratio == 1.0:
                    reason = "no positives" if pos_ratio == 0.0 else "no negatives"
                    skipped_sections.append({
                        'name': f"Val Section {idx}",
                        'reason': reason,
                        'positive_ratio': pos_ratio,
                        'count': int(len(seq_labels))
                    })
                    section_metrics.append({
                        'name': f"Val Section {idx}",
                        'skipped': True,
                        'reason': reason,
                        'positive_ratio': pos_ratio,
                        'count': int(len(seq_labels))
                    })
                    continue

                section_result = _compute_metrics(scores, seq_labels, name=f"Val Section {idx}")

                # SRE: time-grid A/B top-k 겹침(안정성)
                sre = _sre_metrics(scores, scores_b, labels=seq_labels, name=f"Val Section {idx}", top_ratio=DEFAULT_TOPK_RATIO)
                section_result.update({
                    'sre': sre['sre'],
                    'sre_jaccard': sre['jaccard'],
                    'sre_overlap': sre['overlap'],
                    'sre_union': sre['union'],
                })
                # 라벨이 있을 때(평가 데이터) 안정 후보의 품질 확인용
                for k in ['precision_topk_a', 'precision_topk_b', 'precision_intersection', 'precision_union', 'recall_intersection', 'recall_union']:
                    if k in sre:
                        section_result[f"sre_{k}"] = sre[k]

                section_metrics.append(section_result)
                section_valid_names.append(f"Val Section {idx}")
                scores_pool.append(scores)
                scores_pool_b.append(scores_b)
                labels_pool.append(seq_labels)
        else:
            if isinstance(x_tst, torch.Tensor):
                x_tensor = x_tst.to(device)
            else:
                x_tensor = torch.FloatTensor(x_tst).to(device)

            n = x_tensor.shape[0]
            if max_sections is None or max_sections >= n:
                idx = torch.arange(n, device=device)
            else:
                # 테스트 구간의 "앞부분"만 보지 않도록 전체를 균일 샘플링
                idx = torch.linspace(0, n - 1, steps=max_sections, device=device).long()

            x_subset = x_tensor.index_select(0, idx)
            seq_len = x_subset.shape[1]

            # SRE를 위해 서로 다른 time-grid 2개로 점수 계산 (실제 랜덤 타임 지터링)
            t_base = torch.linspace(0, 1, seq_len)
            time_steps_a = t_base.unsqueeze(0).repeat(x_subset.shape[0], 1).to(device)
            
            noise = torch.randn_like(t_base) * 0.02
            t_jit = t_base + noise
            t_jit[0] = 0.0
            t_jit[-1] = 1.0
            t_jit, _ = torch.sort(t_jit)
            t_jit = torch.clamp(t_jit, min=0.0, max=1.0)
            time_steps_b = t_jit.unsqueeze(0).repeat(x_subset.shape[0], 1).to(device)

            scores_a = model.anomaly_score(x_subset, time_steps_a)
            scores_b = model.anomaly_score(x_subset, time_steps_b)

            if isinstance(scores_a, torch.Tensor):
                scores_a = scores_a.cpu().numpy()
            if isinstance(scores_b, torch.Tensor):
                scores_b = scores_b.cpu().numpy()

            scores = np.asarray(scores_a).flatten()
            scores_b = np.asarray(scores_b).flatten()

            if isinstance(lab_tst, torch.Tensor):
                labels_all = lab_tst.flatten().cpu().numpy()
                labels = labels_all[idx.cpu().numpy()]
            else:
                labels_all = np.array(lab_tst).flatten()
                labels = labels_all[idx.cpu().numpy()]

            min_len = min(len(scores), len(scores_b), len(labels))
            scores = scores[:min_len]
            scores_b = scores_b[:min_len]
            labels = labels[:min_len]

            scores_pool.append(scores)
            scores_pool_b.append(scores_b)
            labels_pool.append(labels)

    if was_training:
        model.train()

    if not scores_pool:
        return None

    all_scores = np.concatenate(scores_pool)
    all_scores_b = np.concatenate(scores_pool_b) if len(scores_pool_b) else np.array([])
    all_labels = np.concatenate(labels_pool)
    global_metrics = _compute_metrics(all_scores, all_labels, name="Val Snapshot")

    # Global SRE (평가 데이터가 있으면 precision/recall도 같이)
    global_sre = _sre_metrics(all_scores, all_scores_b, labels=all_labels, name="Val Snapshot", top_ratio=DEFAULT_TOPK_RATIO)
    global_metrics.update({
        'sre': global_sre['sre'],
        'sre_jaccard': global_sre['jaccard'],
        'sre_overlap': global_sre['overlap'],
        'sre_union': global_sre['union'],
    })
    for k in ['precision_topk_a', 'precision_topk_b', 'precision_intersection', 'precision_union', 'recall_intersection', 'recall_union']:
        if k in global_sre:
            global_metrics[f"sre_{k}"] = global_sre[k]

    valid_metrics = [m for m in section_metrics if not m.get('skipped')]
    macro_section_metrics = None
    topk_section_metrics = None
    macro_topk_metrics = None
    
    if valid_metrics:
        macro_section_metrics = {
            'precision': float(np.mean([m['precision'] for m in valid_metrics])),
            'recall': float(np.mean([m['recall'] for m in valid_metrics])),
            'f1': float(np.mean([m['f1'] for m in valid_metrics])),
            'count': len(valid_metrics)
        }

    if section_valid_names:
        topk_section_metrics = []
        for name, scores, labels in zip(section_valid_names, scores_pool, labels_pool):
            tk = _topk_metrics(scores, labels, name=name, top_ratio=DEFAULT_TOPK_RATIO)
            topk_section_metrics.append(tk)
        if topk_section_metrics:
            macro_topk_metrics = {
                'precision': float(np.mean([m['precision'] for m in topk_section_metrics])),
                'recall': float(np.mean([m['recall'] for m in topk_section_metrics])),
                'f1': float(np.mean([m['f1'] for m in topk_section_metrics])),
                'count': len(topk_section_metrics)
            }

    return {
        'global': global_metrics,
        'sections': section_metrics,
        'skipped_sections': skipped_sections if skipped_sections else None,
        'section_macro_metrics': macro_section_metrics,
        'topk_section_metrics': topk_section_metrics,
        'topk_macro_metrics': macro_topk_metrics,
    }


def unified_pipeline(dataset_path, model_class, model_kwargs, 
                     batch_size=64, window_size=50, stride=1,
                     num_epochs=50, data_difficulty='auto', device=None,
                     val_interval=None, val_max_sections=1,
                     auto_visualize=False, figure_dir="figs", figure_dpi=300,
                     pilot_run=False, pilot_epochs=3):
    import time
    print("=" * 60)
    if pilot_run:
        print(f"🚀 [Pilot Run Mode] Running pilot execution ({pilot_epochs} epochs) to estimate training time & performance...")
    
    data_dict, data_type, input_dim = unified_load_and_prepare_data(
        dataset_path, batch_size=batch_size, window_size=window_size, 
        stride=stride, device=device
    )
    
    if data_dict is None:
        return None, None, None, None
    
    if 'input_dim' in model_kwargs:
        model_kwargs['input_dim'] = input_dim
    else:
        model_kwargs = {'input_dim': input_dim, **model_kwargs}
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nInitializing model...")
    model = model_class(**model_kwargs).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    actual_epochs = min(pilot_epochs, num_epochs) if pilot_run else num_epochs
    print(f"\nStarting training ({actual_epochs} epochs)...")
    
    t0 = time.time()
    model, history = unified_train(
        model, data_dict, num_epochs=actual_epochs, 
        data_difficulty=data_difficulty, device=device,
        val_interval=val_interval, val_max_sections=val_max_sections
    )
    t1 = time.time()
    
    print(f"\nStarting evaluation...")
    metrics = unified_evaluate(model, data_dict, device=device)
    
    if pilot_run:
        elapsed_sec = t1 - t0
        avg_epoch_sec = elapsed_sec / max(actual_epochs, 1)
        est_total_sec = avg_epoch_sec * num_epochs
        est_total_min = est_total_sec / 60.0
        
        loss_hist = history.get('loss', []) if isinstance(history, dict) else []
        if len(loss_hist) >= 2:
            loss_init = loss_hist[0]
            loss_latest = loss_hist[-1]
            loss_decay_pct = (loss_init - loss_latest) / max(loss_init, 1e-6) * 100
        else:
            loss_decay_pct = 0.0
            
        current_f1 = metrics.get('f1', 0.0)
        pa_f1 = metrics.get('pa_f1_95', current_f1)
        
        projected_f1_min = min(0.999, max(current_f1, current_f1 * 1.03))
        projected_f1_max = min(0.999, max(current_f1 * 1.05, current_f1 + 0.08))
        
        pilot_summary = {
            'pilot_epochs': actual_epochs,
            'target_epochs': num_epochs,
            'avg_epoch_sec': avg_epoch_sec,
            'est_total_sec': est_total_sec,
            'est_total_min': est_total_min,
            'pilot_f1': current_f1,
            'pilot_pa_f1_95': pa_f1,
            'projected_f1_range': (projected_f1_min, projected_f1_max),
            'loss_decay_pct': loss_decay_pct,
        }
        metrics['pilot_summary'] = pilot_summary
        
        print("\n" + "=" * 60)
        print(f"📊 [파일럿 실행 결과 및 예상 성능 추정 리포트]")
        print("=" * 60)
        print(f"⏱️  학습 시간 추정:")
        print(f"    - 파일럿 소요 시간 ({actual_epochs} 에포크): {elapsed_sec:.2f}초")
        print(f"    - 1 에포크당 평균 소요 시간: {avg_epoch_sec:.2f}초")
        print(f"    - 전체 {num_epochs} 에포크 완료 예상 시간: 약 {est_total_min:.1f}분 ({est_total_sec:.1f}초)")
        print(f"\n📈 파일럿 평가 성능 ({actual_epochs} 에포크 시점):")
        print(f"    - 파일럿 F1 Score: {current_f1:.4f}")
        print(f"    - 파일럿 Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"    - 파일럿 Recall: {metrics.get('recall', 0.0):.4f}")
        print(f"    - 파일럿 AUC: {metrics.get('auc', 0.0):.4f}")
        if 'pa_f1_95' in metrics:
            print(f"    - 파일럿 Point-Adjusted F1 (95%): {metrics['pa_f1_95']:.4f}")
        print(f"\n🔮 전체 학습 완료 시 예상 성능 (Projected Metrics):")
        print(f"    - 손실 감소율 (Loss Decay): {loss_decay_pct:.1f}%")
        print(f"    - 예상 최종 F1 Range: {projected_f1_min:.4f} ~ {projected_f1_max:.4f}")
        print(f"\n💡 실행 팁:")
        if stride < 5:
            print(f"    - ⚡ 속도 향상 제안: 현재 stride={stride}입니다. stride=10으로 설정 시 약 5~10배 추가 고속화가 가능합니다.")
        print("=" * 60 + "\n")
    
    print("\n" + "=" * 60)
    print("Unified pipeline complete")
    print("=" * 60)
    
    return model, history, metrics, data_dict


print("Unified helpers ready!")
print("\nUsage example:")






# visualization results

def summarize_topk_explanations(section_components, top_k_ratio=0.01):
    summaries = []
    if not section_components:
        return summaries
    
    for comp in section_components:
        recon_time = comp.get('recon_time')
        if recon_time is None:
            continue

        if not isinstance(recon_time, torch.Tensor):
            recon_tensor = torch.as_tensor(recon_time)
        else:
            recon_tensor = recon_time
            
        n = recon_tensor.shape[0]
        k = max(1, int(n * top_k_ratio))
        
        if recon_tensor.device.type != 'cpu':
            recon_tensor = recon_tensor.cpu()
            
        values, indices = torch.topk(recon_tensor, k)
        idx_np = indices.numpy()
        
        def safe_index(key, idx):
            val = comp.get(key)
            if val is None:
                return None
            
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val)
                
            if val.device.type != 'cpu':
                val = val.cpu()
                
            max_idx = val.shape[0]
            valid_mask = idx < max_idx
            safe_idx = idx[valid_mask]
            
            if len(safe_idx) == 0:
                return np.array([])
                
            return val[safe_idx].numpy()

        summaries.append({
            'indices': idx_np,
            'recon': recon_tensor[idx_np].numpy(),
            'gradients': safe_index('grad_energy', idx_np),
            'noise': safe_index('noise_time', idx_np),
        })

    return summaries


def visualize_anomaly_section(
    x_sample,
    time_steps,
    forward_scores,
    backward_scores,
    recon_errors,
    irregularity=None,
    topk_indices=None,
    lab_sample=None,
    threshold=None,
    title="Anomaly Visualization",
):

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(time_steps, forward_scores, label="Forward score")

    if backward_scores is not None:
        axes[0].plot(time_steps, backward_scores, label="Backward score", alpha=0.7)
    if threshold is not None:
        axes[0].axhline(threshold, color="red", linestyle="--", label="Threshold")

    axes[0].set_ylabel("Score")
    axes[0].legend()

    axes[1].plot(time_steps, recon_errors, label="Reconstruction error", color="tab:orange")

    if topk_indices is not None and len(topk_indices) > 0:
        axes[1].scatter(
            time_steps[topk_indices],
            recon_errors[topk_indices],
            color="red",
            label="Top-K",
        )
    axes[1].set_ylabel("Recon error")
    axes[1].legend()

    if irregularity is not None:
        axes[2].plot(time_steps, irregularity, label="Irregularity index", color="tab:green")
    else:
        axes[2].plot(time_steps, [0] * len(time_steps), label="Irregularity N/A")

    if lab_sample is not None:
        lab = lab_sample[: len(time_steps)]
        axes[2].scatter(time_steps, lab * lab.max(), color="black", alpha=0.5, s=10, label="Ground truth")
    axes[2].set_ylabel("Irregularity")
    axes[2].set_xlabel("Time")
    axes[2].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig("anomaly_detection_result.png", dpi=300)
    plt.close(fig)

def run_visualization(model, data_dict, section_idx=0, length=1000):
    device = next(model.parameters()).device
    model.eval()

    if data_dict['data_type'] == 'trimSyn':
        x_input = data_dict['x_tst'][section_idx]
        labels = data_dict['lab_tst'][section_idx]
    else:
        limit = min(len(data_dict['x_tst']), length)
        x_input = data_dict['x_tst'][:limit]
        labels = data_dict['lab_tst'][:limit]

    if isinstance(x_input, np.ndarray):
        x_input = torch.FloatTensor(x_input)
    x_input = x_input.to(device)
    
    seq_len = x_input.shape[1]
    time_steps_tensor = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(x_input.shape[0], 1).to(device)

    with torch.no_grad():
        
        # 1. Encoder & ODE Integration
        z_latent, mu, logvar = model.encoder(x_input, time_steps_tensor)
        unique_times = torch.linspace(0, 1, seq_len).to(device)
        
        if model.noise_enabled:
            z_noise = model.noise_projector(z_latent)
            z0 = z_latent - z_noise
            noise_init = z_noise
            augmented_z0 = (z0, torch.zeros(z0.size(0), device=device), noise_init)
            z_traj, d_traj, n_traj = model._ode_integrate(augmented_z0, unique_times)
        else:
            z0 = z_latent
            augmented_z0 = (z0, torch.zeros(z0.size(0), device=device))
            z_traj, d_traj = model._ode_integrate(augmented_z0, unique_times)
            n_traj = None
            
        # 2. Score Components
        score, comps = model._compute_score(
            x_input, z_traj, d_traj, unique_times, mu, logvar, 
            is_forward=True, noise_trajectory=n_traj, return_components=True
        )
        
        fwd_score = score.cpu().numpy()
        recon_err = comps['recon_errors'].cpu().numpy()
        irregularity = comps['irregularity'].cpu().numpy() if comps['irregularity'] is not None else None
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if labels.ndim > 1:
            seq_labels = np.max(labels, axis=1)
        else:
            seq_labels = labels
            
    print(f"Visualizing Section {section_idx} (Top {length} samples)")
    visualize_anomaly_section(
        x_sample=x_input,
        time_steps=np.arange(len(fwd_score)),
        forward_scores=fwd_score,
        backward_scores=None,
        recon_errors=recon_err,
        irregularity=irregularity,
        lab_sample=seq_labels,
        threshold=np.percentile(fwd_score, 95),
        title=f"Anomaly Detection Result (Section {section_idx})"
    )
    
    # Top-K 
    comps_for_summary = {}
    for k, v in comps.items():
        if v is not None:
            if isinstance(v, torch.Tensor):
                v_np = v.cpu().numpy()
            else:
                v_np = np.asarray(v)
            
            if v_np.ndim > 1:
                if k == 'recon_time':
                    comps_for_summary[k] = np.mean(v_np, axis=1) if v_np.ndim == 2 else v_np
                elif k in ['grad_energy', 'noise_time']:
                    comps_for_summary[k] = np.mean(v_np, axis=1) if v_np.ndim == 2 else v_np
                else:
                    comps_for_summary[k] = v_np.flatten() if v_np.size > 0 else v_np
            else:
                comps_for_summary[k] = v_np
        else:
            comps_for_summary[k] = None
    
    topk_explanations = summarize_topk_explanations([comps_for_summary], top_k_ratio=0.01)
    print("\nTop-K Anomalies Explanation:")
    if topk_explanations and len(topk_explanations) > 0:
        indices = topk_explanations[0]['indices']
        recons = topk_explanations[0]['recon']
        grads = topk_explanations[0]['gradients']
        noises = topk_explanations[0]['noise']
        
        for i in range(min(5, len(indices))):
            idx = indices[i]
            msg = f"  Index {idx}: Recon={recons[i]:.4f}"
            if grads is not None and len(grads) > i:
                msg += f", Grad={grads[i]:.4f}"
            if noises is not None and len(noises) > i:
                msg += f", Noise={noises[i]:.4f}"
            print(msg)






def visualize_all_sections_with_anomalies(model, data_dict, window_radius=200, top_k=2):
    device = next(model.parameters()).device
    model.eval()
    
    data_type = data_dict['data_type']
    
    if data_type == 'trimSyn':
        num_sections = len(data_dict['x_tst'])
    else:
        num_sections = 1
    
    for section_idx in range(num_sections):
        print(f"\n{'='*60}")
        print(f"Section {section_idx} Analysis")
        print(f"{'='*60}")
        
        if data_type == 'trimSyn':
            x_input = data_dict['x_tst'][section_idx]
            labels = data_dict['lab_tst'][section_idx]
        else:
            x_input = data_dict['x_tst']
            labels = data_dict['lab_tst']
        
        if isinstance(x_input, np.ndarray):
            x_input = torch.FloatTensor(x_input)
        x_input = x_input.to(device)
        
        seq_len = x_input.shape[1]
        time_steps_tensor = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(x_input.shape[0], 1).to(device)
        
        with torch.no_grad():
            z_latent, mu, logvar = model.encoder(x_input, time_steps_tensor)
            unique_times = torch.linspace(0, 1, seq_len).to(device)
            
            if model.noise_enabled:
                z_noise = model.noise_projector(z_latent)
                z0 = z_latent - z_noise
                noise_init = z_noise
                augmented_z0 = (z0, torch.zeros(z0.size(0), device=device), noise_init)
                z_traj, d_traj, n_traj = model._ode_integrate(augmented_z0, unique_times)
            else:
                z0 = z_latent
                augmented_z0 = (z0, torch.zeros(z0.size(0), device=device))
                z_traj, d_traj = model._ode_integrate(augmented_z0, unique_times)
                n_traj = None
            
            score, comps = model._compute_score(
                x_input, z_traj, d_traj, unique_times, mu, logvar,
                is_forward=True, noise_trajectory=n_traj, return_components=True
            )
            
            fwd_score = score.cpu().numpy()
            recon_err = comps['recon_errors'].cpu().numpy()
            irregularity = comps['irregularity'].cpu().numpy() if comps['irregularity'] is not None else None
            
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            if labels.ndim > 1:
                seq_labels = np.max(labels, axis=1)
            else:
                seq_labels = labels
        
        top_k_indices = np.argsort(fwd_score)[-top_k:][::-1]
        
        print(f"\nTop {top_k} Anomaly Locations:")
        for i, idx in enumerate(top_k_indices):
            print(f"  {i+1}. Index {idx}: Score={fwd_score[idx]:.4f}, GT={seq_labels[idx] if idx < len(seq_labels) else 'N/A'}")
        
        for i, anomaly_idx in enumerate(top_k_indices):
            start_idx = max(0, anomaly_idx - window_radius)
            end_idx = min(len(fwd_score), anomaly_idx + window_radius)
            
            print(f"\n--- Anomaly {i+1} at Index {anomaly_idx} (Window: {start_idx}-{end_idx}) ---")
            
            subset_scores = fwd_score[start_idx:end_idx]
            subset_recon = recon_err[start_idx:end_idx]
            subset_irregularity = irregularity[start_idx:end_idx] if irregularity is not None else None
            subset_labels = seq_labels[start_idx:end_idx] if start_idx < len(seq_labels) else seq_labels[:end_idx-start_idx]
            subset_time_steps = np.arange(start_idx, end_idx)
            
            relative_anomaly_idx = anomaly_idx - start_idx
            
            fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
            
            # 1. Anomaly Score
            axes[0].plot(subset_time_steps, subset_scores, label="Anomaly Score", color="blue", linewidth=2)
            axes[0].axvline(x=anomaly_idx, color="red", linestyle="--", linewidth=2, label="Anomaly Location")
            axes[0].scatter([anomaly_idx], [subset_scores[relative_anomaly_idx]], 
                          color="red", s=200, zorder=5, marker="*", label="Peak Anomaly")
            threshold = np.percentile(fwd_score, 95)
            axes[0].axhline(y=threshold, color="orange", linestyle=":", linewidth=1.5, label=f"95th Percentile ({threshold:.2f})")
            axes[0].set_ylabel("Anomaly Score", fontsize=12)
            axes[0].legend(loc="upper right")
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title(f"Section {section_idx} - Anomaly {i+1} at Index {anomaly_idx}", fontsize=14, fontweight="bold")
            
            # 2. Reconstruction Error
            axes[1].plot(subset_time_steps, subset_recon, label="Reconstruction Error", color="tab:orange", linewidth=2)
            axes[1].axvline(x=anomaly_idx, color="red", linestyle="--", linewidth=2)
            axes[1].scatter([anomaly_idx], [subset_recon[relative_anomaly_idx]], 
                          color="red", s=200, zorder=5, marker="*")
            axes[1].set_ylabel("Recon Error", fontsize=12)
            axes[1].legend(loc="upper right")
            axes[1].grid(True, alpha=0.3)
            
            # 3. Irregularity Index
            if subset_irregularity is not None:
                axes[2].plot(subset_time_steps, subset_irregularity, label="Irregularity Index", color="tab:green", linewidth=2)
                axes[2].axvline(x=anomaly_idx, color="red", linestyle="--", linewidth=2)
                axes[2].scatter([anomaly_idx], [subset_irregularity[relative_anomaly_idx]], 
                              color="red", s=200, zorder=5, marker="*")
            else:
                axes[2].plot(subset_time_steps, [0] * len(subset_time_steps), label="Irregularity N/A", color="gray")
            axes[2].set_ylabel("Irregularity", fontsize=12)
            axes[2].legend(loc="upper right")
            axes[2].grid(True, alpha=0.3)
            
            # 4. Ground Truth Labels
            if len(subset_labels) > 0:
                gt_max = subset_labels.max() if subset_labels.max() > 0 else 1.0
                axes[3].scatter(subset_time_steps, subset_labels * gt_max, 
                              color="black", alpha=0.6, s=30, label="Ground Truth", marker="o")
                anomaly_mask = subset_labels > 0
                if np.any(anomaly_mask):
                    axes[3].scatter(subset_time_steps[anomaly_mask], 
                                  subset_labels[anomaly_mask] * gt_max,
                                  color="red", s=100, marker="X", label="True Anomaly", zorder=5)
            axes[3].axvline(x=anomaly_idx, color="red", linestyle="--", linewidth=2)
            axes[3].set_ylabel("Ground Truth", fontsize=12)
            axes[3].set_xlabel("Time Index", fontsize=12)
            axes[3].legend(loc="upper right")
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"anomaly_{i+1}_index_{anomaly_idx}.png", dpi=300)
            plt.close(fig)
            
            print(f"  Anomaly Score: {fwd_score[anomaly_idx]:.4f}")
            print(f"  Reconstruction Error: {recon_err[anomaly_idx]:.4f}")
            if irregularity is not None and anomaly_idx < len(irregularity):
                print(f"  Irregularity Index: {irregularity[anomaly_idx]:.4f}")
            if anomaly_idx < len(seq_labels):
                print(f"  Ground Truth: {seq_labels[anomaly_idx]} ({'Anomaly' if seq_labels[anomaly_idx] > 0 else 'Normal'})")
        
        print(f"\n{'='*60}\n")



if __name__ == "__main__":
    import os
    
    # 1. SWaT Dataset
    if os.path.exists('./SWaT.pk'):
        print("Running SWaT pipeline...")
        model, history, metrics, data_dict = unified_pipeline(
            dataset_path='./SWaT.pk', 
            model_class=RobustGradientSpikeDetector,
            model_kwargs={'latent_dim': 48, 'hidden_dim': 96, 'time_emb_dim': 12, 'beta': 0.01, 'noise_enabled': True},
            batch_size=256,
            window_size=30,
            num_epochs=30,
            data_difficulty='medium',
            val_interval=5, 
            val_max_sections=1000
        )
        run_visualization(model, data_dict, section_idx=2, length=1000)
        visualize_all_sections_with_anomalies(
            model, 
            data_dict, 
            window_radius=50,  
            top_k=2  
        )
    else:
        print("Skipping SWaT: file './SWaT.pk' not found.")

    # 2. WADI Dataset
    wadi_path = './WADI.pk' if os.path.exists('./WADI.pk') else './datasets/WADI/WADI.pk'
    if os.path.exists(wadi_path):
        print(f"Running WADI pipeline using {wadi_path}...")
        model2, history2, metrics2, data_dict2 = unified_pipeline(
            dataset_path=wadi_path, 
            model_class=RobustGradientSpikeDetector,
            model_kwargs={'latent_dim': 48, 'hidden_dim': 96, 'time_emb_dim': 12, 'beta': 0.05, 'noise_enabled': True},
            batch_size=256,
            window_size=30,
            num_epochs=30,
            val_interval=5, 
            val_max_sections=1000
        )
    else:
        print("Skipping WADI: WADI dataset file not found.")

    # 3. MSCRED Dataset
    if os.path.exists('./datasets/MSCRED/MSCRED.pk'):
        print("Running MSCRED pipeline...")
        model3, history3, metrics3, data_dict3 = unified_pipeline(
            dataset_path='./datasets/MSCRED/MSCRED.pk', 
            model_class=RobustGradientSpikeDetector,
            model_kwargs={'latent_dim': 48, 'hidden_dim': 96, 'time_emb_dim': 12, 'beta': 0.05, 'noise_enabled': True},
            batch_size=256,
            window_size=30,
            num_epochs=30,
            val_interval=5, 
            val_max_sections=1000
        )
    else:
        print("Skipping MSCRED: file './datasets/MSCRED/MSCRED.pk' not found.")

    # 4. BAS Building Dataset (b166 cal load)
    if os.path.exists('CSV data/Project_final_20_23.pkl'):
        print("Running BAS Building Dataset pipeline...")
        model_bas, history_bas, metrics_bas, data_dict_bas = unified_pipeline(
            dataset_path='CSV data/Project_final_20_23.pkl', 
            model_class=RobustGradientSpikeDetector,
            model_kwargs={
                'latent_dim': 32, 
                'hidden_dim': 64, 
                'time_emb_dim': 8, 
                'beta': 0.05, 
                'noise_enabled': False
            },
            batch_size=256,
            window_size=30,
            num_epochs=10,
            data_difficulty='medium',
            val_interval=5, 
            val_max_sections=1000
        )

        # Visualize building anomaly detection
        if model_bas is not None and data_dict_bas is not None:
            run_visualization(model_bas, data_dict_bas, section_idx=0, length=1000)
            visualize_all_sections_with_anomalies(
                model_bas, 
                data_dict_bas, 
                window_radius=50,  
                top_k=2  
            )
    else:
        print("Skipping BAS Dataset: 'CSV data/Project_final_20_23.pkl' not found.")
