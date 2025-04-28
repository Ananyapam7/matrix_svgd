from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from math import pi

import sys
from ops import svgd_gradient, sqr_dist, rbf_kernel
from gmm_models import mixture_weights_and_grads

class SVGD(nn.Module):

    def __init__(self, config):
        super(SVGD, self).__init__()
        self.config = config

        # Initialize weights
        self.W = nn.Parameter(
            torch.rand((self.config.n_particles, self.config.dim), dtype=torch.float32) * 0.2 - 0.1
        )

        # Initialize covariance matrix
        self.register_buffer('cov_dW', torch.ones((self.config.dim, self.config.dim)))

        # Initialize momentum for pSGLD
        if self.config.method in ['SGLD', 'pSGLD']:
            self.register_buffer('acc', torch.ones_like(self.W))
            
        # Initialize step counter
        self.register_buffer('step', torch.tensor(1.0))

    def forward(self, inputs, training=False):
        X, y = inputs
        batch_size = X.size(0)
        
        # Forward pass
        z = torch.sum(X.unsqueeze(0) * self.W.unsqueeze(1), -1)  # n_p * B
        y_prob = torch.sigmoid(z)  # n_p * B

        # Fix: Ensure y has the right shape for broadcasting
        y_expand = y.squeeze().unsqueeze(0)  # 1 * B
        dy = (y_expand - y_prob) / (y_prob - y_prob**2 + 1e-8)  # n_p * B
        dz = torch.sigmoid(z) * (1. - torch.sigmoid(z))  # n_p * B
        dW = torch.mean(X.unsqueeze(0) * (dz * dy).unsqueeze(2), 1)

        grad_loglik_z = (y_prob - y_expand) / (y_prob - y_prob**2 + 1e-8) * dz 
        grad_loglik_W = grad_loglik_z.unsqueeze(2) * X.unsqueeze(0)  # n_p * B * d

        # Calculate mean gradient across particles
        mean_dW = torch.mean(grad_loglik_W, dim=0)  # B * d
        
        # Calculate covariance for each particle
        diff_dW = grad_loglik_W - mean_dW.unsqueeze(0)  # n_p * B * d
        cov_dW_ = torch.mean(
            torch.matmul(
                diff_dW.transpose(1, 2),  # n_p * d * B
                diff_dW  # n_p * B * d
            ),
            dim=0
        ) / batch_size  # d * d

        W_grads = dW * self.config.n_train - self.W

        y_pred = torch.mean(y_prob, 0)
        ll = torch.mean(
            y.squeeze() * torch.log(y_pred + 1e-3) + (1. - y.squeeze()) * torch.log(1. - y_pred + 1e-3)
        )
        
        # Fix: Convert boolean tensor to float before taking mean
        accuracy = torch.mean(
            ((y.squeeze() > 0.5).float() == (y_pred > 0.5).float()).float()
        )

        # Update covariance
        rho = min(1. - 1./self.step, 0.95)
        self.cov_dW.copy_(rho * self.cov_dW + (1. - rho) * cov_dW_)
        H_inv = torch.inverse(self.cov_dW + 1e-2*torch.eye(self.config.dim))

        # SVGD gradients
        svgd_grad = svgd_gradient(self.W, W_grads, kernel=self.config.kernel)
        self.svgd_grads = [-svgd_grad]

        # KFAC gradients
        if self.config.method == 'svgd_kfac':
            kfac_grad = torch.matmul(svgd_grad, H_inv)
            self.kfac_grads = [-kfac_grad]
        
        # Mixture KFAC gradients
        if self.config.method == 'mixture_kfac':
            self.mixture_grads = [self.mixture_kfac_gradient(self.W, W_grads, H_inv)]

        # SGLD/pSGLD gradients
        if self.config.method in ['SGLD', 'pSGLD']:
            G = torch.sqrt(self.acc + 1e-6)
            self.psgld_grads = [
                -self.config.learning_rate * (dW - self.W/self.config.n_train) / G + \
                torch.sqrt(self.config.learning_rate / G) * 2./self.config.n_train * torch.randn_like(self.W)
            ]

        # Store values for training
        self.ll = ll
        self.accuracy = accuracy
        self.log_prob = ll

        return ll, accuracy

    def mixture_kfac_gradient(self, W, W_grads, H_inv):
        def _weighted_svgd(x, d_log_pw, w):
            kxy, dxkxy = rbf_kernel(x, to3d=True)
            velocity = torch.sum(
                w.unsqueeze(0).unsqueeze(2) * 
                kxy.unsqueeze(2) * 
                d_log_pw.unsqueeze(0),
                dim=1
            ) + torch.sum(
                w.unsqueeze(1).unsqueeze(2) * dxkxy,
                dim=0
            )
            return velocity

        def _mixture_svgd_grads(x, d_log_p, mix, mix_grads, H_inv):
            velocity = 0
            for i in range(self.config.n_particles):
                w_i_svgd = _weighted_svgd(x, d_log_p + mix_grads[i], mix[i])
                delta = torch.matmul(w_i_svgd, H_inv)
                velocity += mix[i].unsqueeze(1) * delta
            return velocity

        mix, mix_grads = mixture_weights_and_grads(W)
        velocity = _mixture_svgd_grads(W, W_grads, mix, mix_grads, H_inv)
        return -velocity



