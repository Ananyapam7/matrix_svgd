from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from math import pi


def _sum_log_exp(X, mus, dcovs, weights):
    dim = torch.tensor(mus.size(1), dtype=torch.float32)
    _lnD = torch.sum(torch.log(dcovs), dim=1)

    diff = X.unsqueeze(0) - mus.unsqueeze(1)  # c x n x d
    diff_times_inv_cov = diff * (1./ dcovs).unsqueeze(1)  # c x n x d
    sum_sq_dist_times_inv_cov = torch.sum(diff_times_inv_cov * diff, dim=2)  # c x n 
    ln2piD = torch.log(torch.tensor(2 * np.pi)) * dim
    log_coefficients = (ln2piD + _lnD).unsqueeze(1) # c x 1
    log_components = -0.5 * (log_coefficients + sum_sq_dist_times_inv_cov)  # c x n
    log_weighted = log_components + torch.log(weights).unsqueeze(1)  # c x n + c x 1
    log_shift = torch.max(log_weighted, dim=0, keepdim=True)[0]

    return log_weighted, log_shift



def _log_gradient(X, mus, dcovs, weights):  
    x_shape = X.shape
    assert len(x_shape) == 2, 'illegal inputs'

    def posterior(X):
        log_weighted, log_shift = _sum_log_exp(X, mus, dcovs, weights)
        prob = torch.exp(log_weighted - log_shift) # c x n
        prob = prob / torch.sum(prob, dim=0, keepdim=True)
        return prob

    diff = X.unsqueeze(0) - mus.unsqueeze(1)  # c x n x d
    diff_times_inv_cov = -diff * (1./dcovs).unsqueeze(1)  # c x n x d

    P = posterior(X)  # c x n
    score = torch.matmul(
        P.t().unsqueeze(1), # n x 1 x c
        diff_times_inv_cov.transpose(0, 1).transpose(1, 2) # n x c x d
    ) 
    return torch.squeeze(score) # n x d


def mixture_weights_and_grads(X, mus=None, dcovs=None, weights=None):  
    x_shape = X.shape
    assert len(x_shape) == 2, 'illegal inputs'
    
    if mus is None:
        mus = X.detach()
    if dcovs is None:
        dcovs = torch.ones_like(mus)
    # uniform weights, only care about ratio
    if weights is None: 
        weights = torch.ones(mus.size(0))

    log_weighted, log_shift = _sum_log_exp(X, mus, dcovs, weights)
    exp_log_shifted = torch.exp(log_weighted - log_shift) # c x n
    exp_log_shifted_sum = torch.sum(exp_log_shifted, dim=0, keepdim=True) # 1 x n
    p = exp_log_shifted / exp_log_shifted_sum

    # weights
    mix = p.t()  # n * c
    d_log_gmm = _log_gradient(X, mus, dcovs, weights) # n * d

    d_log_gau = -(X.unsqueeze(1) - mus.unsqueeze(0)) / dcovs.unsqueeze(0) # n x c x d
    mix_grad = d_log_gau - d_log_gmm.unsqueeze(1)

    # c * n, c * n * d
    return mix.t(), mix_grad.transpose(0, 1)



#from models import GaussianMixture
#
#def _simulate_mixture_target(n_components=10, dim = 1, val=5., seed=123):
#
#    with tf.variable_scope('p_target') as scope:
#        np.random.seed(seed)
#        mu0 = tf.get_variable('mu', initializer=np.random.uniform(-val, val, size=(n_components, dim)).astype('float32'), dtype=tf.float32,  trainable=False)
#
#        log_sigma0 = tf.zeros((n_components, dim))
#        weights0 = tf.ones(n_components) / n_components
#        p_target = GaussianMixture(n_components, mu0, log_sigma0, weights0)
#
#        return p_target
#
#
#
#if __name__ == '__main__':
#
#    session_config = tf.ConfigProto(
#        allow_soft_placement=True,
#        gpu_options=tf.GPUOptions(allow_growth=True),
#    )
#
#    from ops import rbf_kernel
#    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
#
#        x_train = tf.constant(np.random.normal(size=(3, 2)).astype('float32'))
#        k1, dk1 = rbf_kernel(x_train)
#        k2, dk2 = rbf_kernel(x_train, to3d=True)
#
#        p_target = _simulate_mixture_target(n_components=3, dim=2, val=1.0)
#        Hs = tf.hessians(p_target.log_prob(x_train), tf.split(x_train, num_or_size_splits=3, axis=0))
#
#        tf.global_variables_initializer().run()
#
#        hs = sess.run([Hs])
#        print(hs.shape)
#        
#        #dxk1, dxk2 = sess.run([dk1, dk2])
#        #print (dxk1)
#        #print (np.sum(dxk2, 0))
#        #k1, dk1 = rbf_kernel(x_train)
#
