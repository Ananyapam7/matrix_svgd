from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from tqdm import tqdm

def sqr_dist(x, y, e=1e-8):
    xx = torch.sum(torch.square(x) + 1e-10, dim=1)
    yy = torch.sum(torch.square(y) + 1e-10, dim=1)
    xy = torch.matmul(x, y.t())
    dist = xx.unsqueeze(1) + yy.unsqueeze(0) - 2. * xy
    return dist

def median_distance(H):
    V = H.reshape(-1)
    n = V.numel()
    top_k, _ = torch.topk(V, k=(n // 2) + 1)
    if n % 2 == 0:
        return (top_k[-1] + top_k[-2]) / 2.0
    return top_k[-1]

def poly_kernel(x, subtract_mean=True, e=1e-8):
    if subtract_mean:
        x = x - torch.mean(x, dim=0)
    kxy = 1 + torch.matmul(x, x.t())
    kxkxy = x * x.size(0)
    return kxy, dxkxy

def rbf_kernel(x, h=-1, to3d=False):
    H = sqr_dist(x, x)
    if h == -1:
        h = torch.maximum(torch.tensor(1e-6), median_distance(H))

    kxy = torch.exp(-H / h)
    dxkxy = -torch.matmul(kxy, x)
    sumkxy = torch.sum(kxy, dim=1, keepdim=True)
    dxkxy = (dxkxy + x * sumkxy) * 2. / h

    if to3d:
        dxkxy = -(x.unsqueeze(1) - x.unsqueeze(0)) * kxy.unsqueeze(2) * 2. / h
    return kxy, dxkxy

def imq_kernel(x, h=-1):
    H = sqr_dist(x, x)
    if h == -1:
        h = median_distance(H)

    kxy = 1. / torch.sqrt(1. + H / h) 

    dxkxy = .5 * kxy / (1. + H / h)
    dxkxy = -torch.matmul(dxkxy, x)
    sumkxy = torch.sum(kxy, dim=1, keepdim=True)
    dxkxy = (dxkxy + x * sumkxy) * 2. / h

    return kxy, dxkxy

def kernelized_stein_discrepancy(X, score_q, kernel='rbf', h=-1, **model_params):
    n, dim = torch.tensor(X.size(0), dtype=torch.float32), torch.tensor(X.size(1), dtype=torch.float32)
    Sqx = score_q(X, **model_params)

    H = sqr_dist(X, X)
    if h == -1:
        h = median_distance(H) # 2sigma^2
    h = torch.sqrt(h/2.)
    # compute the rbf kernel
    Kxy = torch.exp(-H / h ** 2 / 2.)

    Sqxdy = -(torch.matmul(Sqx, X.t()) - torch.sum(Sqx * X, dim=1, keepdim=True)) / (h ** 2)

    dxSqy = Sqxdy.t()
    dxdy = (-H / (h ** 4) + dim / (h ** 2))

    M = (torch.matmul(Sqx, Sqx.t()) + Sqxdy + dxSqy + dxdy) * Kxy 
    return M

def svgd_gradient(x, grad, kernel='rbf', temperature=1., u_kernel=None, **kernel_params):
    assert x.shape[1:] == grad.shape[1:], 'illegal inputs and grads'
    p_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(x.size(0), -1)
        grad = grad.reshape(grad.size(0), -1)

    if u_kernel is not None:
        kxy, dxkxy = u_kernel['kxy'], u_kernel['dxkxy']
        dxkxy = dxkxy.reshape(x.shape)
    else:
        if kernel == 'rbf':
            kxy, dxkxy = rbf_kernel(x, **kernel_params)
        elif kernel == 'poly':
            kxy, dxkxy = poly_kernel(x)
        elif kernel == 'imq':
            kxy, dxkxy = imq_kernel(x)
        elif kernel == 'none':
            kxy = torch.eye(x.size(0))
            dxkxy = torch.zeros_like(x)
        else:
            raise NotImplementedError

    svgd_grad = (torch.matmul(kxy, grad) + temperature * dxkxy) / torch.sum(kxy, dim=1, keepdim=True)

    svgd_grad = svgd_grad.reshape(p_shape)
    return svgd_grad

def lrelu(x, leak=0.2, name="lrelu"):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * torch.abs(x)

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * torch.where(x > 0.0, x, alpha * torch.exp(x) - alpha)

def huber_loss(labels, predictions, delta=1.0):
    residual = torch.abs(predictions - labels)
    condition = residual < delta
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * torch.square(torch.tensor(delta))
    return torch.where(condition, small_res, large_res)

def conv2d(inputs, num_outputs, activation_fn=torch.nn.ReLU(),
           kernel_size=5, stride=2, padding='same', name="conv2d"):
    return torch.nn.Conv2d(
        in_channels=inputs.size(1),
        out_channels=num_outputs,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True
    )(inputs)

def deconv2d(inputs, num_outputs, activation_fn=torch.nn.ReLU(),
        kernel_size=5, stride=2, padding='same', name="deconv2d"):
    return torch.nn.ConvTranspose2d(
        in_channels=inputs.size(1),
        out_channels=num_outputs,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True
    )(inputs)

def fc(input, output_shape, activation_fn=torch.nn.ReLU(), init=None, name="fc"):
    if init is None: 
        init = torch.nn.init.kaiming_uniform_
    layer = torch.nn.Linear(input.size(1), int(output_shape), bias=True)
    init(layer.weight)
    return activation_fn(layer(input))


