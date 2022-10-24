import torch

def calculate_squared_distances(a, b):
    a = a.view([a.data.shape[0], 1, -1])
    b = b.view([1, b.data.shape[0], -1])
    d = a-b
    return (d*d).sum(2)


def plummer_kernel(a, b, dimension, epsilon):
    r = calculate_squared_distances(a, b) + epsilon**2
    return torch.pow(r, -(dimension-2) / 2)


def get_potentials(x, y, dimension, cur_epsilon):
    x_fixed, y_fixed = x.detach(), y.detach()
    pk_xx = plummer_kernel(x_fixed, x, dimension, cur_epsilon)
    pk_yx = plummer_kernel(y, x, dimension, cur_epsilon)
    pk_yy = plummer_kernel(y_fixed, y, dimension, cur_epsilon)
    kxx, kyx, kxy, kyy = pk_xx.sum(0) / x.data.shape[0], pk_yx.sum(0) / y.data.shape[0], pk_yx.sum(1) / x.data.shape[0], pk_yy.sum(0) / y.data.shape[0]
    return kxx - kyx, kxy - kyy


def mean_squared_error(x, y):
    d = (x - y)**2
    return d.mean()
