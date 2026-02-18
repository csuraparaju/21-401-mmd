"""
mmd.py -- MMD energy minimizer library, using pytorch.

Computes and minimizes MMD^2 between a learnable particle set and a target
distribution using gradient descent. The caller specifies the kernel to use
and the target distribution to estimate, then passes the result to run_mmd.

Tensor naming: name_shape, e.g. x_nd is an (n, d) tensor. Scalars have no
shape suffix.

Usage:
    from mmd import run_mmd, gaussian_on_gaussian

    particles, history = run_mmd(
        my_kernel, my_integral, my_constant,
        n=256, d=2, lr=0.01, epochs=1000,
    )
"""

from __future__ import annotations

import torch

# Default kernel implementation 
def gaussian_kernel(x_nd, y_md, sigma_sq):
    """
    K(x, y) = exp(-||x - y||^2 / (2 sigma^2))
    """

    xx_n1 = (x_nd ** 2).sum(dim=1, keepdim=True)
    yy_m1 = (y_md ** 2).sum(dim=1, keepdim=True)
    dist_sq_nm = xx_n1 + yy_m1.T - 2 * x_nd @ y_md.T

    return torch.exp(-dist_sq_nm / (2 * sigma_sq))


# Default closed-form integrals and constants against target measures
def gaussian_integral(x_nd, sigma_sq, target_var, d):
    """
    int K(x_i, y) dmu(y) for mu = N(0, target_var * I_d).
                                = (sigma_sq / (target_var + sigma_sq))^{d/2}
                                * exp(-||x_i||^2 / (2 (target_var + sigma_sq)))
    """
    s = target_var + sigma_sq
    xx_n1 = (x_nd ** 2).sum(dim=1, keepdim=True)
    return (sigma_sq / s) ** (d / 2) * torch.exp(-xx_n1 / (2 * s))


def gaussian_constant(sigma_sq, target_var, d):
    """
    int int K(x, y) dmu(x) dmu(y) for mu = N(0, target_var * I_d).
                                         = (sigma_sq / (2 * target_var + sigma_sq))^{d/2}
    """
    return (sigma_sq / (2 * target_var + sigma_sq)) ** (d / 2)



# MMD energy
def mmd_energy(x_nd, kernel, integral, constant):
    """
    MMD^2 = constant - (2/n) sum_i int K(x_i, y) dmu(y) + (1/n^2) sum_{ij} K(x_i, x_j)
    """

    n, _ = x_nd.shape

    integral_term = integral(x_nd).sum()
    Kernel_nn = kernel(x_nd, x_nd)

    return constant - 2 * integral_term / n + Kernel_nn.sum() / (n ** 2)



# Training
def run_mmd(kernel, integral, constant, *, n, d, lr, epochs):
    """
    Train n particles in R^d to minimise MMD^2 against a target.

    Parameters
    ----------
    kernel    : (x_nd, y_md) -> (n, m) kernel matrix
    integral : (x_nd) -> (n, 1) closed-form int K(x_i, y) dmu(y)
    constant  : scalar int int K(x, y) dmu(x) dmu(y)
    n         : number of particles
    d         : dimension
    lr        : learning rate
    epochs    : number of Adam steps

    Returns
    -------
    particles_nd : (n, d) numpy array of final positions
    history      : dict with 'loss' and 'grad_norm' lists
    """

    device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )

    x_nd = torch.nn.Parameter(torch.randn(n, d, device=device))
    optimizer = torch.optim.Adam([x_nd], lr=lr)
    history = {'loss': [], 'grad_norm': []}

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = mmd_energy(x_nd, kernel, integral, constant)
        loss.backward()

        grad_norm = torch.norm(x_nd.grad).item() if x_nd.grad is not None else 0.0
        history['loss'].append(loss.item())
        history['grad_norm'].append(grad_norm)

        optimizer.step()

    return x_nd.detach().cpu().numpy(), history


