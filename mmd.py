"""
mmd.py -- MMD energy minimizer library, using pytorch.

Computes and minimizes MMD^2 between a learnable particle set and a target
distribution using gradient descent. The caller specifies the kernel to use
and the target distribution to estimate, then passes the result to run_mmd.

Tensor naming: name_shape, e.g. x_nd is an (n, d) tensor. Scalars have no
shape suffix.

Usage:
    from mmd import run_mmd, run_mmd_lbfgs, init_mmd, mmd_step, solve_optimal_weights

    particles, history = run_mmd(kernel, integral, constant, n=n, d=d, lr=lr, epochs=epochs, weighted=True)
    particles, history = run_mmd_lbfgs(kernel, integral, constant, n=n, d=d, epochs=epochs, weighted=True)

    # Custom loop with utils.BandwidthScheduler: x_nd, optimizer = init_mmd(n, d, lr); each epoch,
    # loss, grad_norm, w_n1 = mmd_step(x_nd, optimizer, scheduler.kernel, scheduler.integral,
    # scheduler.constant, w_n1, weighted=True); scheduler.step(loss)
"""

from __future__ import annotations

import torch
from tqdm import tqdm


# ------------------------------------------------------------
# MMD energy and optimal weights (Lemma 4: Kw = z)
# ------------------------------------------------------------
def mmd_energy(x_nd, w_n1, kernel, integral, constant):
    """
    MMD^2 = constant - (2/n) sum_i w_i int K(x_i, y) dmu(y) + (1/n^2) sum_{ij} w_i w_j K(x_i, x_j)
    """

    n, _ = x_nd.shape

    integral_term = (w_n1 * integral(x_nd)).sum()
    Kernel_nn = kernel(x_nd, x_nd)
    kernel_term = ((w_n1 @ w_n1.T) * Kernel_nn).sum() / (n ** 2)

    return constant - 2 * integral_term / n + kernel_term


def solve_optimal_weights(x_nd, kernel, integral, reg=0.0):
    """
    For fixed particle positions, solve the linear system for optimal weights
    (Lemma 4: minimizer of MMD^2 in w satisfies K w = n z).

    With our scaling (weights in mmd_energy use factor 1/n and 1/n^2), the
    optimal w satisfies K w = n * z, where K_ij = k(x_i, x_j) and z_i = int k(x_i, y) dP(y).

    Parameters
    ----------
    x_nd : torch.Tensor, shape (n, d)
        Particle positions (can be detached; used only to form K and z).
    kernel : callable (x_nd, y_md) -> (n, m) kernel matrix
    integral : callable (x_nd) -> (n, 1) int K(x_i, y) dP(y)
    reg : float
        Diagonal regularization (K + reg*I) for numerical stability.

    Returns
    -------
    w_n1 : torch.Tensor, shape (n, 1)
        Optimal weights (same device/dtype as x_nd).

    Notes
    -----
    In general the optimal weights can be negative and will not sum to 1.
    Both constraints (non-negativity and sum-to-one) can be enforced if
    required; the results essentially continue to hold. See e.g. Section 2.3
    of Karvonen et al. [2018].
    """
    n = x_nd.shape[0]
    K_nn = kernel(x_nd, x_nd)
    z_n1 = integral(x_nd)
    K_reg = K_nn + reg * torch.eye(n, device=x_nd.device, dtype=x_nd.dtype)
    w_n1 = torch.linalg.solve(K_reg, n * z_n1)
    return w_n1


def init_mmd(n, d, lr, x0_nd=None, device=None):
    """
    Create particle parameters and AdamW optimizer (same setup as run_mmd).
    """
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )
    if x0_nd is not None:
        x_nd = torch.nn.Parameter(x0_nd)
    else:
        x_nd = torch.nn.Parameter(torch.randn(n, d, device=device, dtype=torch.float64))
    optimizer = torch.optim.AdamW([x_nd], lr=lr, weight_decay=1e-4)
    return x_nd, optimizer


def mmd_step(x_nd, optimizer, kernel, integral, constant, w_n1=None, weighted=False):
    """
    One optimization step: zero grad, optional optimal weights, MMD loss, backward, Adam step.

    Returns
    -------
    loss : torch.Tensor  (0-d, on device)
        Scalar loss tensor. Call .item() at the END of training
    grad_norm : torch.Tensor  (0-d, on device)
    w_n1 : torch.Tensor
         Weights used this step (n, 1); for weighted=True, optimal weights for current positions.

    """
    optimizer.zero_grad()
    n = x_nd.shape[0]
    if weighted:
        w_n1 = solve_optimal_weights(x_nd.detach(), kernel, integral).detach()
    else:
        if w_n1 is None:
            w_n1 = torch.ones(n, 1, device=x_nd.device, dtype=x_nd.dtype)

    loss = mmd_energy(x_nd, w_n1, kernel, integral, constant)
    loss.backward()

    if x_nd.grad is not None:
        grad_norm = x_nd.grad.detach().norm()
    else:
        grad_norm = torch.zeros((), device=x_nd.device, dtype=x_nd.dtype)
    optimizer.step()
    return loss.detach(), grad_norm, w_n1


# Training
def run_mmd(kernel, integral, constant, *,
    x0_nd=None, n, d, lr, epochs, weighted=False, verbose=False, conv_threshold=0.0, log_every=100, device=None):
    """
    Train n particles in R^d to minimize MMD^2 against a target.
    Particles are always initialized randomly (i.i.d. N(0, 1))

    When weighted=True, uses the optimal weights : at each iteration,
    (1) for current positions, solve the linear system K w = n z for optimal
        weights (Lemma 4), and
    (2) take a gradient step on positions only.
    When weighted=False, uses uniform weights w_i = 1 and optimizes positions only.

    Parameters
    ----------
    kernel    : (x_nd, y_md) -> (n, m) kernel matrix
    integral : (x_nd) -> (n, 1) closed-form int K(x_i, y) dmu(y)
    constant  : scalar int int K(x, y) dmu(x) dmu(y)
    n         : number of particles
    d         : dimension
    lr        : learning rate
    epochs    : number of Adam steps
    weighted   : if True, use optimal weights (solve Kw = nz) each step; if False, uniform weights
    verbose   : print debugging output
    conv_threshold : threshold for convergence
    log_every : record positions/weights to history every N epochs (0 = never). Loss and
                grad_norm are always recorded. Default 100 — set to 1 to restore old behavior.
    device    : what device to run on

    Returns
    -------
    particles_nd : (n, d) numpy array of final positions
    history      : dict with 'loss' and 'grad_norm' lists
    weights (optional) : (n, 1) numpy array, only when weighted=True
    """

    history = {'loss': [], 'grad_norm': [], 'positions': [], 'weights': []}


    rng = tqdm(range(epochs), desc="MMD") if verbose else range(epochs)

    x_nd, optimizer = init_mmd(n, d, lr, x0_nd, device=device)
    w_n1 = None

    if verbose:
        print()
        print(f"Running: n = {n}, d={d}, lr={lr}, epochs={epochs}, weighted={weighted} on device {x_nd.device}")

    loss_hist = torch.empty(epochs, device=x_nd.device, dtype=x_nd.dtype)
    grad_hist = torch.empty(epochs, device=x_nd.device, dtype=x_nd.dtype)
    n_recorded = 0

    for epoch in rng:
        if conv_threshold > 0 and epoch >= 50 and epoch % 50 == 0:
            window = loss_hist[epoch - 50:epoch]
            if (window - window[0]).abs().max().item() < conv_threshold:
                if verbose:
                    print(f"Converged at epoch {epoch}")
                break

        loss, grad_norm, w_n1 = mmd_step(
            x_nd, optimizer, kernel, integral, constant, w_n1, weighted
        )
        loss_hist[epoch] = loss
        grad_hist[epoch] = grad_norm
        n_recorded = epoch + 1


        if log_every > 0 and epoch % log_every == 0:
            history['positions'].append(x_nd.detach().cpu().numpy().copy())
            history['weights'].append(w_n1.detach().cpu().numpy().copy())

    if n_recorded:
        history['loss'] = loss_hist[:n_recorded].cpu().tolist()
        history['grad_norm'] = grad_hist[:n_recorded].cpu().tolist()

    if weighted:
        # Final optimal weights at converged positions
        with torch.no_grad():
            w_n1 = solve_optimal_weights(x_nd.detach(), kernel, integral)
        return x_nd.detach().cpu().numpy(), history, w_n1.cpu().numpy()
    else:
        return x_nd.detach().cpu().numpy(), history


def run_mmd_lbfgs(kernel, integral, constant, *,
                  x0_nd=None, n, d, lr=1.0, epochs=100, weighted=False,
                  history_size=10, verbose=False, conv_threshold=0.0,
                  log_every=100, device=None):
    """
    Like run_mmd but uses L-BFGS instead of AdamW. Same return format.

    Each epoch = one optimizer.step(closure) call, which may internally
    evaluate the closure multiple times for the strong-Wolfe line search.

    For weighted=True, uses alternating minimization: solve optimal w once
    per epoch (outside the closure), then take an L-BFGS step on positions
    with w held fixed. This is cheaper than re-solving inside the closure
    but means the line search is optimizing a surrogate (w is stale within
    each epoch); the joint MMD² is not guaranteed to decrease monotonically,
    only the fixed-w surrogate does. In practice this converges fine.

    Parameters
    ----------
    kernel       : (x_nd, y_md) -> (n, m) kernel matrix
    integral     : (x_nd) -> (n, 1) closed-form int K(x_i, y) dmu(y)
    constant     : scalar int int K(x, y) dmu(x) dmu(y)
    n            : number of particles
    d            : dimension
    lr           : L-BFGS step size (default 1.0 works well with strong Wolfe)
    epochs       : number of L-BFGS steps
    weighted     : if True, solve Kw = nz each epoch before the L-BFGS step
    history_size : number of curvature pairs stored by L-BFGS (default 10)
    verbose      : print progress bar and run info
    conv_threshold, log_every, device : same semantics as run_mmd

    Returns
    -------
    particles_nd : (n, d) numpy array of final positions
    history      : dict with 'loss', 'grad_norm', 'positions', 'weights' lists
    weights (optional) : (n, 1) numpy array, only when weighted=True
    """
    history = {'loss': [], 'grad_norm': [], 'positions': [], 'weights': []}

    x_nd, _ = init_mmd(n, d, lr, x0_nd, device=device)
    optimizer = torch.optim.LBFGS(
        [x_nd], lr=lr, history_size=history_size, line_search_fn='strong_wolfe'
    )
    w_n1 = torch.ones(n, 1, device=x_nd.device, dtype=x_nd.dtype)

    if verbose:
        print()
        print(f"Running L-BFGS: n={n}, d={d}, lr={lr}, epochs={epochs}, "
              f"weighted={weighted} on device {x_nd.device}")

    rng = tqdm(range(epochs), desc="MMD L-BFGS") if verbose else range(epochs)

    loss_hist = torch.empty(epochs, device=x_nd.device, dtype=x_nd.dtype)
    grad_hist = torch.empty(epochs, device=x_nd.device, dtype=x_nd.dtype)
    n_recorded = 0

    for epoch in rng:
        if conv_threshold > 0 and epoch >= 50 and epoch % 50 == 0:
            window = loss_hist[epoch - 50:epoch]
            if (window - window[0]).abs().max().item() < conv_threshold:
                if verbose:
                    print(f"Converged at epoch {epoch}")
                break

        if weighted:
            w_n1 = solve_optimal_weights(x_nd.detach(), kernel, integral).detach()

        def closure():
            optimizer.zero_grad()
            loss = mmd_energy(x_nd, w_n1, kernel, integral, constant)
            loss.backward()
            return loss

        loss_t = optimizer.step(closure)

        loss_hist[epoch] = loss_t.detach()
        grad_hist[epoch] = (
            x_nd.grad.detach().norm() if x_nd.grad is not None
            else torch.zeros((), device=x_nd.device, dtype=x_nd.dtype)
        )
        n_recorded = epoch + 1

        if log_every > 0 and epoch % log_every == 0:
            history['positions'].append(x_nd.detach().cpu().numpy().copy())
            history['weights'].append(w_n1.detach().cpu().numpy().copy())

    if n_recorded:
        history['loss'] = loss_hist[:n_recorded].cpu().tolist()
        history['grad_norm'] = grad_hist[:n_recorded].cpu().tolist()

    if weighted:
        with torch.no_grad():
            w_n1 = solve_optimal_weights(x_nd.detach(), kernel, integral)
        return x_nd.detach().cpu().numpy(), history, w_n1.cpu().numpy()
    else:
        return x_nd.detach().cpu().numpy(), history
