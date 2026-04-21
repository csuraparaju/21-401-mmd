import math

import torch


class BandwidthScheduler:
    """
    Decrease kernel bandwidth when training loss plateaus: advance to the next
    bandwidth after `patience` epochs without improvement beyond `threshold`.

    After a bandwidth change, `cooldown` steps skip plateau detection so the
    optimizer can adapt to the new kernel.
    """

    def __init__(
        self,
        kernel_factory,
        integral_factory,
        constant_factory,
        bandwidths,
        patience=50,
        threshold=1e-6,
        cooldown=10,
    ):
        if not bandwidths:
            raise ValueError("bandwidths must be non-empty")
        self._kernel_factory = kernel_factory
        self._integral_factory = integral_factory
        self._constant_factory = constant_factory
        self._bandwidths = list(bandwidths)
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown

        self._idx = 0
        self._best = math.inf
        self._num_bad = 0
        self._cooldown_counter = 0

    @property
    def bandwidth(self):
        return self._bandwidths[self._idx]

    @property
    def kernel(self):
        return self._kernel_factory(self.bandwidth)

    @property
    def integral(self):
        return self._integral_factory(self.bandwidth)

    @property
    def constant(self):
        return self._constant_factory(self.bandwidth)

    @property
    def exhausted(self):
        on_last = self._idx >= len(self._bandwidths) - 1
        return on_last and self._num_bad >= self.patience

    def step(self, loss: float) -> bool:
        """Update plateau state; return True if the bandwidth changed this call."""
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return False

        if loss < self._best - self.threshold:
            self._best = loss
            self._num_bad = 0
        else:
            self._num_bad += 1

        if self._num_bad < self.patience:
            return False
        if self._idx >= len(self._bandwidths) - 1:
            return False

        self._idx += 1
        self._best = math.inf
        self._num_bad = 0
        self._cooldown_counter = self.cooldown
        return True


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

def laplace_kernel(x_nd, y_md, sigma, eps=1e-8):
    """
    K(x, y) = exp(-||x - y|| / sigma)
    eps: added inside sqrt to avoid infinite gradient when ||x-y||=0 (same particle or collision).
    """
    x_n1 = (x_nd ** 2).sum(dim=1, keepdim=True)
    y_m1 = (y_md ** 2).sum(dim=1, keepdim=True)
    dist_sq_nm = (x_n1 + y_m1.T - 2 * x_nd @ y_md.T).clamp(min=0)
    dist_nm = (dist_sq_nm + eps).sqrt()
    return torch.exp(-dist_nm / sigma)

def laplace_gaussian_integral(x_nd, sigma, target_var, m=2000, device=None):
    """
    int K(x_i, y) dmu(y) for mu = N(0, target_var * I_d) is hard to compute in closed-form.
    Monte Carlo estimate: sample y_1..y_m ~ mu; compute (1/m) sum_j K(x_i, y_j).
    """
    n, d = x_nd.shape
    samples_md = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d) * target_var).sample((m,))
    kernel_values_ndm = laplace_kernel(x_nd, samples_md, sigma)
    return kernel_values_ndm.mean(dim=1, keepdim=True)

def laplace_constant(sigma, target_var, d, m=2000, device=None):
    """
    int int K(x, y) dmu(x) dmu(y) for mu = N(0, target_var * I_d).
    Monte Carlo estimate: sample x_1..x_m ~ mu, y_1..y_m ~ mu; compute (1/m^2) sum_ij K(x_i, y_j).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cov = target_var * torch.eye(d, device=device)
    dist = torch.distributions.MultivariateNormal(torch.zeros(d, device=device), cov)
    samples_x_md = dist.sample((m,))
    samples_y_md = dist.sample((m,))
    kernel_mm = laplace_kernel(samples_x_md, samples_y_md, sigma)
    return kernel_mm.mean().item()
