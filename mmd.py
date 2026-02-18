def mmd_energy(particles_nd, kernel_fn, closed_form_integral_fn, constant_term=0.0):
  # Args:
  #     particles_nd: (n, d) tensor, current particle positions
  #     kernel_fn: fn(x_nd, y_md) -> (n, m), function that takes in x, y and returns a kernel matrix with (x_i, y_i) at each entry
  #     closed_form_integral_fn: fn(particles_nd) -> (n,) tensor, function that computes \int{R^d} K(x,y)d\mu(y) in closed form
  #     constant_term: scalar, precomputed \int_{R^d} \int_{R^d} K(x,y) dmu(x) dmu(y)

  #   Returns: scalar loss

    if not closed_form_integral_fn:
      raise ValueError("MMD energy computation expected a function that compute the closed form integral, got None")

    n, _ = particles_nd.shape

    term1 = (-2 / n) * closed_form_integral_fn(particles_nd).sum()

    K_nn = kernel_fn(particles_nd, particles_nd)

    term2 = K_nn.sum() / (n ** 2)

    return constant_term + term1 + term2