"""An abstract class implementing the reversible method for an abstract explicit Runge--Kutta method.
Provides the base classes for EES methods and McCallum-Foster methods
"""

import torch

from .. import adjoint_sde
from .. import base_solver
from .. import misc
from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS, METHODS

def ERK_step_function(A, b, c, t0, y0, dt, dW, f_and_g, prod):

    s = len(b)
    f_k = [] # f(k_i)
    g_k = [] # g(k_i)

    for i in range(s):
        k0 = sum(A[i][j] * f_k[j] for j in range(i)) * dt
        k1 = sum(A[i][j] * g_k[j] for j in range(i))
        if not isinstance(k1, int): #prod doesn't like when k1 : int = 0
            k1 = prod(k1, dW)
        k = y0 + k0 + k1
        f_k_val, g_k_val = f_and_g(t0 + c[i] * dt, k)
        f_k.append(f_k_val)
        g_k.append(g_k_val)

    t_update = sum(b[i] * f_k[i] for i in range(s))
    W_update = sum(b[i] * g_k[i] for i in range(s))

    y1 = t_update * dt + prod(W_update, dW)
    return y1

def ERK_step_function_backprop(A, b, c, t0, adj_y0, y1, dt, dW, f_and_g, prod, adjoint_of_prod, requires_grad, sde_params, adj_params):

    s = len(b)
    f_k = [] # f(k_i)
    g_k = [] # g(k_i)
    k = []

    for i in range(s):
        # We need to reconstruct k here
        k0 = sum(A[i][j] * f_k[j] for j in range(i)) * dt
        k1 = sum(A[i][j] * g_k[j] for j in range(i))
        if not isinstance(k1, int):  # prod doesn't like when k1 : int = 0
            k1 = prod(k1, dW)
        k_val = y1 + k0 + k1

        with torch.enable_grad():
            if not k_val.requires_grad:
                k_val = k_val.detach().requires_grad_()
            f_k_val, g_k_val = f_and_g(t0 + c[i] * dt, k_val)
            f_k.append(f_k_val)
            g_k.append(g_k_val)

            k.append(k_val)

    dLdk = [None for _ in range(s)]

    for i in range(s-1, -1, -1):
        adj_1 = adj_y0 * b[i] + sum(A[j][i] * dLdk[j] for j in range(i+1, s))
        dLdf = [adj_1 * dt, adjoint_of_prod(adj_1, dW)]

        # Get dLdk and dLdp
        dLdk[i], *dLdp = misc.vjp(outputs=(f_k[i], g_k[i]),
                                      inputs=[k[i]] + sde_params,
                                      grad_outputs=dLdf,
                                      allow_unused=True,
                                      retain_graph=True,
                                      create_graph=requires_grad)

        adj_params = misc.seq_add(adj_params, dLdp)

    adj_y1 = sum(dLdk)

    return adj_y1, adj_params


class ReversibleERK(base_solver.BaseSDESolver):
    weak_order = 1.0
    sde_type = SDE_TYPES.stratonovich
    noise_types = NOISE_TYPES.all()
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, A, b, sde, **kwargs):
        self.strong_order = 1.0 if sde.noise_type == NOISE_TYPES.additive else 0.5
        self.A, self.b = A, b
        self.s = len(b)
        self.c = [sum(A[i][j] for j in range(i)) for i in range(self.s)]
        super(ReversibleERK, self).__init__(sde=sde, **kwargs)

    def step(self, t0, t1, y0, extra0):
        dt = t1 - t0
        dW = self.bm(t0, t1)
        y1 = y0 + ERK_step_function(self.A, self.b, self.c, t0, y0, dt, dW, self.sde.f_and_g, self.sde.prod)
        return y1, tuple()


class AdjointReversibleERK(base_solver.BaseSDESolver):
    weak_order = 1.0
    sde_type = SDE_TYPES.stratonovich
    noise_types = NOISE_TYPES.all()
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, A, b, sde, **kwargs):
        if not isinstance(sde, adjoint_sde.AdjointSDE):
            raise ValueError(f"Adjoint reversible ERK can only be used for adjoint_method.")
        self.strong_order = 1.0 if sde.noise_type == NOISE_TYPES.additive else 0.5
        self.A, self.b = A, b
        self.s = len(b)
        self.c = [sum(A[i][j] for j in range(i)) for i in range(self.s)]
        super(AdjointReversibleERK, self).__init__(sde=sde, **kwargs)
        self.forward_sde = sde.forward_sde

        if self.forward_sde.noise_type == NOISE_TYPES.diagonal:
            self._adjoint_of_prod = lambda tensor1, tensor2: tensor1 * tensor2
        else:
            self._adjoint_of_prod = lambda tensor1, tensor2: tensor1.unsqueeze(-1) * tensor2.unsqueeze(-2)

    def step(self, t0, t1, y0, extra0):
        dt = t1 - t0
        dW = self.bm(t0, t1)

        forward_y0, adj_y0, adj_params, requires_grad = self.sde.get_state(t0, y0, extra_states=True)

        y1 = forward_y0 + ERK_step_function(self.A, self.b, self.c, -t0, forward_y0, -dt, -dW, self.forward_sde.f_and_g, self.forward_sde.prod)

        adj_y1, adj_params = ERK_step_function_backprop(self.A, self.b, self.c, -t1, adj_y0, y1, dt, dW, self.forward_sde.f_and_g, self.forward_sde.prod, self._adjoint_of_prod, requires_grad, self.sde.params, adj_params)
        adj_y1 += adj_y0

        y1 = misc.flatten([y1, adj_y1] + adj_params).unsqueeze(0)
        return y1, tuple()

class MCFReversibleERK(base_solver.BaseSDESolver):
    """
    McCallum-Foster method applied to an explicit RK method
    """
    weak_order = 1.0
    sde_type = SDE_TYPES.stratonovich
    noise_types = NOISE_TYPES.all()
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def init_extra_solver_state(self, t0, y0):
        return (y0,)

    def __init__(self, lam, A, b, sde, **kwargs):
        self.strong_order = 1.0 if sde.noise_type == NOISE_TYPES.additive else 0.5
        self.lam = lam
        self.A, self.b = A, b
        self.s = len(b)
        self.c = [sum(A[i][j] for j in range(i)) for i in range(self.s)]
        super(MCFReversibleERK, self).__init__(sde=sde, **kwargs)

    def step(self, t0, t1, y0, extra0):
        z0 = extra0[0]
        dt = t1 - t0
        dW = self.bm(t0, t1)
        y1 = self.lam * (y0 - z0) + z0 + ERK_step_function(self.A, self.b, self.c, t0, z0, dt, dW, self.sde.f_and_g, self.sde.prod)
        z1 = z0 - ERK_step_function(self.A, self.b, self.c, t1, y1, -dt, -dW, self.sde.f_and_g, self.sde.prod)
        return y1, (z1,)


class AdjointMCFReversibleERK(base_solver.BaseSDESolver):
    """
    McCallum-Foster method applied to an explicit RK method
    """
    weak_order = 1.0
    sde_type = SDE_TYPES.stratonovich
    noise_types = NOISE_TYPES.all()
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, lam, A, b, sde, **kwargs):
        if not isinstance(sde, adjoint_sde.AdjointSDE):
            raise ValueError(f"Adjoint reversible ERK can only be used for adjoint_method.")
        self.strong_order = 1.0 if sde.noise_type == NOISE_TYPES.additive else 0.5
        self.lam = lam
        self.lam_inverse = 1. / self.lam
        self.A, self.b = A, b
        self.s = len(b)
        self.c = [sum(A[i][j] for j in range(i)) for i in range(self.s)]
        super(AdjointMCFReversibleERK, self).__init__(sde=sde, **kwargs)
        self.forward_sde = sde.forward_sde

        if self.forward_sde.noise_type == NOISE_TYPES.diagonal:
            self._adjoint_of_prod = lambda tensor1, tensor2: tensor1 * tensor2
        else:
            self._adjoint_of_prod = lambda tensor1, tensor2: tensor1.unsqueeze(-1) * tensor2.unsqueeze(-2)

    def init_extra_solver_state(self, t0, y0):
        # We expect to always be given the extra state from the forward pass.
        raise RuntimeError("Please report a bug to torchsde.")

    def step(self, t0, t1, y0, extra0):
        # See Algorithm 1 in https://arxiv.org/pdf/2410.11648
        z0 = extra0[0]
        dt = t1 - t0
        dW = self.bm(t0, t1)

        forward_y0, adj_y0, (adj_z0, *adj_params), requires_grad = self.sde.get_state(t0, y0, extra_states=True)

        z1 = z0 + ERK_step_function(self.A, self.b, self.c, -t0, forward_y0, -dt, -dW, self.forward_sde.f_and_g, self.forward_sde.prod)
        y1 = z1 + self.lam_inverse * (forward_y0 - z1 - ERK_step_function(self.A, self.b, self.c, -t1, z1, dt, dW, self.forward_sde.f_and_g, self.forward_sde.prod))
        dz0_dy0, adj_params = ERK_step_function_backprop(self.A, self.b, self.c, -t0, -adj_z0, forward_y0, -dt, -dW,
                                                        self.forward_sde.f_and_g, self.forward_sde.prod,
                                                        self._adjoint_of_prod, requires_grad, self.sde.params,
                                                        adj_params)

        adj_y0 += dz0_dy0

        dy0_dz1, adj_params = ERK_step_function_backprop(self.A, self.b, self.c, -t1, adj_y0, z1, dt, dW,
                                                        self.forward_sde.f_and_g, self.forward_sde.prod,
                                                        self._adjoint_of_prod, requires_grad, self.sde.params,
                                                        adj_params)

        adj_y1 = self.lam * adj_y0
        adj_z1 = adj_z0 + (1. - self.lam) * adj_y0 + dy0_dz1

        y1 = misc.flatten([y1, adj_y1, adj_z1] + adj_params).unsqueeze(0)
        return y1, (z1,)
