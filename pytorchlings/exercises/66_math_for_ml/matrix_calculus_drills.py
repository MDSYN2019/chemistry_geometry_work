"""Exercise 66: math refresh drills used directly in ML training code.

Goal:
- practice vectorized linear algebra primitives (SVD, projection, quadratic form)
- compute gradients/Jacobians/Hessians with autograd
- implement a negative log-likelihood objective
"""

import torch


def projection_matrix(basis: torch.Tensor) -> torch.Tensor:
    """Return projection matrix P = Q Q^T for full-rank basis columns.

    basis: [d, k] matrix with k <= d.
    """
    # TODO: use QR decomposition to get an orthonormal Q, then return Q @ Q.T
    q, _ = torch.linalg.qr(basis)
    return q @ q.T


def quadratic_form(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Compute scalar x^T A x for x:[d], A:[d,d]."""
    # TODO: return a scalar tensor representing x^T A x
    return x @ a @ x


def gaussian_nll(x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Diagonal Gaussian negative log-likelihood (mean over batch)."""
    # TODO: implement NLL using exp(log_var) as variance
    var = torch.exp(log_var)
    nll = 0.5 * (torch.log(2 * torch.pi * var) + (x - mu).pow(2) / var)
    return nll.mean()


def jacobian_of_linear_map(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian of f(x)=Wx wrt x using autograd.functional.jacobian."""

    def fn(inp: torch.Tensor) -> torch.Tensor:
        return w @ inp

    # TODO: compute and return Jacobian J with shape [out_dim, in_dim]
    return torch.autograd.functional.jacobian(fn, x)


if __name__ == "__main__":
    torch.manual_seed(0)

    basis = torch.randn(5, 2)
    p = projection_matrix(basis)
    assert p.shape == (5, 5)

    x = torch.randn(4, requires_grad=True)
    a = torch.randn(4, 4)
    a = 0.5 * (a + a.T)
    q = quadratic_form(x, a)
    q.backward()
    assert x.grad is not None

    xb = torch.randn(8, 3)
    mu = torch.randn(8, 3, requires_grad=True)
    lv = torch.zeros(8, 3, requires_grad=True)
    loss = gaussian_nll(xb, mu, lv)
    loss.backward()
    assert mu.grad is not None and lv.grad is not None

    w = torch.randn(3, 4)
    xin = torch.randn(4)
    j = jacobian_of_linear_map(w, xin)
    assert j.shape == (3, 4)
    print("exercise 66 smoke check passed")
