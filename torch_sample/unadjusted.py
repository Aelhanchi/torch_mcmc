import torch
from .sampler import Sampler


class LD(Sampler):

    """Implements the Langevin dynamics MCMC algorithm.

    Arguments:
        params (iterable): the parameters of the model.
        t (float): the integration time of the Langevin dynamics.
    """

    def __init__(self, params, t=0.01):
        super().__init__(params)
        self.t = torch.tensor(t)

    def transition(self, closure=None):
        """Runs one iteration of the Langevin dynamics algorithm.

        Arguments:
            closure (callable, optional): A closure that clears the gradients,
            computes the negative log posterior and its gradient,
            and returns the negative log posterior.
            Example:
                def closure():
                    model.zero_grad()
                    output = model(input)
                    nlp = nlp_func(output, target)
                    nlp.backward()
                    return nlp
        """
        nlp = None
        if closure is not None:
            nlp = closure()

        with torch.no_grad():
            for p in self.params:
                p -= (self.t * p.grad -
                      torch.sqrt(2 * self.t) * torch.randn_like(p.grad))

        return nlp


class ULD(Sampler):
    """Implements the underdamped Langevin dynamics MCMC algorithm.

    Arguments:
        params (iterable): the parameters of the model.
        t (float): the integration time of the Langevin dynamics.
        gam (float): parameter of the algorithm. see reference.
    """

    def __init__(self, params, t=0.01, gam=2.0):
        super().__init__(params)
        self.momentums = list()
        for p in self.params:
            self.momentums.append(torch.zeros_like(p))

        self.t = t
        self.gam = gam

        self.c_1 = torch.exp(-self.gam*self.t)
        self.c_2 = (1 - self.c_1)/self.gam
        t = torch.tensor(t, dtype=torch.float64)
        gam = torch.tensor(gam, dtype=torch.float64)
        cov = torch.tensor(
            [[2*(t*gam + 2*torch.exp(-gam*t) - 0.5*torch.exp(-2*gam*t) - 3/2)/gam**2,
              (1 + torch.exp(-2*gam*t) - 2*torch.exp(-gam*t))/gam],
             [(1 + torch.exp(-2*gam*t) - 2*torch.exp(-gam*t))/gam,
              1 - torch.exp(-2*gam*t)]])
        self.chol_cov = torch.cholesky(cov).to(torch.float32)

    def transition(self, closure=None):
        """Runs one iteration of the underdamped Langevin dynamics algorithm.

        Arguments:
            closure (callable, optional): A closure that clears the gradients,
            computes the negative log posterior and its gradient,
            and returns the negative log posterior.
            Example:
                def closure():
                    model.zero_grad()
                    output = model(input)
                    nlp = nlp_func(output, target)
                    nlp.backward()
                    return nlp
        """
        nlp = None
        if closure is not None:
            nlp = closure()

        with torch.no_grad():
            for m, p in zip(self.momentums, self.params):
                noise = (self.chol_cov @ torch.randn(2, p.nelement())).reshape([2] + list(p.shape))
                p += self.c_2 * m - (self.t - self.c_2) * p.grad + noise[0]
                m *= self.c_1
                m -= self.c_2 * p.grad + noise[1]

        return nlp


class HMC(Sampler):
    """Implements the Hamiltonian Monte Carlo algorithm

    Arguments:
        params (iterable): the parameters of the model.
        t (float): the integration time of the Hamiltonian dynamics.
        L (int): number of leapfrog steps.
    """

    def __init__(self, params, t=0.1, L=10):
        super().__init__(params)
        self.momentums = list()
        for p in self.params:
            self.momentums.append(torch.zeros_like(p))

        self.t = t
        self.L = L

    def transition(self, closure):
        """Runs one iteration of the Hamiltonian Monte Carlo algorithm.

        Arguments:
            closure (callable): A closure that clears the gradients,
            computes the negative log posterior and its gradient,
            and returns the negative log posterior.
            Example:
                def closure():
                    model.zero_grad()
                    output = model(input)
                    nlp = nlp_func(output, target)
                    nlp.backward()
                    return nlp
        """
        nlp = closure()

        for i in range(len(self.momentums)):
            self.momentums[i] = torch.randn_like(self.momentums[i])

        for m, p in zip(self.momentums, self.params):
            m -= (1/2) * (self.t/self.L) * p.grad

        for l in range(self.L):
            with torch.no_grad():
                for m, p in zip(self.momentums, self.params):
                    p += (self.t/self.L) * m
            if (l+1) != self.L:
                nlp = closure()
                for m, p in zip(self.momentums, self.params):
                    m -= (self.t/self.L) * p.grad

        return nlp
