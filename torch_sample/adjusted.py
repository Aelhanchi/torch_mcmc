import torch
from .sampler import Sampler


class MetropolisHastings(Sampler):

    def __init__(self, params, closure):
        super().__init__(params)
        self._save_params_backup()
        self.closure = closure
        self.params_nlp = self.closure()
        self.candidates_nlp = None

    def _save_params_backup(self):
        self.params_backup = list()
        for p in self.params:
            self.params_backup.append(p.clone().detach())

    def _load_params_backup(self):
        with torch.no_grad():
            for p, bp in zip(self.params, self.params_backup):
                p.copy_(bp).detach_().requires_grad_()

    def _clone_list(self, list_of_tensors):
        cloned_list = list()
        for tensor in list_of_tensors:
            cloned_list.append(tensor.clone())
        return cloned_list

    def _propose(self):
        """Generates candidate parameters."""
        return NotImplementedError

    def _calculate_acceptance_ratio(self):
        """Computes the acceptance ratio."""
        return NotImplementedError

    def _decide(self):
        """Decides whether to accept or reject the candidate"""
        return NotImplementedError

    def transition(self):
        self._propose()
        return self._decide()


class MRW(MetropolisHastings):
    """Implements the Metropolis random walk algorithm.

    Args:
        params (iterable): an iterable of Tensors.
            Specifies what parameters should be sampled.
        closure (callable): a closure that computes and returns the negative
            log posterior.
        cov_prop (float, optional): the covariance of the Gaussian
            proposal distribution.
    """

    def __init__(self, params, closure, cov_prop=0.0001):
        super().__init__(params, closure)
        self.chol_cov_prop = torch.tensor(cov_prop).sqrt_()

    def _propose(self):
        """Generates candidate parameters."""
        with torch.no_grad():
            for p in self.params:
                p += self.chol_cov_prop * torch.randn_like(p)

    def _calculate_acceptance_ratio(self):
        """Calculates the acceptance ratio."""
        self.candidates_nlp = self.closure()
        return torch.exp(self.params_nlp - self.candidates_nlp)

    def _decide(self):
        """Decides whether to accept or reject the candidate"""
        if torch.rand(1) <= self._calculate_acceptance_ratio():
            self._save_params_backup()
            self.params_nlp = self.candidates_nlp.clone().detach()
            return (self.params_nlp, True)
        else:
            self._load_params_backup()
            return (self.params_nlp, False)


class MALD(MetropolisHastings):
    """Implements the Metropolis adjusted Langevin dynamics algorithm

    Args:
        params (iterable): an iterable of Tensors.
            Specifies what parameters should be sampled.
        closure (callable): a closure that clears the gradients,
            computes the negative log likelihood and its gradients,
            and returns the negative log likelihood.
        t (float, optional): the integration time of the langevin dynamics.
    """

    def __init__(self, params, closure, t=0.01):
        super().__init__(params, closure)

        self.t = torch.tensor(t)

        self.mean_prop = list()
        self.mean_rev_prop = list()
        with torch.no_grad():
            for p in self.params:
                self.mean_prop.append(p - self.t * p.grad)
                self.mean_rev_prop.append(torch.zeros_like(p))

    def _proposal_density(self, val, mean):
        """Computes the negative log density of the proposal distribution"""
        with torch.no_grad():
            temp = 0
            for v, m in zip(val, mean):
                temp += (1/2) * 1/(2*self.t) * (v - m).pow(2).sum()
            return temp

    def _propose(self):
        """Generates candidate parameters."""
        with torch.no_grad():
            for p in self.params:
                p -= self.t * p.grad + torch.sqrt(2*self.t) * torch.randn_like(p)

    def _calculate_acceptance_ratio(self):
        """Calculates the acceptance ratio."""
        self.candidates_nlp = self.closure()
        with torch.no_grad():
            self.mean_rev_prop = list()
            for p in self.params:
                self.mean_rev_prop.append(p - self.t * p.grad)

            return torch.exp(
                self.params_nlp +
                self._proposal_density(self.params, self.mean_prop) -
                self.candidates_nlp -
                self._proposal_density(self.params_backup, self.mean_rev_prop))

    def _decide(self):
        """Decides whether to accept or reject the candidate"""
        if torch.rand(1) <= self._calculate_acceptance_ratio():
            self._save_params_backup()
            self.params_nlp = self.candidates_nlp.detach().clone()
            for mp, mrp in zip(self.mean_prop, self.mean_rev_prop):
                mp.copy_(mrp)
            return (self.params_nlp, True)
        else:
            self._load_params_backup()
            return (self.params_nlp, False)


class MAHMC(MetropolisHastings):
    """Implements the Metropolis adjusted Hamiltonian dynamics algorithm

    Args:
        params (iterable): an iterable of Tensors.
            Specifies what parameters should be sampled.
        closure (callable): a closure that clear the gradients,
            computes the negative log likelihood and its gradients,
            and returns the negative log likelihood.
        t (float, optional): the integration time of the Hamiltonian dynamics.
        L (int, optional): number of leapfrog steps per integration.
    """

    def __init__(self, params, closure, t=0.01, L=10):
        super().__init__(params, closure)
        self.t = t
        self.L = L

        self.momentums = list()
        for p in self.params:
            self.momentums.append(torch.zeros_like(p))
        self.candidates_momentums = self._clone_list(self.momentums)

        self.zero_grad()

    def _momentum_density(self, momentums):
        """Computes the negative log density of the momentum"""
        temp = 0
        for m in momentums:
            temp += (1/2) * m.pow(2).sum()
        return temp

    def _propose(self):
        """Generates candidate parameters."""
        for i in range(len(self.momentums)):
            self.momentums[i] = torch.randn_like(self.momentums[i])
        self.candidates_momentums = self._clone_list(self.momentums)

        self.candidates_nlp = self.closure()
        for p, m in zip(self.params, self.candidates_momentums):
            m -= (1/2) * (self.t/self.L) * p.grad

        for l in range(self.L):
            with torch.no_grad():
                for p, m in zip(self.params, self.candidates_momentums):
                    p += (self.t/self.L) * m
            self.candidates_nlp = self.closure()
            if (l+1) != self.L:
                for p, m in zip(self.params, self.candidates_momentums):
                    m -= (self.t/self.L) * p.grad

        for p, m in zip(self.params, self.candidates_momentums):
            m -= (1/2) * (self.t/self.L) * p.grad

    def _calculate_acceptance_ratio(self):
        """Calculates the acceptance ratio."""
        return torch.exp(
            self.params_nlp +
            self._momentum_density(self.momentums) -
            self.candidates_nlp -
            self._momentum_density(self.candidates_momentums))

    def _decide(self):
        """Decides whether to accept or reject the candidate"""
        if torch.rand(1) <= self._calculate_acceptance_ratio():
            self._save_params_backup()
            self.params_nlp = self.candidates_nlp.clone().detach()
            return (self.params_nlp, True)
        else:
            self._load_params_backup()
            return (self.params_nlp, False)
