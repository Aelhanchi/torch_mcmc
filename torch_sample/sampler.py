class Sampler():

    def __init__(self, params):
        self.params = list(params)

    def transition(self, closure):
        """Generates the next element of the Markov chain.

        Args:
            closure (callable): A closure that computes and returns the
                negative log posterior at the current parameters.
        """
        raise NotImplementedError

    def zero_grad(self):
        """Clears the gradients of all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
