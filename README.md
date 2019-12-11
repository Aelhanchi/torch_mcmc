# torch_sample

This is an ongoing implementation of variance reduced stochastic gradient-based
optimization and Markov chain Monte Carlo algorithms in PyTorch.
It supports SAGA for variance reduction, as well as the most popular basic MCMC
algorithms such as random walk Metropolis,
HMC and MALA, as well as stochastic versions of them that use subsamples
of the data at each iteration.
