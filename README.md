# Ornstein-Uhlenbeck process + DRAM MCMC

This project was an assignment in the course Statistical Parameter Estimation. In the assignment I estimated parameters from the Ornstein-Uhlenbeck process with Delayed Rejection Adaptive Markov Chain Monte Carlo algorithm. The process can be expressed as follows
```math
dX_{t} = -\theta X_{t} dt + \sigma dW_{t},
```
where $\theta$ and $\sigma$ are the parameters to estimate. $W_{t}$ is a Wiener process.

In the "Ornstein-Uhlenbeck_MCMC.pdf" you can see the written report of the assignment where I have presented all the results of the estimations. The MATLAB code is in a file "Ornstein_Uhlenbeck_MCMC.m".
