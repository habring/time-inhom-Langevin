# Convergent Annealing Schemes

This repository contains the source code to reproduce the results of the paper "Forward-KL Convergence of Time-Inhomogeneous Langevin Diffusions".

The files `experiments_gmm_kD.py` with $k=1,2,n$ contain code to perform sampling for the different experiments and the files `experiments_gmm_kD_plots.py` generate the figures, respectively csv files for the graphs in the paper. The file `experiment_1D_unif_steps.py` generates samples where for all paths the same step size is used. Finally, `steps.py` generates the plots where step sizes are compared.
