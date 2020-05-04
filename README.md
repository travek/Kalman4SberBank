# Kalman4SberBank
Optimization of UKF (Kalman) for Sberbank equity

SberBank equities (stocks) are actively traded on Moscow Exchange. Sber is the most liquid equity on MOEX.

Here we are trying to use genetic programing (using DEAP) to determin best parameters of UKF filter that is used for next step price prediction and price filtering. We are optimizing Direction Symmetry (DS) fitness - maximizing number of correct direction of next step price movements.

This SBER price filtering and optimization is done using DEAP and FilterPy packages.
DEAP is used with multiprocessing feature.

You need to create ./arch directory to store different verions of .py source file. Each time you run .py file source code is copied into arch directory
You need to create ./results directory. Statistics of optimization are saved into separate file in ./results directory


