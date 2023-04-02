
import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
import Globals
import loss_functions as loss

# Problem Definition
problem = structure()
problem.costfunc = loss.l2_loss
problem.nvar = 8
problem.varmin = Globals.medium_varmin
problem.varmax = Globals.medium_varmax
problem.update_vec = Globals.easy_gamma

# GA Parameters
params = structure()
params.maxit = 1000
params.npop = 100
params.beta = 1
params.pc = 5
params.mu = 0.2
params.sigma = Globals.sigma

# Run GA
out = ga.run(problem, params)
print(out.bestsol)

# Results
plt.plot(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()

