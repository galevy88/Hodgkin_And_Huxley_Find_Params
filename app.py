
import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
import Globals
import loss_functions as loss
import pandas as pd

# # Problem Definition
# problem = structure()
# problem.costfunc = [loss.l2_loss, loss.l1_loss]
# problem.nvar = 8
# problem.varmin = Globals.medium_varmin
# problem.varmax = Globals.medium_varmax
# problem.update_vec = Globals.easy_gamma
# problem.batch_size = [64, 128, 1024]

# # GA Parameters
# params = structure()
# params.maxit = [100, 200, 400]
# params.npop = [50, 100, 150]
# params.beta = [0.6, 0.8, 1]
# params.pc = [1 , 2, 3]
# params.mu = [0.1, 0.2, 0.4]
# params.sigma = [Globals.easy_sigma, Globals.medium_sigma, Globals.extreme_sigma]

# # Run GA
# out = ga.run(problem, params)
# print(out.bestsol)

# # Results
# plt.plot(out.bestcost)
# plt.xlim(0, params.maxit)
# plt.xlabel('Iterations')
# plt.ylabel('Best Cost')
# plt.title('Genetic Algorithm (GA)')
# plt.grid(True)
# plt.show()


costfuncs = [loss.l2_loss, loss.l1_loss]
batch_sizes = [512, 1024]
maxits = [1000]
npops = [70]
betas = [0, 1, 10]
pcs = [1, 2]
mus = [0.2, 0.4, 0.8]
sigmas = [Globals.easy_sigma, Globals.medium_sigma, Globals.extreme_sigma]


df = pd.DataFrame(columns=['CostFunc', 'BatchSize', 'MaxIt', 'NPop', 'Beta', 'PC', 'Mu', 'Sigma', 'BestSol1', 'BestSol2', 'BestSol3', 'BestSol4', 'BestSol5', 'AllCost', 'AvgCost'])

num_combinations = len(costfuncs) * len(batch_sizes) * len(maxits) * len(npops) * len(betas) * len(pcs) * len(mus) * len(sigmas)
counter = 1

for costfunc in costfuncs:
    for batch_size in batch_sizes:
        for maxit in maxits:
            for npop in npops:
                for beta in betas:
                    for pc in pcs:
                        for mu in mus:
                            for sigma in sigmas:
                                
                                print(f"{counter}/{num_combinations}")
                                
                                problem = structure()
                                problem.costfunc = costfunc
                                problem.nvar = 8
                                problem.varmin = Globals.medium_varmin
                                problem.varmax = Globals.medium_varmax
                                problem.update_vec = Globals.easy_gamma
                                problem.batch_size = batch_size

                                params = structure()
                                params.maxit = maxit
                                params.npop = npop
                                params.beta = beta
                                params.pc = pc
                                params.mu = mu
                                params.sigma = sigma

                                out = ga.run(problem, params)
                                
                                row = {
                                    'CostFunc': costfunc.__name__,
                                    'BatchSize': batch_size,
                                    'MaxIt': maxit,
                                    'NPop': npop,
                                    'Beta': beta,
                                    'PC': pc,
                                    'Mu': mu,
                                    'Sigma': str(sigma),
                                    'BestSol1': str(out.top_5[0][0]),
                                    'BestSol2': str(out.top_5[0][1]),
                                    'BestSol3': str(out.top_5[0][2]),
                                    'BestSol4': str(out.top_5[0][3]),
                                    'BestSol5': str(out.top_5[0][4]),
                                    'AllCost': str(out.top_5[1]),
                                    'AvgCost': str(sum(out.bestcost)/len(out.bestcost))
                                }
                                df = df.append(row, ignore_index=True)
                                df.to_csv('my_dataframe.csv', index=False)
                                
                                counter+=1



