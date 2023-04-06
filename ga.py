import numpy as np
import ga_functions
from ypstruct import structure
from tqdm import tqdm

def run(problem, params):
    
    # Problem Information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax
    update_vec = problem.update_vec
    batch_size = problem.batch_size
    

    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc = params.pc
    nc = int(np.round(pc*npop/2)*2)
    mu = params.mu
    sigma = params.sigma

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf

    
    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfunc(pop[i].position, batch_size)
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()

    # Best Cost of Iterations
    bestcost = np.empty(maxit)
    
    # Main Loop
    for it in tqdm(range(maxit)):
        costs = np.array([x.cost for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs/avg_cost
        probs = np.exp(-beta*costs)

        popc = []
        for _ in range(nc//2):
            p1 = pop[ga_functions.roulette_wheel_selection(probs)]
            p2 = pop[ga_functions.roulette_wheel_selection(probs)]
            
            c1, c2 = ga_functions.crossover(p1, p2, update_vec)

            c1 = ga_functions.mutate(c1, mu, sigma)
            c2 = ga_functions.mutate(c2, mu, sigma)

            ga_functions.apply_bound(c1, varmin, varmax)
            ga_functions.apply_bound(c2, varmin, varmax)

            c1.cost = costfunc(c1.position, batch_size)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()

            c2.cost = costfunc(c2.position, batch_size)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy()

            popc.append(c1)
            popc.append(c2)
        
        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:npop]
        # Store Best Cost
        bestcost[it] = bestsol.cost

        # Show Iteration Information
        # ga_functions.print_top_5(bestsol, pop, it)

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.top_5 = ga_functions.get_top_5(bestsol, pop)
    out.bestcost = bestcost
    return out

