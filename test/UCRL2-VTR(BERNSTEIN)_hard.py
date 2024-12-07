import sys
sys.path.append('../')
import os
import random
import numpy as np
import json
from tqdm import tqdm
from env.env import *
from algorithms.ucrl2_vtr_bernstein_hard_gurobi import UCRL2_VTR_BERNSTEIN
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

# Function to run a single experiment
def run_experiment(run, seed, d, D, T, delta, epsilon, resultDir):
    random.seed(seed)
        
    # Create the environment
    env = HardLinearMixtureMDP(d=d, D=D, T=T)        
    
    # Initialize the agent
    agent = UCRL2_VTR_BERNSTEIN(env, T=T, delta=delta, epsilon=epsilon)

    # Run the UCLR2_VTR_BERNSTEIN algorithm
    cumulative_return = agent.run()
    print("seed %d done"%(seed))

    # Compute the regret as the difference between the optimal return and cumulative return
    opt_return = env.run_optimal_policy()
    cumulative_regret = np.array(opt_return) - np.array(cumulative_return)
    print(f"Run {run}, seed {seed} done")

    # Save the results (cumulative_return and cumulative_regret) in a JSON file
    results = {
        'cumulative_return': cumulative_return,
        'cumulative_regret': cumulative_regret.tolist()
    }

    if not os.path.exists(resultDir):
        os.makedirs(resultDir)  # Create the directory if it doesn't exist

    with open(os.path.join(resultDir, f'results45_{run}.json'), 'w') as f:
        json.dump(results, f)

    return cumulative_regret

def main():
    runs = 10  # Adjust this based on the number of parallel runs
    
    for Diam in [120]:
        for t in list(range(1000, 6000, 1000)):
            seeds = [123 * i for i in range(runs)]
            d = 8
            D = Diam
            T = t
            delta = 0.05
            epsilon = 1e-2
            
            resultDir = f"../data/hardtolearn/UCRL2_VTR(BERNSTEIN)(discrete)/D={D}/regret_d_{d}_D_{D:.2f}_T_{T}_delta_{delta}_epsilon_{epsilon}"
            print(f"Run UCRL2_VTR(BERNSTEIN)_d_{d}_D_{D:.2f}_T_{T}_delta_{delta}_epsilon_{epsilon}")
            
            # Use multiprocessing to run experiments in parallel
            pool = mp.Pool(mp.cpu_count())  # Use all available CPUs

            # Use pool.starmap to   distribute the runs in parallel
            results = pool.starmap(run_experiment, [(run, seeds[run], d, D, T, delta, epsilon, resultDir) for run in range(runs)])

            pool.close()  # Close the pool to prevent new tasks from being submitted
            pool.join()  # Wait for all worker processes to finish


    return results

if __name__ == '__main__':
    run_returns = main()
    print("All experiments completed!")