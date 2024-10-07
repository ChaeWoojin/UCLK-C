import sys
sys.path.append('../')
import os
import random
import numpy as np
import json
from tqdm import tqdm
from env.env import *
from algorithms.uclk_c_hard_gurobi import UCLK_C
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

# Function to run a single experiment
def run_experiment(run, seed, d, D, T, N, delta, epsilon, resultDir):
    random.seed(seed)
    
    # Create the environment
    env = HardLinearMixtureMDP(d=d, D=D, T=T)
    
    # Initialize the agent
    agent = UCLK_C(env, T=T, delta=delta, N=N, epsilon=epsilon)

    # Run the UCLK_C algorithm
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

    with open(os.path.join(resultDir, f'results_run{run}.json'), 'w') as f:
        json.dump(results, f)

    return cumulative_regret

def main():
    runs = 10  # Adjust this based on the number of parallel runs
    seeds = [123 * i for i in range(runs)]
    d = 8
    D = 
    T = 10000
    N = 200
    delta = 0.05
    epsilon = 0.000001

    resultDir = f"../data/hardtolearn/UCLK_C/regret_d_{d}_D_{D:.2f}_T_{T}_N_{N}_delta_{delta}_epsilon_{epsilon}"
    
    # Use multiprocessing to run experiments in parallel
    pool = mp.Pool(mp.cpu_count())  # Use all available CPUs

    # Use pool.starmap to distribute the runs in parallel
    results = pool.starmap(run_experiment, [(run, seeds[run], d, D, T, N, delta, epsilon, resultDir) for run in range(runs)])

    pool.close()  # Close the pool to prevent new tasks from being submitted
    pool.join()  # Wait for all worker processes to finish

    # Plotting the cumulative returns
    episodes = np.arange(T)
    plt.figure()

    data_mean = np.mean(results, axis=0)
    data_std = np.std(results, axis=0)
        
    plt.fill_between(episodes, data_mean + data_std, data_mean - data_std, alpha=0.2)
    plt.plot(episodes, data_mean, linewidth=1.8)
    plt.title("Hard to Learn, T=5000")
    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Regret")
    plt.show()

    return results

if __name__ == '__main__':
    run_returns = main()
    print("All experiments completed!")
