import os
import sys
sys.path.append('../')
import random
import numpy as np
from tqdm import tqdm
from env.env import HardLinearMixtureMDP
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

# Function to run a single experiment
def run_experiment(run, seed, d, D, T, resultDir):
    random.seed(seed)
    
    # Create the environment
    env = HardLinearMixtureMDP(d=d, D=D, T=T)
    
    # Run the agent
    returns = env.run_optimal_policy()
    print("seed %d done"%(seed))

    # Save the result
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)  # Create the directory if it doesn't exist
    np.save(resultDir + f'/return{run}.npy', returns)

    return returns

def main():
    runs = 10  # Adjust this based on the number of parallel runs
    seeds = [123 * (i + 1) for i in range(runs)]
    d = 8
    D = 5
    T = 500

    resultDir = '../data/hardtolearn/T=' + str(T) + '/Optimal Policy'
    
    # Use multiprocessing to run experiments in parallel
    pool = mp.Pool(mp.cpu_count())  # Use all available CPUs

    # Use pool.starmap to distribute the runs in parallel
    results = pool.starmap(run_experiment, [(run, seeds[run], d, D, T, resultDir) for run in range(runs)])

    pool.close()  # Close the pool to prevent new tasks from being submitted
    pool.join()  # Wait for all worker processes to finish

    episodes = np.arange(T)

    plt.figure()
    data_mean = np.mean(results, axis=0)
    data_std = np.std(results, axis=0)
        
    plt.fill_between(episodes, data_mean + data_std, data_mean - data_std, alpha=0.2)
    plt.plot(episodes, data_mean, linewidth=1.8)
    plt.title("Hard to Learn, T=500")
    plt.xlabel("Timesteps")

    plt.ylabel("Cumulative Returns")
    plt.show()

    return results

if __name__ == '__main__':
    run_returns = main()
    print("All experiments completed!")