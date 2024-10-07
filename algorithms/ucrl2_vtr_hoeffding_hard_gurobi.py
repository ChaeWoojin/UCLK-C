import sys
sys.path.append('../')
import numpy as np
import itertools
import os
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from env.env import *  # Assuming your environment module is correctly set up

class UCRL2_VTR_HOEFFDING(object):
    def __init__(self, env, T, N, delta, epsilon):
        self.env = env
        self.T = T
        self.d = env.d

        self.theta_star = self.env.theta_tilde
     
        self.B = max(self.env.triangle ** 2 + 1, np.linalg.norm(self.theta_star, ord=2))
        self.lam = 1 / (self.B**2) 
        self.delta = delta
        self.epsilon = epsilon

        self.A = self.lam * np.identity(self.d)
        self.Ainv = np.linalg.inv(self.A)
        self.b = self.lam * np.zeros(self.d)

        self.N = N
        self.phi = env.phi
        self.theta = np.zeros(self.d)
        
    def mixture(self, s, a, u):
        return np.sum(np.array([np.multiply(u[s_], self.phi[(s, a, s_)]) for s_ in range(self.env.nState)]), axis=0)

    def Beta(self, t_k):
        return self.env.D * np.sqrt(self.d * np.log((self.lam + (t_k * (self.env.D ** 2))) / (self.delta * self.lam))) + np.sqrt(self.lam) * self.B

    def EVI(self, t_k):
        cnt = 0
        u = np.zeros(self.env.nState)
        Beta_t = self.Beta(t_k)
        while True:
            u_old = u.copy()
            cnt += 1
            for s in range(self.env.nState):
                max_value = -1e9
                for a in range(self.env.nAction):
                    phi_u = self.mixture(s, a, u)

                    model = gp.Model("optimization")
                    theta = model.addMVar(self.d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")
                    
                    model.setObjective(phi_u @ theta, GRB.MAXIMIZE)
                    
                    model.addConstr((theta - self.theta) @ self.A @ (theta - self.theta) <= Beta_t**2, "C_t")
                    model.addConstr(theta.sum() == 1, "sum_to_one")
                    model.addConstr(theta >= 0, "non_negative")
                    model.addConstr(theta @ theta <= self.B**2, "norm_constraint")

                    model.setParam('OutputFlag', 0)
                    
                    try:
                        model.optimize()
                        if model.status == GRB.OPTIMAL:
                            value = self.env.reward[s,a][0] + model.objVal
                            max_value = max(max_value, value)
                        else:
                            print(f"Optimization failed for state {s}, action {a} at {t_k}. Status: {model.status}")
                    except gp.GurobiError as e:
                        print(f"Gurobi error for state {s}, action {a} at {t_k}: {e}")
                u[s] = max_value

            if cnt == self.N or max(u - u_old) - min(u - u_old) <= self.epsilon:
                break
                
        return u

    def POLICY(self, u_k, t_k):
        pi = {}
        Beta_t = self.Beta(t_k)
        for s in range(self.env.nState):
            q = []
            for a in range(self.env.nAction):
                phi_u = self.mixture(s, a, u_k)

                model = gp.Model("optimization")
                theta = model.addMVar(self.d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")
                
                model.setObjective(phi_u @ theta, GRB.MAXIMIZE)
                
                model.addConstr((theta - self.theta) @ self.A @ (theta - self.theta) <= Beta_t**2, "C_t")
                model.addConstr(theta.sum() == 1, "sum_to_one")
                model.addConstr(theta >= 0, "non_negative")
                model.addConstr(theta @ theta <= self.B**2, "norm_constraint")

                model.setParam('OutputFlag', 0)
                
                try:
                    model.optimize()
                    if model.status == GRB.OPTIMAL:
                        theta_k = theta.X
                        q.append(self.env.reward[s, a][0] + np.dot(theta_k, phi_u))
                    else:
                        print(f"Optimization failed for state {s}, action {a} at {t_k}. Status: {model.status}")
                except gp.GurobiError as e:
                    print(f"Gurobi error for state {s}, action {a} at {t_k}: {e}")
            
            pi[s] = self.env.argmax(np.array(q))
        return pi

    def run(self):
        print('UCRL2_VTR')
        episode_return = []

        A_k = self.A.copy()
        t_k = 1
        w_k = np.ones(self.env.nState)
        R = 0
        for t in tqdm(range(1, self.T + 1)):
            if np.linalg.det(self.A) > 2 * np.linalg.det(A_k):
                t_k = t
                A_k = self.A.copy()

                u_k = self.EVI(t_k)
                pi = self.POLICY(u_k, t_k)
                
                tmp = (max(u_k) - min(u_k)) / 2
                w_k = {s: u_k[s] - tmp for s in range(self.env.nState)}

            s = self.env.state
            if t_k == 1:
                a = np.random.choice([a for a in range(self.env.nAction)])
            else:
                a = pi[s]
                
            r, s_ = self.env.advance(a)
            R += r

            tmp = self.mixture(s, a, w_k)

            self.A += np.outer(tmp, tmp)
            self.Ainv -= np.dot(np.outer(np.dot(self.Ainv, tmp), tmp), self.Ainv) / (1 + np.dot(np.dot(tmp, self.Ainv), tmp))
            self.b += np.multiply(w_k[s_], tmp)

            self.theta = np.dot(self.Ainv, self.b)

            episode_return.append(R)

        return episode_return
        
def evaluate_hyperparameters(args):
    """
    Function to initialize environment and agent, run the algorithm, and store the cumulative regret.
    Args is a tuple containing (d, D, T, delta, epsilon).
    """
    d, D, T, N, delta, epsilon = args
    env = HardLinearMixtureMDP(d=d, D=D, T=T)
    agent = UCRL2_VTR_HOEFFDING(env, T=T, delta=delta, N=N, epsilon=epsilon)  # N is set arbitrarily

    # Run the UCLK_C algorithm
    cumulative_return = agent.run()
    
    # Compute the regret as the difference between the optimal return and cumumlative return
    opt_return = env.run_optimal_policy()
    cumulative_regret = np.array(opt_return) - np.array(cumulative_return)

    results = {
        'cumulative_return': cumulative_return,
        'cumulative_regret': cumulative_regret.tolist()
    }

    # Save cumulative regret to a file for this hyperparameter set
    filename = f"./UCRL2-VTR(HOEFFDING)/N={N}/regret_d_{d}_D_{D:.2f}_T_{T}_N_{N}_delta_{delta}_epsilon_{epsilon}.json"
    with open(filename, 'w') as f:
        json.dump(results, f)

    # Return the final results for comparison
    return {
        'd': d, 'D': D, 'T': T, 'delta': delta, 'epsilon': epsilon,
        'cumulative_return': cumulative_return[-1],
        'opt_return': opt_return[-1],
        'regret_file': filename
    }


def parallel_hyperparameter_search(d_values, D_values, T_values, N_values, delta_values, epsilon_values):
    # Create a list of all hyperparameter combinations
    param_combinations = list(itertools.product(d_values, D_values, T_values, N_values, delta_values, epsilon_values))

    # Use multiprocessing to parallelize the hyperparameter search
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(evaluate_hyperparameters, param_combinations), total=len(param_combinations)))

    return results


def plot_regret(filename):
    """
    Load the cumulative regret from a file and plot it.
    """
    with open(filename) as f:
        data = json.load(f)
        cumulative_regret = data['cumulative_regret']

    # Convert the list back to a numpy array for easier plotting
    cumulative_regret = np.array(cumulative_regret)

    # Plot the cumulative regret
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_regret, label="Cumulative Regret")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret of Best Parameter Set")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Define the hyperparameter ranges
    d_values = [8]  # Dimension 'd'
    
    D_values = np.linspace(50, 150, 6)  # Example range for 'D'
    # D_values = np.linspace(1.5, 3, 10)  # Example range for 'D'

    T_values = [5000]  # Time horizon
    
    N_values = [200, 300]
    
    # delta_values = [0.01]  # Exploration-exploitation trade-off parameter
    delta_values = [0.05]  # Exploration-exploitation trade-off parameter
    
    epsilon_values = [0.000001]  # Precision for stopping the EVI

    # Run the parallel hyperparameter search
    results = parallel_hyperparameter_search(d_values, D_values, T_values, N_values, delta_values, epsilon_values)

    # Find the best hyperparameter combination based on maximum episodic return
    best_result = max(results, key=lambda x: x['cumulative_regret'])

    # # Print out the best result
    print("Best hyperparameters found:")
    print(f"d={best_result['d']}, D={best_result['D']}, T={best_result['T']}, delta={best_result['delta']}, epsilon={best_result['epsilon']}")
    print(f"Episodic return: {best_result['cumulative_return']}, Optimal return: {best_result['opt_return']}")
    print(f"Regret file: {best_result['regret_file']}")

    # Plot the cumulative regret of the best parameter set
    plot_regret(best_result['regret_file'])
