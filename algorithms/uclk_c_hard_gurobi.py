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

class UCLK_C(object):
    def __init__(self, env, T, delta, N, epsilon):
        self.env = env
        self.T = T
        self.d = env.d
        self.gamma = 1 - np.sqrt(self.d / (self.env.H * self.T))

        self.theta_star = self.env.theta_tilde
        
        self.A_hat = np.identity(self.d)
        self.A_til = np.identity(self.d)
        self.Ainv_hat = np.linalg.inv(self.A_hat)
        self.Ainv_til = np.linalg.inv(self.A_til)
        self.b_hat = np.zeros(self.d)
        self.b_til = np.zeros(self.d)

        self.theta_hat = np.zeros(self.d)
        self.theta_til = np.zeros(self.d)
        
        self.B = max(self.env.triangle ** 2 + 1, np.linalg.norm(self.theta_star, ord=2))
        self.lam = 1 / (self.B**2) 
        self.delta = delta
        self.N = N
        self.epsilon = epsilon

        self.phi = env.phi
        
    def mixture(self, s, a, u):
        return np.sum(np.array([np.multiply(u[s_], self.phi[(s, a, s_)]) for s_ in range(self.env.nState)]), axis=0)

    def Beta(self, t_k):
        return 8 * np.sqrt(self.d * np.log(1 + t_k / self.lam) * np.log(4 * t_k**2 / self.delta)) + 4 * np.sqrt(self.d) * np.log(4 * t_k**2 / self.delta) + np.sqrt(self.lam) * self.B

    def EVI(self, t_k):
        cnt = 0
        v = np.ones(self.env.nState) * 1 / (1 - self.gamma)
        q = np.ones((self.env.nState, self.env.nAction)) * 1 / (1 - self.gamma)
        Beta_t = self.Beta(t_k)
        while True:
            v_old = v.copy()
            cnt += 1
            for s in range(self.env.nState):
                for a in range(self.env.nAction):
                    phi_v = self.mixture(s, a, v)

                    model = gp.Model("optimization")
                    theta = model.addMVar(self.d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")
                    
                    model.setObjective(phi_v @ theta, GRB.MAXIMIZE)
                    
                    model.addConstr((theta - self.theta_hat) @ self.A_hat @ (theta - self.theta_hat) <= Beta_t**2, "C_t")
                    model.addConstr(theta.sum() == 1, "sum_to_one")
                    model.addConstr(theta >= 0, "non_negative")
                    model.addConstr(theta @ theta <= self.B**2, "norm_constraint")

                    model.setParam('OutputFlag', 0)
                    
                    try:
                        model.optimize()
                        if model.status == GRB.OPTIMAL:
                            q[s,a] = self.env.reward[s,a][0] + model.objVal
                        else:
                            print(f"Optimization failed for state {s}, action {a} at {t_k}. Status: {model.status}")
                    except gp.GurobiError as e:
                        print(f"Gurobi error for state {s}, action {a} at {t_k}: {e}")
            
            for s in range(self.env.nState):
                v[s] = max(q[s,:])
            
            for s in range(self.env.nState):
                v[s] = min(v[s], min(v) + self.env.H)

            if cnt == self.N or np.max(v - v_old) - np.min(v - v_old) <= self.epsilon:
                break
                
        return q, v

    def POLICY(self, q):
        pi = {s: 0 for s in range(self.env.nState)}
        for s in range(self.env.nState):
            pi[s] = np.argmax(q[(s,a)] for a in range(self.env.nAction))
        return pi

    def VARIANCE(self, s, a, w, t):
        phi_w2 = self.mixture(s, a, np.square(w))
        phi_w = self.mixture(s, a, w)
        
        Beta_check = 8 * self.d * np.sqrt(np.log(1 + t / self.lam) * np.log(4 * t**2 / self.delta)) \
                    + 4 * np.sqrt(self.d) * np.log(4 * t**2 / self.delta) + np.sqrt(self.lam) * self.B
        Beta_til = 8 * self.env.H**2 * np.sqrt(self.d * np.log(1 + t * self.env.H**2 / (self.d * self.lam)) * np.log(4 * t**2 / self.delta)) \
                    + 4 * self.env.H**2 * np.log(4 * t**2 / self.delta) + np.sqrt(self.lam) * self.B
        
        VW = np.clip(np.dot(phi_w2, self.theta_til), 0, self.env.H**2) - np.clip(np.dot(phi_w, self.theta_hat), 0, self.env.H)**2
        E = min(self.env.H**2, 2 * self.env.H * Beta_check * np.sqrt(np.dot(np.dot(phi_w, self.Ainv_hat), phi_w))) \
            + min(self.env.H**2, Beta_til * np.sqrt(np.dot(np.dot(phi_w2, self.Ainv_til), phi_w2)))

        return VW + E

    def run(self):
        print('UCLK_C')
        cumulative_return = []

        A_hat_k = self.A_hat.copy()
        A_til_k = self.A_til.copy()
        
        t_k = 1
        q_k, v_k = self.EVI(t_k)
        w_k = v_k - np.min(v_k) 
        pi = self.POLICY(q_k)
        
        R = 0
        for t in tqdm(range(1, self.T + 1)):
            if np.linalg.det(self.A_hat) > 2 * np.linalg.det(A_hat_k):
                A_hat_k = self.A_hat.copy()

                t_k = t
                q_k, v_k = self.EVI(t_k)
                w_k = v_k - np.min(v_k) 
                pi = self.POLICY(q_k)

            s = self.env.state
            a = pi[s]
                
            r, s_ = self.env.advance(a)
            R += r
            
            sig = np.sqrt(max(self.env.H**2 / self.d, self.VARIANCE(s, a, w_k, t)))

            phi_w = self.mixture(s, a, w_k)
            phi_w2 = self.mixture(s, a, np.square(w_k))

            self.A_hat += 1/(sig**2) * np.outer(phi_w, phi_w) 
            self.b_hat += 1/(sig**2) * np.multiply(w_k[s_], phi_w)
            self.A_til += np.outer(phi_w2, phi_w2)
            self.b_til += np.multiply(w_k[s_]**2, phi_w2)


            self.Ainv_hat -= np.dot(np.outer(np.dot(self.Ainv_hat, phi_w/(sig**2)), phi_w), self.Ainv_hat) / (1 + np.dot(np.dot(phi_w/(sig**2), self.Ainv_hat), phi_w))
            self.Ainv_til -= np.dot(np.outer(np.dot(self.Ainv_til, phi_w2), phi_w2), self.Ainv_til) / (1 + np.dot(np.dot(phi_w2, self.Ainv_til), phi_w2))
            
            self.theta_hat = np.dot(self.Ainv_hat, self.b_hat)
            self.theta_til = np.dot(self.Ainv_til, self.b_til)

            cumulative_return.append(R)

        return cumulative_return
    
    

def evaluate_hyperparameters(args):
    """
    Function to initialize environment and agent, run the algorithm, and store the cumulative regret.
    Args is a tuple containing (d, D, T, delta, epsilon).
    """
    d, D, T, delta, epsilon = args
    env = HardLinearMixtureMDP(d=d, D=D, T=T)
    agent = UCLK_C(env, T=T, delta=delta, N=100, epsilon=epsilon)  # N is set arbitrarily

    # Run the UCLK_C algorithm
    episodic_return = agent.run()
    
    # Compute the regret as the difference between the optimal return and episodic return
    opt_return = env.run_optimal_policy()
    cumulative_regret = np.array(opt_return) - np.array(episodic_return)

    # Save cumulative regret to a file for this hyperparameter set
    filename = f"regret_d_{d}_D_{D:.2f}_T_{T}_delta_{delta}_epsilon_{epsilon}.json"
    with open(filename, 'w') as f:
        json.dump(cumulative_regret.tolist(), f)

    # Return the final results for comparison
    return {
        'd': d, 'D': D, 'T': T, 'delta': delta, 'epsilon': epsilon,
        'episodic_return': episodic_return[-1],
        'opt_return': opt_return[-1],
        'regret_file': filename
    }


def parallel_hyperparameter_search(d_values, D_values, T_values, delta_values, epsilon_values):
    # Create a list of all hyperparameter combinations
    param_combinations = list(itertools.product(d_values, D_values, T_values, delta_values, epsilon_values))

    # Use multiprocessing to parallelize the hyperparameter search
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(evaluate_hyperparameters, param_combinations), total=len(param_combinations)))

    return results


def plot_regret(filename):
    """
    Load the cumulative regret from a file and plot it.
    """
    with open(filename, 'r') as f:
        cumulative_regret = json.load(f)

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
    
    D_values = np.linspace(10, 100, 9)  # Example range for 'D'
    # D_values = np.linspace(1.5, 3, 10)  # Example range for 'D'

    T_values = [500]  # Time horizon
    
    # delta_values = [0.01]  # Exploration-exploitation trade-off parameter
    delta_values = [0.01, 0.05, 0.25]  # Exploration-exploitation trade-off parameter
    
    epsilon_values = [0.0001]  # Precision for stopping the EVI

    # Run the parallel hyperparameter search
    results = parallel_hyperparameter_search(d_values, D_values, T_values, delta_values, epsilon_values)

    # Find the best hyperparameter combination based on maximum episodic return
    best_result = max(results, key=lambda x: x['episodic_return'])

    # Print out the best result
    print("Best hyperparameters found:")
    print(f"d={best_result['d']}, D={best_result['D']}, T={best_result['T']}, delta={best_result['delta']}, epsilon={best_result['epsilon']}")
    print(f"Episodic return: {best_result['episodic_return']}, Optimal return: {best_result['opt_return']}")
    print(f"Regret file: {best_result['regret_file']}")

    # Plot the cumulative regret of the best parameter set
    plot_regret(best_result['regret_file'])