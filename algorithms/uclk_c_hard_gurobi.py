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
        
        self.B = max(self.env.triangle ** 2 + 1, np.linalg.norm(self.theta_star, ord=2))
        self.lam = 1 / (self.B**2) 
        self.delta = delta
        self.N = N
        self.epsilon = epsilon
        
        self.A_hat = self.lam * np.identity(self.d)
        self.A_til = self.lam * np.identity(self.d)
        self.Ainv_hat = np.linalg.inv(self.A_hat)
        self.Ainv_til = np.linalg.inv(self.A_til)
        self.b_hat = np.zeros(self.d)
        self.b_til = np.zeros(self.d)

        self.theta_hat = np.zeros(self.d)
        self.theta_til = np.zeros(self.d)

        self.phi = self.env.phi
        
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
                    model.addConstr(theta @ theta <= self.B**2, "norm_constraint")

                    # # Add non-negativity constraints for transition probabilities
                    # for s in range(self.env.nState):
                    #     for a in range(self.env.nAction):
                    #         for s_ in range(self.env.nState):
                    #             model.addConstr(self.phi[(s, a, s_)] @ theta >= 0, name=f"probability_nonneg_{s}_{a}_{s_}")

                    # # Add constraint that the sum of transition probabilities must equal 1 for each (s, a)
                    # for s in self.psi.keys():
                    #     for a in range(self.env.nAction):
                    #         phi_sum = gp.quicksum(self.phi[(s, a, s_)] @ theta for s_ in range(self.env.nState))
                    #         model.addConstr(phi_sum == 1, name=f"probability_sum_{s}_{a}")

                    model.setParam('OutputFlag', 0)
                    
                    try:
                        model.optimize()
                        if model.status == GRB.OPTIMAL:
                            q[s,a] = self.env.reward[s,a][0] + self.gamma * model.objVal
                        else:
                            print(f"Optimization failed for state {s}, action {a} at {t_k}. Status: {model.status}")
                    except gp.GurobiError as e:
                        print(f"Gurobi error for state {s}, action {a} at {t_k}: {e}")
            
            for s in range(self.env.nState):
                v[s] = max(q[s,:])
            
            for s in range(self.env.nState):
                v[s] = min(v[s], min(v) + self.env.H)

            if cnt == self.N or np.max(v - v_old) - np.min(v - v_old) <= self.epsilon:
            # if cnt == self.N:
                break
                
        return q, v

    def POLICY(self, q):
        pi = {s: 0 for s in range(self.env.nState)}
        for s in range(self.env.nState):
            pi[s] = self.env.argmax(np.array([q[(s,a)] for a in range(self.env.nAction)]))
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
    d, D, T, N, delta, epsilon = args
    env = HardLinearMixtureMDP(d=d, D=D, T=T)
    agent = UCLK_C(env, T=T, delta=delta, N=N, epsilon=epsilon)  # N is set arbitrarily

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
    # filename = f"./UCLK-C/N={N}/regret_d_{d}_D_{D:.2f}_T_{T}_N_{N}_delta_{delta}_epsilon_{epsilon}.json"
    filename = f"./delta_1_over_2/UCLK-C/N={N}/regret_d_{d}_D_{D:.2f}_T_{T}_N_{N}_delta_{delta}_epsilon_{epsilon}.json"
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
    
    D_values = np.linspace(50, 150, 11)  # Example range for 'D'

    T_values = [10000]  # Time horizon
    
    N_values = [200, 300]
    
    delta_values = [0.05]  # Exploration-exploitation trade-off parameter
    
    epsilon_values = [0.000001]  # Precision for stopping the EVI

    # Run the parallel hyperparameter search
    results = parallel_hyperparameter_search(d_values, D_values, T_values, N_values, delta_values, epsilon_values)

    # Find the best hyperparameter combination based on maximum episodic return
    best_result = max(results, key=lambda x: x['cumulative_return'])

    # # Print out the best result
    print("Best hyperparameters found:")
    print(f"d={best_result['d']}, D={best_result['D']}, T={best_result['T']}, delta={best_result['delta']}, epsilon={best_result['epsilon']}")
    print(f"Episodic return: {best_result['cumulative_return']}, Optimal return: {best_result['opt_return']}")
    print(f"Regret file: {best_result['regret_file']}")

    # Plot the cumulative regret of the best parameter set
    plot_regret(best_result['regret_file'])


    # filename = "./UCLK-C/N=200/regret_d_8_D_181.50_T_5000_N_200_delta_0.05_epsilon_1e-06.json"
    # plot_regret(filename)
    