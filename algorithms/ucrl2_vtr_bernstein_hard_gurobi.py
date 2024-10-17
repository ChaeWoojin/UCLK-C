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

class UCRL2_VTR_BERNSTEIN(object):
    def __init__(self, env, T, N, delta, epsilon):
        self.env = env
        self.T = T
        self.d = env.d
        self.gamma = 1 - np.sqrt(self.d / (self.env.D * self.T))

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
                    
                    model.addConstr((theta - self.theta_hat) @ self.A_hat @ (theta - self.theta_hat) <= Beta_t**2, "C_t")
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
        pi = np.zeros(self.env.nState)
        Beta_t = self.Beta(t_k)
        for s in range(self.env.nState):
            q = []
            for a in range(self.env.nAction):
                phi_u = self.mixture(s, a, u_k)

                model = gp.Model("optimization")
                theta = model.addMVar(self.d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")
                
                model.setObjective(phi_u @ theta, GRB.MAXIMIZE)
                
                model.addConstr((theta - self.theta_hat) @ self.A_hat @ (theta - self.theta_hat) <= Beta_t**2, "C_t")
                model.addConstr(theta @ theta <= self.B**2, "norm_constraint")
                
                # Add non-negativity constraints for transition probabilities
                for s in range(self.env.nState):
                    for a in range(self.env.nAction):
                        for s_ in range(self.env.nState):
                            model.addConstr(self.phi[(s, a, s_)] @ theta >= 0, name=f"probability_nonneg_{s}_{a}_{s_}")

                # Add constraint that the sum of transition probabilities must equal 1 for each (s, a)
                for s in range(self.env.nState):
                    for a in range(self.env.nAction):
                        phi_sum = gp.quicksum(self.phi[(s, a, s_)] @ theta for s_ in range(self.env.nState))
                        model.addConstr(phi_sum == 1, name=f"probability_sum_{s}_{a}")

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

    def VARIANCE(self, s, a, w, t):
        phi_w2 = self.mixture(s, a, np.square(w))
        phi_w = self.mixture(s, a, w)
        
        Beta_check = 8 * self.d * np.sqrt(np.log(1 + t / (4 * self.lam)) * np.log(4 * t**2 / self.delta)) \
                    + 4 * np.sqrt(self.d) * np.log(4 * t**2 / self.delta) + np.sqrt(self.lam) * self.B
        Beta_til = 2 * self.env.D**2 * np.sqrt(self.d * np.log(1 + (t * self.env.D**2) / (4 * self.d * self.lam)) * np.log(4 * t**2 / self.delta)) \
                    + self.env.D**2 * np.log(4 * t**2 / self.delta) + np.sqrt(self.lam) * self.B
        
        VW = np.clip(np.dot(phi_w2, self.theta_til), 0, self.env.D**2) - np.clip(np.dot(phi_w, self.theta_hat), 0, self.env.D)**2
        E = min(self.env.D**2 / 4 , self.env.D * Beta_check * np.sqrt(np.dot(np.dot(phi_w, self.Ainv_hat), phi_w))) \
            + min(self.env.D**2 / 4 , Beta_til * np.sqrt(np.dot(np.dot(phi_w2, self.Ainv_til), phi_w2)))

        return VW + E

    def run(self):
        print('UCRL2-VTR(BERNSTEIN)')
        cumulative_return = []

        t_k = 1
        A_hat_k = self.A_hat.copy()

        w_k = np.ones(self.env.nState)
        R = 0

        for t in tqdm(range(1, self.T + 1)):
            if np.linalg.det(self.A_hat) > 2 * np.linalg.det(A_hat_k):
                t_k = t
                A_hat_k = self.A_hat.copy()

                u_k = self.EVI(t_k)
                pi = self.POLICY(u_k, t_k)
                
                w_k = u_k - (max(u_k) - min(u_k)) / 2

            s = self.env.state
            if t_k == 1:
                a = np.random.choice([a for a in range(self.env.nAction)])
            else:
                a = pi[s]
            r, s_ = self.env.advance(a)
            R += r
            
            sig = np.sqrt(max(self.env.D**2 / self.d, self.VARIANCE(s, a, w_k, t)))

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