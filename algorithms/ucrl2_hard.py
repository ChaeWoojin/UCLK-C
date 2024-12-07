import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from env.env import *  

class UCRL2(object):
    def __init__(self, env, T, delta):
        self.env = env
        self.T = T
        self.delta = delta
        
        # Initialize counts and estimates
        self.N_sa = np.zeros((self.env.nState, self.env.nAction))
        self.R_sa = np.zeros((self.env.nState, self.env.nAction))
        self.r_sa_hat = np.zeros((self.env.nState, self.env.nAction))
        self.P_sas = np.zeros((self.env.nState, self.env.nAction, self.env.nState))
        self.p_sas_hat = np.zeros((self.env.nState, self.env.nAction, self.env.nState))

        self.d_r = np.zeros((self.env.nState, self.env.nAction))
        self.d_p = np.zeros((self.env.nState, self.env.nAction))
                    
    def confidence_bounds(self, t_k):
        for s in range(self.env.nState):
            for a in range(self.env.nAction):
                self.d_r[s, a] = np.sqrt(7 * np.log(2 * self.env.nState * self.env.nAction * t_k / self.delta) / (2 * max(1, self.N_sa[s, a])))
                self.d_p[s, a] = np.sqrt(14 * self.env.nState * np.log(2 * self.env.nAction * t_k / self.delta) / max(1, self.N_sa[s, a]))
        
    
    def EVI(self, t_k):
        epsilon = 1 / np.sqrt(t_k)
        cnt = 0
        
        u = np.zeros(self.env.nState)
        Q = np.zeros((self.env.nState, self.env.nAction))
        pi = np.zeros(self.env.nState, dtype=int)        

        p = self.p_sas_hat.copy()
        while True:
            cnt += 1
            u_old = u.copy()
            s_dec = np.argsort(u_old)[::-1]
            
            # Best state -> bonus
            for s in range(self.env.nState):
                for a in range(self.env.nAction):
                    p[s, a, s_dec[0]] = min(1.0, self.p_sas_hat[s, a, s_dec[0]] + self.d_p[s, a] * 0.5)
            
            # Adjust probability
            for s in range(self.env.nState):
                for a in range(self.env.nAction):
                    l = len(s_dec) - 1
                    while (np.sum(p[s, a, :]) > 1):
                        p[s, a, s_dec[l]] = max(0, 1 - np.sum(p[s, a, :]) + p[s, a, s_dec[l]])
                        l = l - 1

            # Evaluate state values
            for s in range(self.env.nState):
                for a in range(self.env.nAction):
                    r_tilde_sa = self.r_sa_hat[s, a] + self.d_r[s, a]
                    Q[s, a] = r_tilde_sa + np.sum( [p[s, a, s_] * u[s_] for s_ in range(self.env.nState)] )
                u[s] = np.max(Q[s, :])

            # Check convergence
            diff = max(abs(u[s] - u_old[s]) for s in range(self.env.nState)) - min(abs(u[s] - u_old[s]) for s in range(self.env.nState))
            if cnt == 200 or  diff <= epsilon:
                break

        # Extract policy
        for s in range(self.env.nState):
            action_values = np.array(Q[s, :])
            pi[s] = np.random.choice(np.flatnonzero(action_values == action_values.max()))

        return pi

        
    def run(self):
        print('UCRL2')
        
        cumulative_return = []
        R = 0
        t_k = 1
        N_k = self.N_sa.copy()
        v_k = np.zeros((self.env.nState, self.env.nAction))
        pi = np.random.randint(0, self.env.nAction, size=self.env.nState).astype(int)

        for t in tqdm(range(1, self.T+1)):
            s = self.env.state
            a = pi[s]
            r, s_ = self.env.advance(a)
            R += r

            v_k[s, a] += 1  
            self.N_sa[s, a] += 1
            self.P_sas[s, a, s_] += 1
            self.R_sa[s, a] += r
            cumulative_return.append(R)
            
            # episode update
            if v_k[s, a] >= max(1, N_k[s, a]): 
                t_k = t + 1
                v_k = np.zeros((self.env.nState, self.env.nAction))
                N_k = self.N_sa.copy()
                
                for s in range(self.env.nState):
                    for a in range(self.env.nAction):
                        self.r_sa_hat[s, a] = self.R_sa[s, a] / max(1, self.N_sa[s, a])
                        for s_ in range(self.env.nState):                            
                            self.p_sas_hat[s, a, s_] = self.P_sas[s, a, s_] / max(1, self.N_sa[s, a])
                

                self.confidence_bounds(t_k)
                pi = self.EVI(t_k)

        return cumulative_return

    