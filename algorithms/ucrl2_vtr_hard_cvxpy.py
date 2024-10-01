import numpy as np
from tqdm import tqdm
import cvxpy as cp

class UCRL2_VTR(object):
    def __init__(self, env, T, c, delta, lam, epsilon):
        self.env = env
        self.T = T
        self.d = env.d

        # ground truth
        self.theta_star = self.env.theta_tilde

        # gram matrix
        self.A = np.identity(self.d)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros(self.d)
        
        self.lam = lam  
        self.B = max(self.env.triangle ** 2 + 1, np.linalg.norm(self.theta_star, ord=2))
        self.delta = delta
        self.epsilon = epsilon

        self.phi = env.phi

        # theta
        self.theta = np.zeros(self.d)
        
    def mixture(self, s, a, u):
        return np.sum(np.array([np.multiply(u[s_], self.phi[(s, a, s_)]) for s_ in range(self.env.nState)]), axis=0)

    def act(self, s, w_k):
        return self.env.argmax( np.array([self.env.reward[s,a][0] + np.dot(self.theta, self.mixture(s, a, w_k)) for a in range(self.env.nAction)]) )

    def Beta(self, t_k):
        return self.env.D * np.sqrt(self.d * np.log((self.lam + ((self.env.D ** 2) * t_k)) / (self.delta * self.lam))) + np.sqrt(self.lam) * self.B

    def EVI(self, t_k):
        cnt = 0
        u = {s: 0.0 for s in range(self.env.nState)}
        Beta_t = self.Beta(t_k)
        while True:
            u_old = u.copy()
            cnt += 1
            for s in range(self.env.nState):
                max_value = -1e9
                for a in range(self.env.nAction):
                    phi_u = self.mixture(s, a, u)

                    theta = cp.Variable(self.d, value=self.theta if self.theta is not None else None)
                    objective = cp.Maximize( phi_u @ theta)

                    C_t = cp.quad_form(theta - self.theta, self.A)
                    constraints = [C_t <= Beta_t**2,
                                cp.sum(theta) == 1,
                                theta >= 0,
                                cp.norm(theta) <= self.B]

                    prob = cp.Problem(objective, constraints)
                
                    try:
                        prob.solve(solver=cp.GUROBI, warm_start=True, verbose = False)
                        if prob.status == cp.OPTIMAL:
                            value = self.env.reward[s,a][0] + prob.value # Calculate Q-value
                            max_value = max(max_value, value)
                        else:
                            print(f"Optimization failed for state {s}, action {a} at {t_k}. Status: {prob.status}")
                    except cp.error.SolverError:
                        print(f"Solver error for state {s}, action {a} at {t_k}")
                u[s] = max_value

            # Check for convergence
            if cnt == 100 or max(u[s] - u_old[s] for s in range(self.env.nState)) - min(u[s] - u_old[s] for s in range(self.env.nState)) <= self.epsilon:
                break
            
        return u

    def POLICY(self, u_k, t_k):
        pi = {}
        Beta_t = self.Beta(t_k)
        for s in range(self.env.nState):
            Q = []
            for a in range(self.env.nAction):
                phi_u = self.mixture(s, a, u_k)

                theta = cp.Variable(self.d, value=self.theta if self.theta is not None else None)
                objective = cp.Maximize( phi_u @ theta )

                C_t = cp.quad_form(theta - self.theta, self.A)
                constraints = [C_t <= Beta_t**2,
                                cp.sum(theta) == 1,
                                theta >= 0,
                                cp.norm(theta) <= self.B]

                prob = cp.Problem(objective, constraints)     
                try:
                    prob.solve(solver=cp.GUROBI, warm_start=True, verbose = False)
                    if prob.status == cp.OPTIMAL:
                        theta_k = theta.value
                        Q.append(self.env.reward[s, a][0] + np.dot(theta_k, phi_u))
                    else:
                        print(f"Optimization failed for state {s}, action {a} at {t_k}. Status: {prob.status}")
                except cp.error.SolverError:
                    print(f"Solver error for state {s}, action {a} at {t_k}")      
            
            pi[s] = np.argmax(Q)
        return pi

    def run(self):
        print('UCRL2_VTR')
        episode_return = []  # round_return

        A_k = self.A.copy()  # Copy of the Gram matrix
        t_k = 1
        w_k = {s: 1.0 for s in range(self.env.nState)}
        R = 0
        for t in tqdm(range(1, self.T + 1)):
            if np.linalg.det(self.A) > 2 * np.linalg.det(A_k):  # Update at episode boundaries
                t_k = t
                A_k = self.A.copy()

                # Perform EVI
                u_k = self.EVI(t_k)
                pi = self.POLICY(u_k, t_k)
                
                tmp = (max(u_k) - min(u_k)) / 2
                w_k = {s: u_k[s] - tmp for s in range(self.env.nState)}

            s = self.env.state
            if t_k == 1:
                a = np.random.choice([a for a in range(self.env.nAction)])  # Initial random action selection
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



