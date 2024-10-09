import numpy as np
import itertools

class HardLinearMixtureMDP:
    def __init__(self, d, D, T):
        self.d = d
        self.D = D
        self.T = T
        self.delta = 1 / self.D
        self.nState = 2
        self.nAction = 2**(self.d - 1)
        self.timestep = 0
        self.state = 0 
        
        # self.triangle = 1/45 * (self.d - 1) / np.sqrt((2 * self.T * np.log(2)) / (5 * self.delta))
        self.triangle = 1/2 * (self.d - 1) / np.sqrt((2 * self.T * np.log(2)) / (5 * self.delta))
        self.alpha = np.sqrt(self.triangle / ((self.d - 1) * (1 + self.triangle)))
        self.beta =  np.sqrt(1 / (1 + self.triangle))
        
        self.theta = np.random.choice([-1, 1], self.d - 1) * self.triangle / (self.d - 1)
        self.theta_tilde = np.concatenate((self.theta / self.alpha, np.array([1 / self.beta])))
        self.actions = np.array(list(itertools.product([-1, 1], repeat=self.d - 1)))
        self.reward = self.generate_reward()
        self.phi = self.generate_phi()
        
        self.J_star = (self.delta + self.triangle) / (2 * self.delta + self.triangle)
        self.H = 2 / (2 * self.delta + self.triangle)
        self.action_rank = np.argsort(np.array([np.dot(self.actions[i], self.theta) for i in range(self.nAction)]))[::-1]
                      
    def reset(self):
        self.state = 0
        self.timestep = 0
        return self.state
        
    def generate_reward(self):
        reward = {}
        for i in range(self.nAction):
            reward[0, i] = (0, 0)
            reward[1, i] = (1, 0)
        return reward
    
    def generate_phi(self):
        phi = {(s, a, s_): np.zeros(self.d) for s in range(self.nState) for a in range(self.nAction) for s_ in range(self.nState)} 
        
        for i in range(self.nAction):
            action_vector = self.actions[i]  # Assuming self.actions is a list of action vectors with size (self.d - 1)
            phi[(0, i, 0)] = np.concatenate((-self.alpha * action_vector, [self.beta * (1 - self.delta)]))
            phi[(0, i, 1)] = np.concatenate((self.alpha * action_vector, [self.beta * self.delta]))
            phi[(1, i, 0)] = np.concatenate((np.zeros(self.d - 1), [self.beta * self.delta]))
            phi[(1, i, 1)] = np.concatenate((np.zeros(self.d - 1), [self.beta * (1 - self.delta)]))
            
        return phi
    
    def transition_prob(self, s, a):
        action = self.actions[int(a)]
        if s == 0:  # from state x0
            prob_x0 = 1 - self.delta - np.dot(action, self.theta)
            prob_x1 = self.delta + np.dot(action, self.theta)
        else:       # from state x1
            prob_x0 = self.delta
            prob_x1 = 1 - self.delta

        return np.array([prob_x0, prob_x1])
    
    def advance(self, action):
        state = self.state
        probs = self.transition_prob(state, action)
        reward = self.reward[state, action][0]
        next_state = np.random.choice([0, 1], p=probs)
        self.state = next_state
        self.timestep += 1
        return reward, self.state
    
    def run_optimal_policy(self):
        optimal_policy = [self.action_rank[0], self.action_rank[0]]
        total_reward_optimal = []
        R = 0
        self.reset() 
        for t in range(self.T):
            s_t = self.state
            a_t = optimal_policy[s_t] 
            _, reward = self.advance(a_t) 
            R += reward
            total_reward_optimal.append(R)
        
        # total_reward_optimal = []
        # R = 0
        # for t in range(self.T):
        #     R += self.J_star
        #     total_reward_optimal.append(R)       
        return total_reward_optimal

    def argmax(self,b):
        return np.random.choice(np.flatnonzero(b == b.max()))   

def run_mdp_with_hyperparams(d, D, T):
    mdp = HardLinearMixtureMDP(d=d, D=D, T=T)
    triangle = mdp.triangle
    actions = mdp.actions
    theta = mdp.theta
    theta_til = mdp.theta_tilde
    delta = mdp.delta
    alpha = mdp.alpha
    beta = mdp.beta
    H = mdp.H
    rank = mdp.action_rank

    print(f"Running MDP with d={d}, D={D}, T={T}")
    print("delta:", delta, "triangle:", triangle, "alpha:", alpha, "beta:", beta)
    print("theta:", theta)
    print("theta_til:", theta_til)
    print("Condition 1 (3*triangle <= delta) LHS:", triangle * 3, "RHS:", delta)
    print("Condition 2 (T >= 16(d-1)**2/(2025 delta)) LHS:", T, "RHS:", 16 * (d-1)**2 / (delta * 2025))
    print("P(x1|x0, a_best)", delta + np.dot(theta, actions[rank[0]]))
    print("P(x1|x0, a_worst)", delta + np.dot(theta, actions[rank[-1]]))
    print("P(x0|x0, a_best):", 1 - delta - np.dot(theta, actions[rank[0]]))
    print("P(x0|x0, a_worst):", 1 - delta - np.dot(theta, actions[rank[-1]]))
    print("P(x1|x1, _)", 1 - delta)
    print("P(x0|x1, _)", delta)
    print("H:", H)

    # Here you can run the optimal policy to check its performance
    total_reward_optimal = mdp.run_optimal_policy()
    print(f"Total reward for optimal policy: {total_reward_optimal[-1]}")
    print("\n")
    return total_reward_optimal

# # Define the hyperparameters ranges to try
# d_values = [8]  # Example values for 'd'
# D_values = np.linspace(130, 150, 2)  # Example values for 'D'
# T_values = [10000]  # Example values for 'T'

# # Iterate over all hyperparameter combinations
# for d, D, T in itertools.product(d_values, D_values, T_values):
#     total_reward_optimal = run_mdp_with_hyperparams(d, D, T)