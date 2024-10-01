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
        
        # self.triangle = (1/45 * np.sqrt(2*np.log(2)/5)) * (self.d) / np.sqrt(self.D * self.T)
        self.triangle = (1/5 * np.sqrt(2*np.log(2))) * (self.d) / np.sqrt(self.D * self.T)
        self.alpha = np.sqrt(self.triangle / ((self.d - 1) * (1 + self.triangle)))
        self.beta =  np.sqrt(1 / (1 + self.triangle))
        
        self.theta = np.random.choice([-1, 1], self.d - 1) * self.triangle / (self.d - 1)
        self.theta_tilde = np.concatenate((self.theta / self.alpha, np.array([1 / self.beta])))
        self.actions = np.array(list(itertools.product([-1, 1], repeat=self.d - 1)))
        self.reward = self.generate_reward()
        self.phi = self.generate_phi()
        
        self.J_star = (self.delta + self.triangle) / (2 * self.delta + self.triangle)
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
        return total_reward_optimal

    def argmax(self,b):
        return np.random.choice(np.flatnonzero(b == b.max()))   

# d = 8
# D = 5
# T = 500

# mdp = HardLinearMixtureMDP(d=d, D=D, T=T)
# triangle = mdp.triangle
# actions = mdp.actions
# theta = mdp.theta
# delta = mdp.delta
# rank = mdp.action_rank

# print(triangle * 4, delta)
# print(1/delta - 1, T/5)

# print(1 - delta - np.dot(theta, actions[rank[0]]))
# print(1 - delta - np.dot(theta, actions[rank[-1]]))

# print(mdp.phi)