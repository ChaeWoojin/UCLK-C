from scipy.stats import dirichlet, beta
from tqdm import tqdm
import numpy as np


class TSDE(object):
    def __init__(self, env, T, alpha = 1, beta = 1):
        self.env = env
        self.T = T
        
        self.nState = env.nState
        self.nAction = env.nAction

        # Dirichlet parameters for transitions (prior)
        self.state_action_counts = np.zeros((self.nState, self.nAction))
        self.transition_counts = np.ones((self.nState, self.nAction, self.nState))
        
    def sample_model(self):
        """
        Sample transition probabilities and self.env.reward from the posterior.
        """
        sampled_transitions = np.array([
            [dirichlet(self.transition_counts[s, a]).rvs()[0] for a in range(self.nAction)]
            for s in range(self.nState)
        ])
        
        return sampled_transitions
        
    def update_posterior(self, state, action, next_state, reward):
        """
        Update the posterior with observed (state, action, next_state, reward).
        """
        # Update visit counts
        self.state_action_counts[state, action] += 1
        
        # Update transition counts
        self.transition_counts[state, action, next_state] += 1
        
        
    def solve_mdp(self, transitions, discount_factor=0.9, eps=1e-6):
        """
        Solve the MDP using value iteration given transitions.
        """
        value = np.zeros(self.nState)
        while True:
            new_value = np.zeros(self.nState)
            for s in range(self.nState):
                action_values = [
                    np.sum([transitions[s, a, s_prime] * (self.env.reward[s, a] + discount_factor * value[s_prime]) 
                            for s_prime in range(self.nState)])
                    for a in range(self.nAction)
                ]
                new_value[s] = max(action_values)
            if np.max(np.abs(new_value - value)) < eps:
                break
            value = new_value

        # Derive policy
        policy = np.zeros(self.nState, dtype=int)
        for s in range(self.nState):
            action_values = np.array([
                np.sum([transitions[s, a, s_prime] * (self.env.reward[s, a] + discount_factor * value[s_prime]) 
                        for s_prime in range(self.nState)])
                for a in range(self.nAction)
            ])
            policy[s] = np.random.choice(np.flatnonzero(action_values == action_values.max()))  
        return policy

    def episode_update(self, N_k, t, t_k, T_k1):
        if t > t_k + T_k1:
            return True
        
        for state in range(self.nState):
            for action in range(self.nAction):
                if (self.state_action_counts[state, action] > 2 * N_k[state, action]):
                    return True
        return False
            
    
    def run(self):
        """
        Run Thompson Sampling on the given environment.
        """
        cumulative_return = []
        T_k1 = 0
        t_k = 0
        N_k = self.state_action_counts.copy()
        sampled_transitions = self.sample_model()
        policy = self.solve_mdp(sampled_transitions)
        
        R = 0
        for t in tqdm(range(1, self.T + 1)):
            if self.episode_update(N_k, t, t_k, T_k1):
                # print(f"episode_update at time {t}")
                # Update episode
                N_k = self.state_action_counts.copy()
                T_k1 = t - t_k
                t_k = t
                
                # Sample from posterior
                sampled_transitions = self.sample_model()

                # Solve the sampled MDP
                policy = self.solve_mdp(sampled_transitions).copy()

            # Run over episode
            state = self.env.state
            action = policy[state]
            reward, next_state = self.env.advance(action)
            # print(f"state: {state}, action: {action}, reward: {reward}")

            # Update posterior
            self.update_posterior(state, action, next_state, reward)

            R += reward

            cumulative_return.append(R)

        return cumulative_return
    
