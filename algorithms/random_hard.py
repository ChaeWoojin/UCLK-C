from tqdm import tqdm
import numpy as np


class RANDOM(object):
    def __init__(self, env, T):
        self.env = env
        self.T = T
    
    def run(self):
        print('RANDOM')
        cumulative_return = []
        nAction = self.env.nAction  
        
        R = 0
        for t in tqdm(range(1, self.T + 1)):
            s = self.env.state
            a = np.random.randint(0, nAction, size = 1)[0]
            
            r, s_ = self.env.advance(a) 
            R += r 
            cumulative_return.append(R)
            
        return cumulative_return