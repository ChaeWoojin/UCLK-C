from env.env import HardLinearMixtureMDP
from algorithms.ucrl2_vtr_hard_gurobi import UCRL2_VTR
import matplotlib.pyplot as plt
import numpy as np

d = 8
D = 5
T = 500

env = HardLinearMixtureMDP(d=d, D=D, T=T)
epsilon = 0.01
agent = UCRL2_VTR(env, T=T, c=1e-2, delta=0.01, lam=1, epsilon=epsilon)    
returns = agent.run()

# Assuming returns is the array of rewards collected over time

# Plot the cumulative rewards over time
plt.figure(figsize=(10, 6))
plt.plot(returns, label='Cumulative Returns')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Rewards')
plt.title('Cumulative Rewards over Time')
plt.legend()
plt.grid(True)
plt.show()
