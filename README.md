# UCLK-C

This repository implements and compares the performance of the following algorithms: 'UCLK-C' (our proposed algorithm), 'UCRL2-VTR (Bernstein-type)', 'TSDE', 'UCRL2', and 'RANDOM'.

## Algorithms
- **UCLK-C**: introduced in "Learning Infinite-Horizon Average-Reward Linear Mixture MDPs of Bounded Span"
- **UCRL2-VTR**: introduced in "Nearly Minimax Optimal Regret for Learning Infinite-horizon Average-reward MDPs with Linear Function Approximation"
- **TSDE**: introduced in "Learning Unknown Markov Decision Processes: A Thompson Sampling Approach"
- **UCRL2**: introduced in "Near-optimal Regret Bounds for Reinforcement Learning"
- **RANDOM**: take arbitrary action 

## Project Structure
- `algorithms/`: Algorithm implementations  
- `env/`: Hard-To-Learn MDP environments  
- `test/`: Create experiment outputs  
- `data/`: Logs and experiment outputs  
- `image/`: Regret plots
- `plot.ipynb`: Regret visualization notebook  

## Dependencies
**Install required packages**:
```bash
pip install -r requirements.txt
```

## Results


## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.





