# UCRL-C

This repository implements and compares the algorithms 'UCRL-C' and 'UCRL2-VTR (Bernstein-typed)'.

## Algorithms
- **UCRL-C**: Nearly-minimax optimal algorithm for Linear Mixture MDP with Bounded Span (introduced in "Learning Infinite-Horizon Average-Reward Linear Mixture MDPs of Bounded Span")
- **UCRL2-VTR**: Nearly-minimax optimal algorithm for Linear Mixture MDP with bounded Diameter (introduced in "Nearly Minimax Optimal Regret for Learning Infinite-horizon Average-reward MDPs with Linear Function Approximation")

## Project Structure
- `algorithms/`: Algorithm implementations  
- `env/`: Hard-To-Learn MDP environments  
- `test/`: Create experiment outputs  
- `data/`: Logs and experiment outputs  
- `plot.ipynb`: Regret visualization notebook  

## Dependencies
**Install required packages**:
```bash
pip install -r requirements.txt
```

## Results
The results are visualized in the following PDF:  
[Hard-to-learn, UCRL-C, UCRL2-VTR(Bernstein)]((Hard-to-learn_UCLK-C_vs_UCRL2-VTR(Bernstein))_regret_d_8_D_120_T_10000_delta_0.05_epsilon_1e-06.pdf)



## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.





