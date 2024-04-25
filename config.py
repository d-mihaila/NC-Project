class Config:   
    N_RUNS = 100  # Number of runs to evaluate each genome


class Paths:
    RESULTS_PATH = 'results'      # PATH to the results directory
    NEAT_CONFIG_PATH = 'config-neat'  # Path to the NEAT configuration file
    DRAW_NET_PATH = 'target_pursuit_2000_results/winner-feedforward.gv'  # Path to the neural network visualization
    WINNER_PATH = 'winner-feedforward'  # Path to the winner genome


#brew install graphviz