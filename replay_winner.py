from __future__ import print_function
from snake import *
from config import *

import multiprocessing
import pickle
import os
import neat
import visualize
import sys

#python3 replay_winner.py target_pursuit_2000_results/winner-feedforward


def replay_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config) # Initialize the neural network from the passed genome.

    # Define the node names for visualization.
    node_names = {
        -1 : "Wall_N",
        -2 : "Wall_S",
        -3 : "Wall_E",
        -4 : "Wall_W",
        -5 : "Tail_N",
        -6 : "Tail_S",
        -7 : "Tail_E",
        -8 : "Tail_W",
        -9 : "Apple_N",
        -10 : "Apple_S",
        -11 : "Apple_E",    
        -12 : "Apple_W",
        -13 : "Wall_NE",
        -14 : "Wall_SE",
        -15 : "Wall_SW",
        -16 : "Wall_NW",
        -17 : "Apple_NE",
        -18 : "Apple_SE",
        -19 : "Apple_SW",
        -20 : "Apple_NW",
        -21 : "Obstacle_N",
        -22 : "Obstacle_S",
        -23 : "Obstacle_E",
        -24 : "Obstacle_W",
        -25 : "Obstacle_NE",
        -26 : "Obstacle_SE",
        -27 : "Obstacle_SW",
        -28 : "Obstacle_NW",
        0: 'Up', 1 : "Left", 2 : "Down", 3 : "Right"
    }

    # Visualize the neural network.
    visualize.draw_net(config, genome, False, node_names=node_names, filename=f"{Paths.RESULTS_PATH}/network.gv")

    simulate_animation(net, genome, config) # Simulate the environment with a GUI.

def test_winner(config_file, genome):
    with open(genome, "rb") as f:
        winner = pickle.load(f, encoding="latin-1")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    replay_genome(winner, config)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, Paths.NEAT_CONFIG_PATH)
    test_winner(config_path, Paths.WINNER_PATH)