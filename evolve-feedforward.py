from __future__ import print_function
from snake import *
from replay_winner import *
from config import Config


import multiprocessing
import pickle
import os
import neat
import visualize
import sys


def eval_genomes(genomes, config):
    """
    Iterates through each genome, uses NEAT to create a neural
    network from the genome, and then evaluates its fitness using a simulation function.
    
    Args:
    genomes (list of tuples): List of (genome_id, genome) tuples.
    config (neat.Config): NEAT configuration settings for network creation.
    """
    best_genome = None
    best_fit = -1
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config) # Create a neural network from the genome.
        genome.fitness = simulate_headless(net)  # Simulate the environment without a GUI.

        # Update the best genome if the current genome has a higher fitness.
        if genome.fitness > best_fit:
            best_fit = genome.fitness
            best_genome = genome

    if best_fit >= 20:
        replay_genome(best_genome, config)  # Replay the best genome if its fitness is high enough.

def eval_genome(genome, config):
    """
    Create a neural network from a single genome and evaluate its fitness.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = simulate_headless(net)  # Evaluate the genome in a headless simulation.
    return fitness

def run(config_file):
    """
    Configure NEAT from a file and run the evolutionary algorithm to find the best performing genome.

    """
    # Load configuration into a NEAT object.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    #Set results directory
    if not os.path.exists(Config.RESULTS_DIR):
        os.makedirs(Config.RESULTS_DIR)

    # Create the population
    p = neat.Population(config)

    # Add various reporters that log the process to the terminal and provide statistics.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, filename_prefix=f"{Config.RESULTS_DIR}/population-"))

    # Use parallel evaluation to efficiently utilize available CPU cores.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    # Runs NEAT’s genetic algorithm for at most n generations
    winner = p.run(pe.evaluate, n=Config.N_RUNS)  

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    # Visualize statistics and species progression over generations.
    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")


def test_population(config_file, pop_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Checkpointer.restore_checkpoint(pop_file)
    p.run(eval_genomes, 1)



if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    run(config_path)
