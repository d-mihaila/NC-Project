from __future__ import print_function
from snake import *
from config import *
from visualize import *
from replay_winner import *

from random import random
import random as random_module
import multiprocessing
import pickle
import os
import neat
import visualize
import numpy as np
import pygame
import configparser

random_module.seed(42)


"""     this page created the interface to be shown to the user
        1. choosing the parameters
        2. showing the animation of the winner
        3. presenting the statistics of the run
        this can be repeated.
"""

def parameter_board(config_file):
    pygame.init()

    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption('Enter Parameters')

    # Colors
    black = (0, 0, 0)
    white = (255, 255, 255)
    gray = (200, 200, 200)
    dark_gray = (50, 50, 50)

    # Font setup
    font = pygame.font.SysFont('Arial', 24)
    label_font = pygame.font.SysFont('Arial', 18)

    # Define rectangles for input boxes and button
    input_bias = pygame.Rect(50, 130, 200, 30)
    input_weight = pygame.Rect(50, 200, 200, 30)
    start_button = pygame.Rect(200, 300, 100, 50)

    # Text within the boxes
    text_bias = ''
    text_weight = ''
    active_bias = False
    active_weight = False

    # Rendering the "START" button text
    start_text = font.render('START', True, white)

    # Labels for the input fields
    bias_label_text = label_font.render('Bias Mutation Rate:', True, black)
    weight_label_text = label_font.render('Weight Mutation Rate:', True, black)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if input boxes are clicked
                if input_bias.collidepoint(event.pos):
                    active_bias = True
                    active_weight = False
                elif input_weight.collidepoint(event.pos):
                    active_weight = True
                    active_bias = False
                else:
                    active_bias = False
                    active_weight = False

                # Check if the start button is clicked
                if start_button.collidepoint(event.pos):
                    # Assuming the necessary parameters are ready
                    # Here you might still want to read or set bias and weight if needed for `test_winner`
                    bias, weight = text_bias, text_weight
                    new_params = {"bias_mutate_rate": bias, "weight_mutate_rate": weight}
                    file_path = '/Users/utilizator/Desktop/Neuroevolution project/NC-Project/config-feedforward_commented'
                    section = "DefaultGenome"
                    # overwrite_config(file_path, section, new_params)

                    # Assuming sys.argv[1] should be the path for 'one run'
                    # If test_winner needs other arguments, make sure they are provided here

                    # somehow figure out how to show the statistics after the run. 
                    # Call test_winner with a callback that handles the plotting
                    test_winner(file_path, sys.argv[1], callback=on_animation_complete)

                    running = False


            if event.type == pygame.KEYDOWN:
                if active_bias:
                    if event.key == pygame.K_BACKSPACE:
                        text_bias = text_bias[:-1]
                    else:
                        text_bias += event.unicode
                elif active_weight:
                    if event.key == pygame.K_BACKSPACE:
                        text_weight = text_weight[:-1]
                    else:
                        text_weight += event.unicode

        screen.fill(white)  # Set background to white

        # Draw the labels above the input boxes
        screen.blit(bias_label_text, (50, 100))
        screen.blit(weight_label_text, (50, 170))

        # Draw the input boxes for bias and weight
        pygame.draw.rect(screen, gray if active_bias else dark_gray, input_bias, 0)
        pygame.draw.rect(screen, gray if active_weight else dark_gray, input_weight, 0)
        bias_surf = font.render(text_bias, True, black)
        weight_surf = font.render(text_weight, True, black)
        screen.blit(bias_surf, (input_bias.x + 5, input_bias.y + 5))
        screen.blit(weight_surf, (input_weight.x + 5, input_weight.y + 5))

        # Draw the start button
        pygame.draw.rect(screen, dark_gray, start_button)
        screen.blit(start_text, (start_button.x + 10, start_button.y + 10))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return text_bias, text_weight


def on_animation_complete(config_file):
    # This function will be called after the animation
    # Load configuration into a NEAT object.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Set results directory
    if not os.path.exists(f"{Paths.RESULTS_PATH}/checkpoints"):
        os.makedirs(f"{Paths.RESULTS_PATH}/checkpoints")

    # Create the population
    p = neat.Population(config)

    # Add various reporters that log the process to the terminal and provide statistics.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    # visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")



def overwrite_config(file_path, section, new_params):
    # Create a configparser object
    config = configparser.ConfigParser()

    # Read the config file
    config.read(file_path)

    # Overwrite the parameters with the new values
    for key, value in new_params.items():
        if key in config[section]:
            config[section][key] = str(value)

    # Write the updated data back to the config file
    with open(file_path, 'w') as f:
        config.write(f)


