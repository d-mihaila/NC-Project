# Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Motivation](#motivation)
- [Literature Review](#literature-review)
  - [Evolving Deep Network Architectures](#evolving-deep-network-architectures)
  - [Learning Atari Games using NE](#learning-atari-games-using-ne)
- [Methods](#methods)
  - [Snake-NE](#snake-ne)
  - [NEAT Setup](#neat-setup)
  - [Input Strategies](#input-strategies)
  - [Ablation Experiments](#ablation-experiments)
  - [Educational Website](#educational-website)
- [Results](#results)
  - [Input Strategy Experiments](#input-strategy-experiments)
  - [Ablation Experiments Results](#ablation-experiments-results)
- [Discussion](#discussion)
  - [Limitations](#limitations)
  - [Future Work](#future-work)
- [Conclusion](#conclusion)

---

# Introduction
Welcome to the handbook for our educational project on the neuroevolution snake game. This guide aims to provide both introductory and detailed information about evolutionary algorithms, specifically focusing on the NeuroEvolution of Augmenting Topologies (NEAT) method.

# Problem Statement
Our project, named Snake-NE, aims to simplify the learning of NeuroEvolution (NE), a method that blends evolutionary algorithms with machine learning. We teach the NEAT method to evolve both the weights, biases, and topology of a neural network that learns to play the Snake game.

# Motivation
Snake-NE offers several benefits as an educational tool, making learning more engaging and accessible by using the simple yet fun Snake game to demonstrate complex neuroevolution concepts.

# Literature Review
In this section, we review relevant literature that underpins our project and provides context for our methods and experiments.

## Evolving Deep Network Architectures
We explore the adaptation of NEAT for Neural Architecture Search (NAS) and discuss bilevel optimization, scalability, and adaptation to environmental constraints.

## Learning Atari Games using NE
We investigate the effectiveness of non-gradient-based evolutionary algorithms (EAs) for training deep neural networks on reinforcement learning tasks, such as Atari games.

# Methods
This section details the setup and implementation of our Snake-NE project, including the NEAT algorithm and various experiments.

## Snake-NE
The complete source code for our Snake-NE project, developed using the PyGame library, is available on GitHub. The NEAT algorithm was implemented using the neat-python package.

## NEAT Setup
We describe the NEAT algorithm, including network initialization, genetic encoding, crossover strategy, speciation strategy, and mutation strategy.

## Input Strategies
We investigate how different input features influence the snake's learning and strategies. This includes experiments with baseline, binary, and collision input strategies.

## Ablation Experiments
We analyze the essential components of the NEAT algorithm by systematically removing key components and observing their impact on performance.

## Educational Website
We developed an educational webpage featuring explanations, GIFs, videos, and links to the code, allowing users to experiment on their own.

# Results
This section presents the outcomes of our experiments, focusing on input strategy experiments and ablation experiments.

## Input Strategy Experiments
We describe the results of various input strategy experiments, highlighting the best-performing networks and observed behaviors.

## Ablation Experiments Results
We discuss the findings from our ablation experiments, illustrating the importance of key components in the NEAT algorithm.

# Discussion
We reflect on the limitations of our project and propose future work to build on our findings.

## Limitations
We acknowledge the constraints and limitations of our project, such as computational resources and visualization tools.

## Future Work
We suggest potential directions for future research and improvements, including exploring new input strategies, fitness function design, and comparisons with reinforcement learning.

# Conclusion
[To be written]
