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

## Learning Outcomes
After working through our handbook, we expect you to have the following know-how's (please rephrase):
* have a solid understanding of Neurovolution's origins
* understand how NE works in general
* understand the NEAT algorithm, in full
* distinguish between variants of the method
* understand NE's limitations
* thus, know when NE is useful and when not
* understand the proposed enhanced methods
* be able to apply the algorithm yourself (maybe we should have a template / empty ish file for use in other cases)

  
## Project Overview
Our project, named Snake-NE, aims to simplify the learning of NeuroEvolution (NE), a method that blends evolutionary algorithms with machine learning. We teach the NEAT method to evolve both the weights, biases, and topology of a neural network that learns to play the Snake game.

Now, we will more clearly introduce our project.
In short, NE aims to improve the computation efficiency and performance of a neural network by not only changing the wrights of the network, but also evolve the network's architecture using evlutionary algorithms. Let's begin by clearly introducing all of these mentioned terms, to offer us a solid ground for the rest of this lesson.

## Definitions
**Neural Netoworks**

**Reinforcement Learning**

**Evolutionary Algorithms**

## Motivation
Snake-NE offers several benefits as an educational tool, making learning more engaging and accessible by using the simple yet fun Snake game to demonstrate complex neuroevolution concepts.


*hint to the extensive literature review // talk here about then to use it briefly. 


# Literature Review
In this section, we review relevant literature that underpins our project and provides context for our methods and experiments.

## NEAT paper

## Evolving Deep Network Architectures
We explore the adaptation of NEAT for Neural Architecture Search (NAS) and discuss bilevel optimization, scalability, and adaptation to environmental constraints.
* some extended methods

## Learning Atari Games using NE
We investigate the effectiveness of non-gradient-based evolutionary algorithms (EAs) for training deep neural networks on reinforcement learning tasks, such as Atari games.
* basically *when* NE is good


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

# Do It Yourself
We prepared the google colab for you. Now, it's the time for you to have your first tries with the code, familiarise yourself with it before we go on implementing our methods. 

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
