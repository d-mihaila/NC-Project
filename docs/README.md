# Table of Contents
NOTE: to add references later...!?

- [1. Introduction](#1-introduction)
  - [1.1 Learning Outcomes](#11-learning-outcomes)
  - [1.2 Project Overview](#12-project-overview)
  - [1.3 Problem Statement](#14-problem-statement)
  - [1.4 Definitions](#13-definitions)
  - [1.5 Motivation](#15-motivation)
- [2. Literature Review](#2-literature-review)
  - [2.1 NEAT paper](#21-neat-paper)
  - [2.2 Evolving Deep Network Architectures](#22-evolving-deep-network-architectures)
  - [2.3 Learning Atari Games using NE](#23-learning-atari-games-using-ne)
- [3. Methods](#3-methods)
  - [3.1 Snake-NE](#31-snake-ne)
  - [3.2 NEAT Setup](#32-neat-setup)
  - [3.3 Input Strategies](#33-input-strategies)
  - [3.4 Ablation Experiments](#34-ablation-experiments)
- [4. Do It Yourself](#4-do-it-yourself)
- [5. Results](#5-results)
  - [5.1 Input Strategy Experiments](#51-input-strategy-experiments)
  - [5.2 Ablation Experiments Results](#52-ablation-experiments-results)
- [6. Discussion](#6-discussion)
  - [6.1 Limitations](#61-limitations)
  - [6.2 Future Work](#62-future-work)
- [7. Conclusion](#7-conclusion)


---

# 1. Introduction
Welcome to the handbook for our educational project on the neuroevolution snake game. This guide aims to provide both introductory and detailed information about evolutionary algorithms, specifically focusing on the NeuroEvolution of Augmenting Topologies (NEAT) method.

## 1.1 Learning Outcomes [come back to this as a last thing]
After working through our handbook, we expect you to have the following know-how's (please rephrase):
* have a solid understanding of Neurovolution's origins
* understand how NE works in general
* understand the NEAT algorithm, in full
* distinguish between variants of the method
* understand NE's limitations
* thus, know when NE is useful and when not
* understand the proposed enhanced methods
* be able to apply the algorithm yourself (maybe we should have a template / empty ish file for use in other cases)


## 1.2 Project Overview
Our project, named Snake-NE, aims to simplify the learning of NeuroEvolution (NE), a method that blends evolutionary algorithms with machine learning. We teach the NEAT method to evolve both the weights, biases, and topology of a neural network that learns to play the Snake game.

Now, we will more clearly introduce our project.
In short, NE aims to improve the computation efficiency and performance of a neural network by not only changing the weights of the network, but also evolve the network's architecture using evolutionary algorithms. It simulates the process of natural evolution to automatically design and train neural networks without the need for gradient-based methods. 
Let's begin by clearly introducing all of these mentioned terms, to offer us a solid ground for the rest of this lesson.


## 1.3 Problem Statement [do we even need this here?!]
Our project, Snake-NE, aims to present and teach NeuroEvolution (NE) in an immersive, interactive manner, having both a guide and practical coding experiments that students can run. We will do so by teaching the NeuroEvolution of Augmenting Topologies (NEAT) method to evolve both the weights, biases, and topology of a neural network that learns to play the Snake game.

## 1.4 Definitions
**Neural Networks**
Neural networks are computational models inspired by the human brain, consisting of interconnected nodes (neurons) that process data and learn to recognize patterns. They are widely used in machine learning for tasks such as image and speech recognition.

**Reinforcement Learning**
Reinforcement learning (RL) is a type of machine learning where agents learn to make decisions by performing actions in an environment to maximize cumulative rewards. It involves learning optimal policies through trial and error interactions.

**Evolutionary Algorithms**
Evolutionary algorithms (EAs) are optimization methods inspired by natural selection, where populations of candidate solutions evolve over generations. These algorithms use selection, crossover, and mutation to improve solutions iteratively, finding optimal or near-optimal solutions for complex problems.

**NEAT** [cuz that's what we use here]
NEAT is a specific NeuroEvolution method that evolves both the weights and the topology of neural networks. It starts with simple networks and complexifies them over generations through the addition of nodes and connections, preserving innovation through speciation.


## 1.5 Motivation [why is NE even useful?]
Snake-NE offers several benefits as an educational tool, making learning more engaging and accessible by using the simple yet fun Snake game to demonstrate complex neuroevolution concepts.

*hint to the extensive literature review // talk here about then to use it briefly. 

---



# 2. Literature Review
In this section, we review relevant literature that underpins our project and provides context for our methods and experiments.

## 2.1 NEAT paper
* first start off with the abstract re-worded
* describe the aim  / promise

This is the founding paper of the NeuroEvolution of Augmenting Topologies (NEAT) method. It shows its origins by first describing and then presenting the issues with the previous Topology and Weight Evolving Artificial Neural Networks (TWEANNs) method and explaining how it tackes these issues, creating NEAT. Finally and most importantly it proves its computational advantage and tests this through a series of ablation studies. The increased efficiency is thus thanks to:
* 1. employing a principled (? the historical stuff) method of crossover of different topologies
  2. protecting structural innovation using speciation (the shared fittness?)
  3. incrementally growing from minimal structure

[maybe i should only introduce the problems here using also some figures and then the solutions etc we describe in out methods section anyway no?!]

* describe TWEANNs (maybe some of the background subchapters too....!?)
* describe the problems with those TWEANNs as well -- with them figures
* how does NEAT tackle them -- maybe just reference to our methods section


## 2.2 Evolving Deep Network Architectures
We explore the adaptation of NEAT for Neural Architecture Search (NAS) and discuss bilevel optimization, scalability, and adaptation to environmental constraints.
* some extended methods

## 2.3 Learning Atari Games using NE
We investigate the effectiveness of non-gradient-based evolutionary algorithms (EAs) for training deep neural networks on reinforcement learning tasks, such as Atari games.

Now, a burning question is: why bother when back-prop etc are like good enough no? (in NEAT paper as well check out...)
<img width="250" alt="image" src="https://github.com/d-mihaila/NC-Project/assets/53557315/ebb38733-1ee2-4729-8ef6-33a0c218e83e">


* basically *when* NE is good

---

# 3. Methods
This section details the setup and implementation of our Snake-NE project, including the NEAT algorithm and various experiments.

## 3.1 Snake-NE
The complete source code for our Snake-NE project, developed using the PyGame library, is available on GitHub. The NEAT algorithm was implemented using the neat-python package.

## 3.2 NEAT Setup
We describe the NEAT algorithm, including network initialization, genetic encoding, crossover strategy, speciation strategy, and mutation strategy.

## 3.3 Input Strategies
We investigate how different input features influence the snake's learning and strategies. This includes experiments with baseline, binary, and collision input strategies.

## 3.4 Ablation Experiments
We analyze the essential components of the NEAT algorithm by systematically removing key components and observing their impact on performance.

---

# 4. Do It Yourself
We prepared the google colab for you. Now, it's the time for you to have your first tries with the code, familiarise yourself with it before we go on implementing our methods. 

---

# 5. Results
This section presents the outcomes of our experiments, focusing on input strategy experiments and ablation experiments.

## 5.1 Input Strategy Experiments
We describe the results of various input strategy experiments, highlighting the best-performing networks and observed behaviors.

## 5.2 Ablation Experiments Results
We discuss the findings from our ablation experiments, illustrating the importance of key components in the NEAT algorithm.

---

# 6. Discussion
We reflect on the limitations of our project and propose future work to build on our findings.

## 6.1 Limitations
We acknowledge the constraints and limitations of our project, such as computational resources and visualization tools.

## 6.2 Future Work
We suggest potential directions for future research and improvements, including exploring new input strategies, fitness function design, and comparisons with reinforcement learning.

---

# 7. Conclusion
[To be written]

---

# 8. References


