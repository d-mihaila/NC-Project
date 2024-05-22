
# 2. Literature Review 
[NOTE sorry but here i kinda used quite a lot of AI to rather get quantity rn to see the whole website play out and will make finetuning afterwards.] 
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
NEAT is just an example of algorithm that describes how these TWEANNs should be evolved.
* describe the problems with those TWEANNs as well -- with them figures
The problems with the other TWEANNs include: Competing Conventions, Protecting Innovation with Speciation, Initial Populations and Topological Innovation. Let's now describe each of them.

**Competing Conventions**
The Competing Conventions Problem, also known as the Permutations Problem, is a significant challenge in neuroevolution. It arises when multiple ways exist to represent a solution in a neural network, causing different genomes to encode the same solution differently. This leads to problematic crossovers during reproduction, often resulting in offspring that lose crucial information.

In the case of a simple three-hidden-unit network, as depicted in Figure 1, the hidden neurons (A, B, and C) can be permuted in 3! = 6 different ways, all representing the same functional solution. When these different permutations are crossed, critical information can be lost. For example, crossing genomes [A, B, C] and [C, B, A] might produce [C, B, C], which lacks some of the original information.
<img width="796" alt="image" src="https://github.com/d-mihaila/NC-Project/assets/53557315/2b8433af-7ec7-4362-a898-f97ab540b2ec">

Solution: Historical Marking!

**Protecting Innovation with Speciation**
In TWEANNs, innovation occurs through structural mutations such as adding new nodes or connections to networks. However, these changes often initially decrease the network's fitness. For instance, a new node introduces a nonlinearity or a new connection might reduce fitness before its weight is optimized. Since immediate benefits from these structural changes are unlikely, the innovations might not survive long enough to demonstrate their potential value.

To address this, innovations must be protected to allow sufficient time for optimization. One approach is to add nonfunctional structures, hoping they eventually become useful. However, this can lead to extraneous parameters if these structures never integrate into the functional network.


Solution: Explicit Fitness Sharing


**Initial Populations and Topological Innovation**
In many TWEANN systems, the initial population consists of randomly generated topologies, ensuring diversity from the start. However, this approach presents several issues. Random initial populations often include infeasible networks with no valid paths from inputs to outputs, which need to be eliminated over time. More critically, random starting points rarely lead to minimal solutions. These populations typically contain many unnecessary nodes and connections that have not been justified by any evaluation process.

Such extraneous structures must be removed, which is inefficient. Larger networks might dominate due to their high fitness, regardless of their unnecessary complexity. To counteract this, some TWEANNs penalize larger networks in the fitness function. However, determining the appropriate penalty is challenging and problem-specific, making this method ad hoc and potentially altering the intended evolutionary dynamics.

Solution: starting with minimal population

Finally, joining all the 3 solutions together, we obtain this relationship: 
<img width="796" alt="image" src="https://github.com/d-mihaila/NC-Project/assets/53557315/3aadf5e6-9c7f-46af-af97-cfee932d6dd8">


* how does NEAT tackle them -- maybe just reference to our methods section
<img width="796" alt="image" src="https://github.com/d-mihaila/NC-Project/assets/53557315/a9886265-c498-455d-8821-44afedc9e1de">
-- description --
<img width="796" alt="image" src="https://github.com/d-mihaila/NC-Project/assets/53557315/dc8364dd-d096-4116-907c-93c5efcababa">
-- description --

They also conducted the Ablation experiemnts we will describe in the following chapters as well...!?

Maybe add here some description about conclusion of when this NE is good .... // link to the 3rd paper 




## 2.2 Evolving Deep Network Architectures
We explore the adaptation of NEAT for Neural Architecture Search (NAS) and discuss bilevel optimization, scalability, and adaptation to environmental constraints.

Risto Miikkulainen, a co-creator of NEAT, recently explored the adaptation of NEAT for Neural Architecture Search (NAS) in his paper, "Evolving Deep Neural Networks" \cite{miikkulainen2024evolving}. This approach evolves generations of Deep Neural Networks (DNNs) through processes like crossover and mutations, similar to those used in NEAT. However, the key difference is that each link in the genome represents entire neural network layers and their connections, rather than just individual nodes and connections. The fitness of each network is evaluated based on its training performance over a limited set number of epochs using gradient descent.

### Adaptation
The study also noted an interesting adaptation to environmental constraints: the necessity to only partially train networks due to limited resources caused the evolutionary process to bias the evolution toward fast learners rather than top performers. This reveals that evolution can be steered toward goals other than sheer accuracy, such as reducing training time, execution speed, or the memory footprint of the network. This could be extremely useful for edge devices or specialized applications where the environment dictates the unique architectural needs.

We would like to highlight this aspect of neuroevolution in our product as well. We will do this by adding an experiment where the snake has a limited time to eat an apple, which should encourage the development of faster, more efficient snakes rather than just longer ones. This will help students see how adapting to constraints can direct evolutionary processes in neural networks.

### Other
Two additional important concepts introduced in this paper are: Bilevel optimization and scalability. We did not introduce them in our code in this project due to time contraints but are a great direction to go into for an extension. Feel free to skip this section if you are only interested in the concepts relevant to our implementation.

#### Bilevel Optimization
Neural Architecture Search typically focuses on finding the best network topology and hyperparameters. Traditionally, this search is done without gradient information, relying on manual adjustments or scalable architectures like EfficientNet \cite{tan2019efficientnet}. Manual tuning is often limited by the complex interactions between parameters, covering only a small portion of the possible configurations.

We currently handle a simplified parameter set with ten free variables, though NEAT supports up to 38. Managing these parameters manually is challenging and inefficient. Implementing bilevel optimization, as discussed in the literature review, could automate the tuning of NEAT's hyperparameters through an evolutionary process. This approach would require access to multiple CPUs since each optimization cycle would involve running a series of genetic algorithm simulations.

A solution for this is bilevel optimization. This method uses a high-level evolutionary process to search for network parameters more effectively. This method can be applied not only in NAS but also in parameter searches for other complex systems \cite{liang2015evolutionary}. The paper presents a hybrid model where neuroevolution manages the high-level optimization, and either backpropagation or further neuroevolution fine-tunes the weights. This dual strategy provides a valuable learning opportunity for students, demonstrating how integrating different methods can give better results and that solutions are not restricted to a single technique.

In our Snake-NE project, while we face challenges similar to those in manual tuning, we plan not to adopt this method directly. We believe that from an educational standpoint, achieving the best performance parameters is less critical as long as the learning experience remains effective. However, we address the challenges of manual tuning and present bilevel optimization as a practical alternative to consider.

#### Scalability
One of the primary advantages of using neuroevolution for Neural Architecture Search (NAS) is its scalability \cite{salimans2017evolution}. Once established, neuroevolution can function on a large scale and operate in parallel with little human intervention, automatically exploring the best architectures across vast parameter spaces. The paper points out that because Evolutionary Algorithms are less prone to getting stuck in local minima, they are able to discover networks that surpass manually created designs or a single design that has stopped improving due to local minima \cite{papavasileiou2021systematic}.

However, the elephant in the room is the significant computational resources required for this process, as each fitness evaluation involves training a deep neural network. Despite this, the authors mention that with the increasing availability of extensive computational resources through cloud and grid computing, the evolutionary optimization of neural networks is becoming a more feasible approach for the future. This method can fully leverage such resources due to its high degree of parallelizability.

Due to the high computational costs and our aim to stay focused on specific topics, we have decided not to include this approach in our project. Nevertheless, we addressed this topic because its scalability is crucial for real-world applications beyond simple game models. If more computational resources were available, one could consider evolving (deep) convolutional networks trained directly on game pixel data, which would introduce less human bias.


## 2.3 Learning Atari Games using NE
We investigate the effectiveness of non-gradient-based evolutionary algorithms (EAs) for training deep neural networks on reinforcement learning tasks, such as Atari games.

Now, a burning question is: why bother when back-prop etc are like good enough?

### Learning Atari Games using NE
The concisely named paper "Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning" explores whether non-gradient-based evolutionary algorithms (EAs) can be effective at the scale of deep neural networks (DNNs) \cite{such2017deep}. Instead of using NEAT, it evolves the weights and biases of a fixed DNN architecture with a simple, gradient-free, population-based genetic algorithm (GA). This method has shown strong performance on challenging deep reinforcement learning (RL) problems, including Atari games and humanoid locomotion. What stands out in this research is the scale: the team managed to evolve a network with over four million parameters, achieving state-of-the-art results in RL challenges using just a high-end desktop computer.

#### Zero-gradient method
Contrary to expectations, GAs have proven to be a competitive choice for RL, performing comparably to well-established methods like A3C, DQN, and ES. This suggests that in RL, where obtaining reliable gradient information is particularly difficult, EAs can be a valuable tool. The gradient in DQN can be biased, and in the complex, hard-to-navigate landscape of RL, EAs offer a viable alternative. This insight reinforces the notion that following the gradient isn't always the best approach for optimizing performance.

While we initially planned a brief comparison with other RL methods, this paper might serve as a more engaging follow-up for interested readers, allowing us to focus more on other areas. The key takeaway for students is understanding when NE methods are advantageous: particularly when gradient information is unreliable, unavailable, or when sparse computational resources exist.

#### Random Search
We were also curious about how NE compares to simple random search strategies. Surprisingly, the study found that while the GA consistently outperformed random search, random search still surpassed more advanced deep RL algorithms in some Atari games. This suggests that local optima, saddle points, noisy gradients, or other obstacles can hinder progress in methods based on backpropagation.

However, the broad applicability of EAs can lead to them being misused as a 'one-size-fits-all' solution for problems that might be more effectively addressed by using a heuristic based on the mathematical structure of the problem. For many problems where quality gradient information is available, gradient-based optimization algorithms will typically outperform evolutionary algorithms.

This highlights another domain where NE excels: although they have a wide range of applications, EAs particularly shine when augmented with domain-specific knowledge. Essentially, an EA is a form of biased random search, and the design of this bias falls to the researcher. We plan to demonstrate this by explaining how incorporating domain knowledge affects performance and what happens when it is removed, through ablation experiments, which will be detailed later.


* basically *when* NE is good ----- make a note // describe this better later on....!?


---
