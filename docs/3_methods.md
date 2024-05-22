# 3. Methods
In this section, we first explain our setup for Snake-NE and for the NEAT algorithm // specifically, the solutions that the NEAT algorithms promises which we only briedly mentioned in the literature review// . We then detail a series of experiments conducted to address our first subgoal: analyzing how different input features influence the snake's learning and strategies. Following this, we describe a second series of experiments aimed at identifying the key components of the NEAT algorithm and their roles in learning. 


## 3.1 Snake-NE
The complete source code for our Snake-NE project, developed using the PyGame library, is available on GitHub, alongside with the relevant references. The NEAT algorithm was implemented using the neat-python package.

### Game Environment
The goal of the snake game is to make a snake moving in a 2D space eat as many food blocks as possible without biting itself or colliding with a wall. The snake starts with a body length of one. Each time the snake eats an apple, its body grows with a length of one. The game ends if the snake collides with a wall or itself, or if it fails to eat an apple within 100 time steps.

In each simulation, the snake and an apple are randomly initialised on a 10x10 grid. Its initial direction is a vector pointing downward. The snake's movements are controlled by outputs from a dedicated neural network, which determines its next direction based on sensory inputs from the current game state.

### Neural Network
The 'brain' of the snake in our game is modeled using a simple Multilayer Perceptron (MLP). This neural network is designed with a set of hand-crafted input features that help the snake navigate the game, such as the distance from the tail-end to walls and to its own body. The output of the network corresponds to possible movement directions of the snake. We will discuss variations in inputs and outputs in the 'Input Strategies' section further ahead. 

### Visualisation Methods
We used several visualization techniques to both monitor and present the results of our experiments. We tracked the evolutionary progress across generations, focusing on metrics such as average fitness, the standard deviation of fitness, and the fitness of the top-performing genome. We also visualized the topologies of the most successful networks across runs to evaluate whether more complex or simpler networks were more effective. This analysis helped clarify how network structures evolved and their efficacy in tackling the assigned tasks.

Additionally, to analyze the behaviour of the snakes and the strategies they learned, we simulated a selected genome playing the Snake game, highlighting the neural network activations that influenced the snake's decisions. In Figure \ref{fig:nn+snake visualization}, we show these activations where excitatory connections are marked in orange and inhibitory ones in blue, with color intensity indicating the level of activation. To support our analysis on strategies and behaviours, we recorded specific simulations that best represented the observed behaviours.

<img width="733" alt="image" src="https://github.com/d-mihaila/NC-Project/assets/53557315/b3d32cc7-7e07-468f-b67e-904061ec2824">

## 3.2 NEAT Setup
We describe the NEAT algorithm, including network initialization, genetic encoding, crossover strategy, speciation strategy, and mutation strategy.

The neural network was evolved using the NeuroEvolution of Augmenting Topologies (NEAT) algorithm [Stanley2002]. NEAT is a method that not only adjusts the weights but also the *structure* of neural networks through evolutionary algorithms. It can add or remove nodes (neurons) and connections as part of the evolutionary process, allowing the network to adapt its complexity to the problem at hand. Our experiments typically used varying parameter settings, hence this description will remain general. 

We use an evolutionary process for a population of $N_{\text{pop}}$ neural network genomes. Each genome represents a potential solution, evolving over $N_{\text{gen}}$ generations. A genome's fitness was assessed based on the number of apples the snake consumes before the game ends, whether through collision or reaching the maximum time limit. To ensure reliability, the fitness score for each genome is calculated as the average outcome of 10 runs. The evolution was guided by the highest fitness achieved by any genome, ending the process when a genome reached the maximum possible fitness score of 100 or after completing the designated number of generations.

To ensure that each simulation was reproducible, we used a fixed seed value (42) for the random number generator. Furthermore, we implemented checkpoints every generation. This allowed us to restore a population from a checkpoint and further evolve or analyze it under controlled conditions to validate the reproducibility and reliability of the evolutionary outcomes.

### Network Initalisation

Following the recommendations from the original NEAT paper, we started with small initial networks, This idea is they key to gaining an advantage from the evolution of topology: it ensures that the system searches for the solutions in the lowest-dimensional weight space possible, resulting in dramatic performance gains. New structures are introduced incrementally as structural mutations occur, with only those structures that prove beneficial through fitness evaluations surviving. 

For this reason, we initialised the neural networks with an input layer consisting of the selected input features and an output layer consisting of the selected output features, starting without any hidden nodes to minimize initial complexity. Initial connections were randomly made for 50\% of the nodes, restricted to feed-forward connections to avoid loops. The nodes used a standard sigmoid activation function, and connection weights were initialised with a mean of 0.0 and a standard deviation of 1.0.

### Genetic Encoding

NEAT uses a straightforward method to outline each network's structure in the genetic code, detailing every connection and node. The genome acts as a linear map of the network's connectivity, listing all the connection genes that link two nodes together. These nodes include input, hidden, and output types. Each connection gene clearly defines the starting node (in-node), ending node (out-node), the strength of the connection (weight) and whether the connection is active (an enable bit).

Each new node or connection that appears in the genome also gets a unique innovation number which helps in identifying and tracking each connection. This innovation number helps in matching genomes that come from the same ancestor during the mixing of genes from different parents (crossover), ensuring that their connections are also correctly paired. This tracking is crucial for maintaining the structure and functionality of the network as it evolves.

### Crossover Strategy

In NEAT, when creating offspring, genes in both parent genomes that share the same innovation numbers (known as matching genes) are aligned. Genes that don’t align are called excess genes and represent structures absent in the other genome. During reproduction, matching genes are randomly selected from either parent, while all excess genes are always included from the fitter parent. If both parents are equally fit, excess genes are chosen randomly from both.

Throughout our simulations, we maintained a constant population size of $N_{\text{pop}}$ genomes. Our reproduction approach uses elitism and a survival threshold, where the top two individuals from each species automatically move to the next generation, and only the top 20\% of the remaining individuals, based on fitness, are allowed to reproduce. This strategy helps preserve successful genomes while introducing new genetic variations.

### Speciation Strategy

NEAT groups similar network architectures into species through a process called "speciation." This method protects new structural innovations by allowing them time to improve over generations before they must compete with the broader population. Speciation is based on a compatibility score calculated using the number of excess genes and the average weight differences of matching genes, including disabled ones. The formula used is:

\begin{equation} \label{eq: compatibility score}
    \delta = \frac{c_1E}{N} + c_2 \cdot \overline{W}
\end{equation}
where $c_1$ and $c_2$ are coefficients adjusting the importance of each factor, and $N$ is the number of genes in the larger genome.

Speciation involves calculating distances between genomes to determine groupings. Each species is represented by a random genome from the previous generation. A new genome is placed in the first species where it matches the representative genome; if no match is found, a new species is formed with the genome as its representative. We set the compatibility threshold at $T_c=3.0$ to classify genomes into species based on their similarities. To encourage diversity and prevent any one species from becoming dominant, a species can only persist for up to 20 generations without improvement before being considered stagnant.

### Mutation Strategy

In NEAT, genomes can undergo mutations each generation that may affect weights, biases, nodes, or connections, and these types of mutations can happen simultaneously. \textit{Connection mutations} create a new link between two previously unconnected nodes with a randomly assigned weight. This new connection is added to the end of the genome and is given the next available innovation number. \textit{Node mutations} involve disabling an existing connection and inserting a new node in its place, effectively splitting the connection into two: one leading into the new node with a weight of 1, and another leaving it with the original weight. This approach reduces the impact of mutations, allowing the network to integrate new changes smoothly.

The mutation rate for existing connection weights is set with a specific value $\mu_{\text{weight}}$ with a mutation power\footnote{Mutation power refers to the standard deviation of the zero-centered Gaussian distribution from which a bias mutation is drawn} of 0.5, and weights could vary between -30 and 30. Likewise, the bias mutation rate is denoted by $\mu_{\text{bias}}$, with the same mutation power of 0.5. The mutation rate of both adding and removing nodes is defined by $\mu_{\text{node}}$. The mutation rate of both adding and removing connections is defined by $\mu_{\text{connect}}$. 

For the weights and biases, the $\mu$ mutation rate has a replacement rate $\eta$ counterpart. When a $\mu$ mutation occurs, a randomly drawn sample from a zero-centered Gaussian distribution with a std equal to the mutation power is \text{added} to the existing value. However, $\eta$ corresponds to the rate at which a mutation can completely replace a value with a newly initialized value

## 3.3 Input Strategies
We investigate how different input features influence the snake's learning and strategies. This includes experiments with baseline, binary, and collision input strategies.

Our first subgoal explores how various inputs to the neural network influence the snake's learning and strategies in the Snake-NE game. Determining the optimal inputs for the snake involves finding a balance between simplicity and providing sufficient information for the snake to develop advanced game-playing techniques. We tested the following input features:

**Baseline Snake**
The Baseline Snake uses the input/output features from our initial model (illustrated in Figure \ref{fig:nn+snake visualization}). Input features include distances from the snake's head to the walls in all four compass directions (N, W, S, E), from the head to the tail, and to an apple, also segmented by NSWE. These distances were normalized to fall within the range [0,1] to match node activations.

**Collision Snake**
This version combines the separate wall and tail inputs into a single input that provides information about nearby collision threats, effectively merging wall and tail inputs.

**Optimised Snake**
In this case, our snake is merging the proximity information of the walls and the obstacle, only keeping the closest value (biggest current threat), in each of the cardinal directions. This is called 'optimised' due to the lower number of input nodes, making the computation more efficient.

**Snake's frame of reference**
We noticed that not 4 directions (N, S, E, W) are relevant to our computation. The snake will never turn 'backwards', to the direction it just came from in the previous 'timestep'. This would lead to it eating itself, which means losing the game. Thus, we reduced the amount of input information about the surroundings to only 3 directions: left, right, forwards, relative to its current head orientation. 

**Optimised snake's frame of reference**
This last version is a combined one from the last 2 described ones. It has as input the left, right and forwards distance to the nearest obstacle / wall in that direction, apple and its tail. This is the highest performing version of our project. Futher details will be discussed in the Results section. 

The following experiments that we are describing are aimed to analyze the qualitative behaviour of the best strategies that emerged from specific inputs. This means we were primarily interested in finding the best result obtained rather than increasing the average fitness. This approach is based on the understanding that evolutionary algorithms (EAs), when given sufficient evaluations, will eventually converge to a global optimum \cite{eiben1991global}.

To ensure we achieved converged learning strategies, we ran five extended runs of 1000 generations each, with a population size of 2000 for each experiment. We tracked the highest fitness levels achieved and used the average best fitness across these runs to determine if the algorithm had consistently converged on an optimal strategy. We further analyzed these runs by re-running simulations of the top-performing genome from various generations, observing how fitness changes influenced learned behaviours both with and without obstacles. For each experiment, we also performed a run without obstacles in which the time to eat a fruit was lowered to 20 time steps to check whether we could observe the adaptation behaviour mentioned in the Literature review. We recorded the simulations that best represented the observed behaviour patterns.


For this set of experiments, the parameters were kept fixed to the values found in Table \ref{tab:design experiments}

\begin{table}[hbt]
    \centering
    \begin{tabular}{|c|c|c|c|c|c|l|c|l|c|c|c|} \hline 
         $N_{\text{runs}}$&  $N_{\text{gen}}$&  $N_{\text{pop}}$&  $\mu_{\text{weight}}$&  $\mu_{\text{bias}}$&  $\mu_{\text{node}}$ &$\mu_{\text{connect}}$&    $\eta_{\text{weight}}$&$\eta_{\text{bias}}$&  $c_1$&  $c_2$& $T_c$\\ \hline 
         5&  1000&  2000&  0.8&  0.7&  0.5&0.7&    0.1&0.1&  1.0&  0.5& 3.0\\ \hline
    \end{tabular}
    \caption{Parameters used for the input strategy experiments}
    \label{tab:design experiments}
\end{table}



## 3.4 Ablation Experiments
We analyze the essential components of the NEAT algorithm by systematically removing key components and observing their impact on performance.

% Jasper u have the mic here
Our second objective is to analyze the essential components of the NEAT algorithm and their roles in learning. Inspired by the original NEAT paper \cite{stanley2002evolving}, we ran ablation experiments by systematically removing key components of NEAT to see their impact on performance. We chose these experiments because they are binary in nature, allowing students to learn about critical components of NEAT and verify their importance without dealing with tricky parameter tuning.

These experiments focus on the removal of crucial components of NEAT: innovation numbers that ensure accurate crossover, a speciation mechanism that protects innovations, and a minimal-network strategy that simplifies the search space. Additionally, we compared NEAT to a standard neuroevolution method that only evolves weights and biases while keeping the network topology constant.

For these experiments, we measured the average fitness of the population over five runs to confirm our findings. This method allows for straightforward evaluation, helping students understand the significance of each component without complex parameter adjustments.

Details on the implementation of each experiment can be found in Table \ref{tab:ablation}.
\begin{table}[hbt]
    \centering
    \begin{tabular}{|l|l|}
    \hline
    \textbf{Experiment Name} & \textbf{Implementation Details} \\ \hline
    Fixed Topology NEAT & Set $\mu_\text{node} = 0$ and $\mu_\text{connection} = 0$ \\ \hline
    Nonspeciated NEAT & Set compatibility threshold to an arbitrarily high value $T_c = 10^9$ \\ \hline
    Nonmating NEAT & Set fraction for each species allowed to reproduce to 0 \\ \hline
    Initial Random NEAT & Initialize hidden nodes randomly from Uniform distribution between 0 and 20 \\ \hline
    \end{tabular}
    \caption{Ablation experiments and their implementations}
    \label{tab:ablation}
\end{table}





---

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; text-align: left;">
    <a href="2_literature_review.md" style="text-decoration: none; font-size: 1.2em; border: 1px solid #ccc; padding: 10px; display: inline-block;">&laquo; Previous: Literature Review</a>
  </div>
  <div style="flex: 1; text-align: right;">
    <a href="4_do_it_yourself.md" style="text-decoration: none; font-size: 1.2em; border: 1px solid #ccc; padding: 10px; display: inline-block;">Next: Do It Yourself &raquo;</a>
  </div>
</div>
