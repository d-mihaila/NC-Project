# NeuroEvolution Snake game
Authors: Jasper, Daria

Welcome to our educational material tackling. We are here to guide you through the theory, practice and the rest of in and outs of NeuroEvolution. Specifically, applying the NeuroEvolution of Augmenting Topologies (NEAT) [Stanley2002] algorithm to play the snake game. We will first provide you with the background to understanding what's going on. Then, having a critical toolbox ready in your mind, you will get to experiment on your own with our variations of the implementation using our GoogleColab files. 
Then, finally, we will once again evaluate the importance, use and benefits / drawbacks of Neuroevolution together and propose some future work. 

Let's Start! 

## Table of Contents

- [1. Introduction](1_introduction.md)
  - [1.1 Learning Outcomes](1_introduction.md#11-learning-outcomes)
  - [1.2 Project Overview](1_introduction.md#12-project-overview)
  - [1.3 Problem Statement](1_introduction.md#13-problem-statement)
  - [1.4 Definitions](1_introduction.md#14-definitions)
  - [1.5 Motivation](1_introduction.md#15-motivation)
- [2. Literature Review](2_literature_review.md)
  - [2.1 NEAT paper](2_literature_review.md#21-neat-paper)
  - [2.2 Evolving Deep Network Architectures](2_literature_review.md#22-evolving-deep-network-architectures)
  - [2.3 Learning Atari Games using NE](2_literature_review.md#23-learning-atari-games-using-ne)
- [3. Methods](3_methods.md)
  - [3.1 Snake-NE](3_methods.md#31-snake-ne)
  - [3.2 NEAT Setup](3_methods.md#32-neat-setup)
  - [3.3 Input Strategies](3_methods.md#33-input-strategies)
  - [3.4 Ablation Experiments](3_methods.md#34-ablation-experiments)
- [4. Do It Yourself](4_do_it_yourself.md)
- [5. Results](5_results.md)
  - [5.1 Input Strategy Experiments](5_results.md#51-input-strategy-experiments)
  - [5.2 Ablation Experiments Results](5_results.md#52-ablation-experiments-results)
- [6. Discussion](6_discussion.md)
  - [6.1 Limitations](6_discussion.md#61-limitations)
  - [6.2 Future Work](6_discussion.md#62-future-work)
- [7. Conclusion](7_conclusion.md)

---
## Acknowledgements 
* the NEAT paper
* the people with the code
* inge

---
## References
1. Miikkulainen, R., J. Liang, E. Meyerson, et al. 2024. “Evolving Deep Neural Networks.” In Artificial Intelligence in the Age of Neural Networks and Brain Computing, 269–287. Elsevier.

2.  Such, F. P., V. Madhavan, E. Conti, J. Lehman, K. O. Stanley, and J. Clune. 2017. “Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning.” arXiv preprint arXiv:1712.06567.

3.  Eiben, A. E., E. H. Aarts, and K. M. Van Hee. 1991. “Global Convergence of Genetic Algorithms: A Markov Chain Analysis.” In Parallel Problem Solving from Nature: 1st Workshop, PPSN I Dortmund, FRG, October 1–3, 1990 Proceedings 1, 3–12. Springer.

4.  Ehlis, T., J. Hattan, and D. Sikora. 2000. “Application of Genetic Programming to the Snake Game.” Gamedev.Net 175.

5.  Vignesh Kumar, K., R. Sourav, C. Shunmuga Velayutham, and V. Balasubramanian. 2020. “Fitness Function Design for Neuroevolution in Goal-Finding Game Environments.” In Advances in Computational Collective Intelligence: 12th International Conference, ICCCI 2020, Da Nang, Vietnam, November 30–December 3, 2020, Proceedings 12, 503–515. Springer.

6. Gaier, A., and D. Ha. 2019. “Weight Agnostic Neural Networks.” Advances in Neural Information Processing Systems 32.

7.  Tan, M., and Q. Le. 2019. “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.” In International Conference on Machine Learning, 6105–6114. PMLR.

8. Liang, J. Z., and R. Miikkulainen. 2015. “Evolutionary Bilevel Optimization for Complex Control Tasks.” In Proceedings of the 2015 Annual Conference on Genetic and Evolutionary Computation, 871–878.

9. Salimans, T., J. Ho, X. Chen, S. Sidor, and I. Sutskever. 2017. “Evolution Strategies as a Scalable Alternative to Reinforcement Learning.” arXiv preprint arXiv:1703.03864.

10. Papavasileiou, E., J. Cornelis, and B. Jansen. 2021. “A Systematic Literature Review of the Successors of 'Neuroevolution of Augmenting Topologies'.” Evolutionary Computation 29 (1): 1–73.

---

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; text-align: right;">
    <a href="1_introduction.md" style="text-decoration: none; font-size: 1.2em; border: 1px solid #ccc; padding: 10px; display: inline-block;">Next: Introduction &raquo;</a>
  </div>
</div>


