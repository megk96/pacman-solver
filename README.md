# Pacman Solver

The repository contains three different implementations of solving Pacman 

The codebase is built upon the Berkley AI Pacman project. [The project can be found here.](http://ai.berkeley.edu/project_overview.html)

1. `classifierAgents.py` implements a Random Forest Classifier (Ensemble model of a Decision Tree Classifier) in a deterministic environment. 
2. `mlLearningAgents.py` implements a Q-Learning algorithm with Reinforcement Learning 
3. `mdpAgents.py` implements Value Iteration, an implementation of Markov Decision Process in a non-deterministic environment. 

Code can be run by `python pacman.py -p mdpAgent -l mediumClassic` as an example. 
