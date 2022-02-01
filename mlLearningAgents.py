# python 2.7 script
# Chen/13-apr-2017
# based on the script written by Parsons
#
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game

# For Counter, since it allows a class to be an index for hashing
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # A Counter is used to store the Q value look up table which takes Q(s, a) as input
        # where s is the state and a is the action
        self.Q = util.Counter()
        # The previous score is also tracked in a variable
        self.score = 0
        # We need to keep track of the last state and last action for calculations of Q value and updating it
        self.last_state = None
        self.last_action = None
        # We need to keep track of the current state for getting the action with maximum reward
        self.current_state = None
        # The current reward is also stored in a variable
        self.reward = None



    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    # Accessor function to return the episodes so far
    def getEpisodesSoFar(self):
        return self.episodesSoFar

    # Accessor function to get the total number of training runs
    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    # Epsilon is used for the greedy aproach to balance between exploration and exploitation
    def setEpsilon(self, value):
        self.epsilon = value
    # Alpha is the learning rate that determines the degree to which the q value changes in each updation
    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
    # Gamma is the discount factor
    # While calculating the utility/reward of the next states, since it is only a hypothetical, a discount factor is applied
    # 0.8 means that the utility of next state is considered 80% as valuable as it is
    def getGamma(self):
        return self.gamma

    # functions for calculation
    # get Q(s,a)
    # This is the function to retrieve the value of the state and action from the lookup table
    def getQValue(self, state, action):
        return self.Q[(state,action)]

    # return the maximum Q of state
    # The Q values are measured for the current state and all the possible legal actions for it
    # This maximum of these q values is returned
    def getMaxRewardAction(self, legal):
        q_values = []
        for move in legal:
            q = self.getQValue(self.current_state, move)
            q_values.append(q)
        if len(q_values) == 0:
            return 0
        return max(q_values)

    # This is the main method for updating the Q value

    # This is the part of the learner where the Q values are updated

    # First, the maxReward is obtained based on the legal moves that are allowed from the current state
    # Then the q value for the last stata and last action are obtained
    # The q value is then updated using the formula Q[s, a] =  Q[s, a] + alpha*(reward + gamma*maxReward - Q[s, a])
    # This uses alpha as learning rate and gamma as discount rate and the max reward calculates the expected reward in the best possible action for the current state
    # The alpha controls the learning rate - the degree of change of the prev value of Q[s, a]
    # The discount rate is the degree of importance of the max reward of the actions of the current state, since it is a probability
    # and all possible actions and consequences would not be fully explored at the time
    def updateQValue(self, legal):
        maxReward = self.getMaxRewardAction(legal)
        q = self.getQValue(self.last_state,self.last_action)
        self.Q[(self.last_state,self.last_action)] = q + self.alpha*(self.reward + self.gamma*maxReward - q)

    # This is the core qLearning algorithm

    # This is the part of the learner where the appropriate action is chosen based on the Q values
    # For the legal actions of the current state, the Q value is found
    # This is stored in a Counter
    # The argument for which the q value is maximum - the action is returned
    def qLearning(self, legal):
        action_values = util.Counter()
        for action in legal:
          action_values[action] = self.getQValue(self.current_state, action)
        return action_values.argMax()

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):
        # THe current state is stored as an instance variable
        self.current_state = state

        # All the legal actions for the current state are fetched
        legal = state.getLegalPacmanActions()

        # STOP is not considered an option as we want the game to continue
        # Hence it is removed from the legal actions list
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # The reward is calculated as the current score subtracting the previous score that is stored in self.score
        # This difference in score represents the "reward" or "utility" for being in this current state from the previous state
        self.reward = state.getScore()-self.score

        # If the last state is None, it means that we are in the first move, and we take a random step, there is nothing to update
        if self.last_state is not None:
            self.updateQValue(legal)


        # Printing values to keep track of the iterations
        # Please uncomment these if anything needs to be verified
        #print "Legal moves: ", legal
        #print "Pacman position: ", state.getPacmanPosition()
        #print "Ghost positions:", state.getGhostPositions()
        #print "Food locations: "
        #print state.getFood()
        #print "Score: ", state.getScore()

        # This is the epsilon greedy approach
        # The util flipCoin function is used which selects a binary state
        # with the probability of epsilon
        # This is to encourage exploration to epsilon fraction of times
        # For the rest of the times, the algorithm exploits - using qLearning
        if util.flipCoin(self.epsilon):
            pick =  random.choice(legal)
        else:
            pick =  self.qLearning(legal)

        # The attributes are updated for the next run
        # Score is set as the current score
        # Last state is the current state
        # Last action is the one that ended up getting "pick"ed in this run
        self.score = state.getScore()
        self.last_state = state
        self.last_action = pick

        return pick

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        # This is the terminal state so we need to update the values

        #The final reward is updated using the current score - previous score formula
        self.reward = state.getScore()-self.score

        # Since there is no action from the last state - the max reward is set to 0, and the Q values are updated.
        q = self.getQValue(self.last_state, self.last_action)
        self.Q[(self.last_state, self.last_action)] = q + self.alpha * (self.reward + self.gamma * 0 - q)

        # reset attributes
        self.score = 0
        self.last_state = None
        self.last_action = None
        self.current_state = None


        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() % 100 == 0:
            print "Completed %s runs of training" % self.getEpisodesSoFar()

        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)
