# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py


"""

by Meghana Kumar
King's College London
20093575
"""

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import itertools



EPSILON = 0.001
FOOD_REWARD = 10
CAPSULE_REWARD = 20
GHOST_RADIUS = 2
EATABLE_GHOST_REWARD = 4
"""
Class to build a map representation of Pacman
Part of this code is adopted from MapAgent provided in KEATS
"""
class Map:
    def __init__(self, state):
        # Variables are initialized the grid for the map is built
        # by obtaining the height and width of the board
        corners = api.corners(state)
        self.height = -1
        self.width = -1
        self.getHeight(corners)
        self.getWidth(corners)
        self.grid = self.buildMap()

    def getHeight(self, corners):
        for i in range(len(corners)):
            if corners[i][1] > self.height:
                self.height = corners[i][1]
        self.height = self.height + 1
        return

    def getWidth(self, corners):

        for i in range(len(corners)):
            if corners[i][0] > self.width:
                self.width = corners[i][0]
        self.width = self.width + 1
        return

    def prettyDisplay(self):
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print self.grid[self.height - (i + 1)][j],
            # A new line after each line of the grid
            print
            # A line after the grid
        print

    # Set and get the values of specific elements in the grid.
    # Here x and y are indices.
    def setValue(self, x, y, value):
        self.grid[y][x] = value

    def buildMap(self):
        subgrid = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(0)
            subgrid.append(row)


        return subgrid




class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print "Starting up MDPAgent!"
        name = "Pacman"
        self.stateTransitions = {}
        self.stateRewards = {}
        self.utilities = {}
        self.GHOST_PUNISHMENT = -100
        self.GHOST_RADIUS_PUNISHMENT = -75
        self.DISCOUNT_FACTOR = 0.6

        # variables for use in assigning rewards to map locations. Here and elsewhere long iterables such as walls and
        # other map features are stored as sets for speedy iteration
        self.walls = set()
        self.grid = set()
        self.small_grid = False


    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        print "Running registerInitialState for MDPAgent!"
        print "I'm at:"
        print api.whereAmI(state)
        self.getStateTransitions(state)
        self.walls = set(api.walls(state))
        self.createMap(state)

    def getStateTransitions(self, state):
        """
        The State Transition function contains all the states accessible from the current state
        """
        stateDict = dict.fromkeys(self.stateRewards.keys())

        # Since it is a Non-Deterministic Stochastic model, each possible direction
        # Will be accompanied by two other directions left and right of it
        # That can be accessed with a smaller probability.
        # This is stored as a list where the first element is the intended direction
        # And the next two are right and left of it.

        for i in stateDict.keys():
            neighbors = self.neighbours(i)
            stateDict[i] = {'North': [neighbors[3], neighbors[0], neighbors[2]],
                            'South': [neighbors[1], neighbors[0], neighbors[2]],
                            'East': [neighbors[0], neighbors[3], neighbors[1]],
                            'West': [neighbors[2], neighbors[3], neighbors[1]],
                            }
            # If any wall is found for any of the states
            # This is replaced by the current state itself
            # As in, the same state is repeated

            for states in stateDict[i].values():
                for pos in states:
                    if pos in self.walls:
                        states[states.index(pos)] = i

        self.stateTransitions = stateDict


    def setRewards(self, state):
        # Rewards are set separately for food, ghosts, capsules
        food = set(api.food(state))
        ghosts_times = api.ghostStatesWithTimes(state)
        ghosts = api.ghostStates(state)

        capsules = set(api.capsules(state))

        # Initilizing rewards to -1
        self.stateRewards = {key: -1 for key in self.grid if key not in self.walls}

        # Initializing all utilities to 0
        self.utilities = {key: 0 for key in self.grid if key not in self.walls}
        # The constant values are set for the food and capsules
        foodDict = {k: FOOD_REWARD for k, v in self.stateRewards.items() if k in food}
        self.stateRewards.update(foodDict)
        capsuleDict = {k: CAPSULE_REWARD for k, v in self.stateRewards.items() if k in capsules}
        self.stateRewards.update(capsuleDict)

        # First, it is assessed if the ghost is not scared
        # This implies an active ghost
        # And hence a punishment value needs to be assigned
        # A slightly lower punishment value is assigned to ghost neighbors as well
        # Upto Radius 2
        for ghost, ghost_time in zip(ghosts, ghosts_times):
            if ghost[0] in self.stateRewards.keys():
                if ghost[1] == 0:
                    self.stateRewards[ghost[0]] = self.GHOST_PUNISHMENT
                    ghostNeighbours = self.ghostRadius(state, ghost[0], GHOST_RADIUS)

                    ghostRadius = {k: self.GHOST_RADIUS_PUNISHMENT for k, v in self.stateRewards.items() if k in ghostNeighbours}
                    self.stateRewards.update(ghostRadius)
                else:
                    # This part of the code executes only if it is not a small grid
                    if not self.small_grid:
                        # If the ghost is eatable
                        # The pacman is driven to see this as a reward
                        # The more time left in the eatable state
                        # The higher the reward
                        # The values are chosen such that even a high value of reward is still less
                        # Than the punishment of an active ghost
                        # So survival is more important than scoring
                        time_left = ghost_time[1]
                        CHASE_GHOST_REWARD = time_left*EATABLE_GHOST_REWARD
                        self.stateRewards[ghost[0]] = CHASE_GHOST_REWARD





    """ This part of the code is borrowed from https://github.com/Jay-Down/MDP-PacMan-Agent/blob/master/mdpAgents.py"""
    def ghostRadius(self, state, ghosts, r, next=None):
        """ Creates a radius r around ghosts that are considered dangerous """
        ghostLocs = api.ghosts(state)
        walls = set(api.walls(state))


        # generate neighbours of ghost locations on first call of function
        if next is None:
            next = []
            ghostNeighbours = self.neighbours(ghosts)
            ghostNeighbours = [i for i in ghostNeighbours if i not in walls]
            ghostNeighbours = [i for i in ghostNeighbours if i not in ghostLocs]

        # generate neighbours from the results of the function's previous pass
        if next:
            ghostNeighbours = [self.neighbours(i) for i in ghosts]
            ghostNeighbours = itertools.chain.from_iterable(ghostNeighbours)
            ghostNeighbours = [i for i in ghostNeighbours if i not in walls]
            ghostNeighbours = [i for i in ghostNeighbours if i not in ghostLocs]

        # return a final set of ghost neighbour locations if on the last pass
        if r == 1:
            next.append(set(ghostNeighbours))
            final = [list(i) for i in next]
            final = set(itertools.chain.from_iterable(final))
            return final

        # else decrement 'r' by one and call the function on itself
        else:
            r = r-1
            next.append(set(ghostNeighbours))
            return self.ghostRadius(state, set(ghostNeighbours), r, next)

    # The dictionary of a map is used to evaluate the state of the game
    def createMap(self, state):
        self.map = Map(state)

        for wall in self.walls:
            self.map.setValue(wall[0], wall[1], '|    |')

        corners = api.corners(state)
        BL = (0, 0)
        BR = corners[1]
        TL = corners[2]
        map_width = range(BL[0], BR[0])
        map_height = range(BL[1], TL[1])
        if len(map_width)==6:
            self.small_grid = True
            self.GHOST_PUNISHMENT = -50
            self.GHOST_RADIUS_PUNISHMENT = -25
            self.DISCOUNT_FACTOR = 0.6
        self.grid = set((x, y) for x in map_width for y in map_height)

        
    # This is what gets run in between multiple games
    def final(self, state):
        print "Looks like the game just ended!"
        print(self.small_grid)

    def valueIteration(self, state):
        """ implementation of the Value Iteration algorithm for solving an MDP """

        # Discount factor determined emperically i.e by trial and error
        # Average number of wins over 10 times for 25 games
        gamma = self.DISCOUNT_FACTOR

        # Reasonable error bound for termination condition
        epsilon = EPSILON * (1-self.DISCOUNT_FACTOR)/self.DISCOUNT_FACTOR

        # The different parts of the Bellman equation are calculated
        # By accessing the values available in dictionaries
        # For state, rewards, utilities
        states = self.stateTransitions  # set of states, S
        rewards = self.stateRewards  # set of rewards, R(s)
        utilities = self.utilities  # set of utilities initialised at 0, U

        # Loop until the break condition is met at convergence
        while True:
            # Delta tracks the variable
            delta = 0

            # The current utilitiy and the updated utility is compared
            for space, utility in utilities.items():
                U_old = utility
                iter_utils = {}

                # Bellman Equation for Maximum Expected Utility
                # This updates the new utility value
                for direction, state in states[space].items():

                    U_new = rewards[space] + gamma * (
                                0.8 * utilities[state[0]] + 0.1 * utilities[state[1]] + 0.1 * utilities[state[2]])
                    iter_utils[direction] = U_new

                # The maximum expected utility is found
                utilities[space] = max(iter_utils.values())

                # The delta value is calculated and checked for convergence
                delta = max(delta, abs(utilities[space] - U_old))
            if delta < epsilon:
                for state, utility in utilities.items():
                    val = '{:3.2f}'.format(utility)
                    self.map.setValue(state[0], state[1], val)

                self.map.prettyDisplay()
                return utilities
    # Function to set rewards and state transition function
    def prepareState(self, state):
        self.setRewards(state)
        if not self.stateTransitions:
            self.getStateTransitions(state)

    # Returns the four coordinate values neighboring a point
    def neighbours(self, current):
        (x, y) = current
        news = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]  # E, S, W, N
        return news

    # Once the utilities are calculated, the optimal state and the respective Direction
    # Needs to be calculated
    def getOptimalState(self, state, utilities):
        """
        Given the current position, and utilities
        The Direction that gives Max Expected Utility is calculated
        """

        current = api.whereAmI(state)
        # All the neighbors are extracted
        neighborStates = [i for i in self.neighbours(current) if i not in self.walls]
        # Their respective utilities are obtained
        utilitiesNeighbors = [utilities[i] for i in neighborStates]
        # The argument value of the maximum expected utility is
        optimalMove = neighborStates[utilitiesNeighbors.index(max(utilitiesNeighbors))]

        # The Direction of the Optimal move needs to be calculated
        direction = tuple(x - y for x, y in zip(current, optimalMove))
        if direction == (-1, 0):
            return Directions.EAST
        if direction == (1, 0):
            return Directions.WEST
        if direction == (0, -1):
            return Directions.NORTH
        if direction == (0, 1):
            return Directions.SOUTH

    def getAction(self, state):
        # Set the rewards for the state
        self.prepareState(state)
        utilities = self.valueIteration(state)
        directionOptimal = self.getOptimalState(state, utilities)

        legal = api.legalActions(state)

        return api.makeMove(directionOptimal, legal)



