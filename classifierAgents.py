from pacman import Directions
from game import Agent
import api
import random
from math import sqrt
import numpy as np
import pandas as pd


# An additional class Decision Tree is written
# This class performs methods of a standard decision tree
# Min-set is the minimum number of instances in set to be able to make a split, default 5
class DecisionTree:
    def __init__(self, tree_data, tree_target, feature_index, indices, min_set=5):
        self.x, self.y, self.features, self.indices, self.min_set = [tree_data, tree_target, feature_index, indices,
                                                                     min_set]
        self.n = len(self.y)
        self.val = np.mean(tree_target.values[indices])
        self.score = float('inf')
        self.evaluateFeatures()
        self.f_idx, self.split = [0, 0]

    # Gini Impurity is used to calculate the purity of the split
    # G = 1 - sum(pc*pc) where pc is probability of each class
    # Lowest G is 0 for completely pure split
    def calculateGiniImpurity(self, left, right, tree_target):
        # Obtain distinct classes
        classes = np.unique(tree_target)
        # Gets an array of instances on left and right respectively.
        # n is total number of instances related to node
        n = len(left) + len(right)
        left_gini = 0
        right_gini = 0
        # Every class is included in probability
        for c in classes:
            # Counts the mapping of where target is equal to class
            # Calculates left gini and right gini separately
            # The actual gini value will be 1 - this value
            # The adjusted gini value will be a weighted sum of proportion of instances
            p1 = np.count_nonzero(tree_target[left] == c) / len(left)
            left_gini += p1 * p1
            p2 = np.count_nonzero(tree_target[right] == c) / len(left)
            right_gini += p2 * p2
        weighted_gini = (1 - left_gini) * (len(left) / n) + (1 - right_gini) * (len(right) / n)

        return weighted_gini

    # This method finds the best split given the feature
    def findBestSplit(self, feature):
        x, y = self.x.values[self.indices, feature], self.y.values[self.indices]
        self.n = len(y)
        sort_idx = np.argsort(x)
        # Sorts indexes for iteration
        sort_y = y[sort_idx]
        sort_x = x[sort_idx]
        # A minimum set of min_set instances are added by default
        # Hence range is N - min_set - 1 (accounting for index starting from 0)
        for i in range(0, self.n - self.min_set - 1):
            if i < self.min_set or sort_x[i] == sort_x[i + 1]: continue
            # The tree is separated into left or right split based on current instance
            lhs = np.nonzero(sort_x <= sort_x[i])[0]
            rhs = np.nonzero(sort_x > sort_x[i])[0]
            if rhs.sum() == 0: continue
            # The gini impurity is calculated for lhs and rhs with the sorted class labels
            gini = self.calculateGiniImpurity(lhs, rhs, sort_y)
            # We want to minimize gini, so we set new gini as the new score
            # f_idx is the feature index the split occurs at
            if gini < self.score:
                self.f_idx, self.score, self.split = feature, gini, sort_x[i]

    def evaluateFeatures(self):
        # For each subset of features, the best split among them is found
        for feature in self.features:
            self.findBestSplit(feature)
        # A base case for leaf node
        if self.score == float('inf'): return
        # X based on indices and the chosen index for split: f_idx
        x = self.x.values[self.indices, self.f_idx]
        # Left and right trees are determined
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        # Recursive function to get decision tree for lhs and rhs
        self.lhs = DecisionTree(self.x, self.y, self.features, self.indices[lhs])
        self.rhs = DecisionTree(self.x, self.y, self.features, self.indices[rhs])


# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        Agent.__init__(self)
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray

    def predict(self, node, x):
        # The node is a class of DecisionTree
        # The score is inf for leaf node, then the value is returned
        # Next node to traverse is found based on what the next split feature f_idx is
        # If value for feature selected is less than split, then lhs, else rhs
        # Recursive call to traverse the tree to find prediction
        if node.score == float('inf'): return node.val
        next_node = node.lhs if x[node.f_idx] <= node.split else node.rhs
        return self.predict(next_node, x)

    # This is an ensemble model for decision tree
    # Bagging is Bootstap Aggregation
    def predictBagging(self, x):
        # Get predictions for each tree
        predictions = [self.predict(tree, x) for tree in self.forest]
        # Returns the most popular value in the predictions
        return max(set(predictions), key=predictions.count)

    def createTree(self, num_features, sample_size=None):
        # This returns a random permutation of features of size num_features
        feature_index = np.random.permutation(self.data.shape[1])[:num_features]
        # By default we take the full training set, but we sample it with replacement
        if sample_size is None:
            sample_size = self.data.shape[0]
        # This is to ensure the trees are uncorrelated by making them train on different training sets
        random_index = np.random.choice(self.data.shape[0], size=sample_size, replace=True)
        # The data and target of the random indexes are passed
        tree_data = self.data.iloc[random_index]
        tree_target = self.target.iloc[random_index]
        # A Decision Tree is built with the sampled tree data and target
        return DecisionTree(tree_data, tree_target, feature_index, np.array(range(sample_size)))

    # A scratch implentation of Random Forest
    def scratchRandomForest(self, num_trees):
        # For Random Forest, we get the number of trees as input argument: 10 default
        # This uses a subset of features to create trees which are highly uncorrelated
        # Number of sub-features in each tree is determined by square root of the total number of features
        num_features = int(sqrt(self.data.shape[1]))
        # The forest is formed by creating a subtree for number of features
        self.forest = [self.createTree(num_features) for i in range(num_trees)]

    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.
        # The pandas dataframe is required
        self.data = pd.DataFrame(self.data)
        self.target = pd.DataFrame(self.target)
        # The Classifier is called and saved as self.forest
        # The Random Forest Classifier is coded from scratch
        # It is an ensemble model that works well for few data points (100-150)
        self.scratchRandomForest(num_trees=10)

    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)
        # The prediction is made by using the predictBagging method
        # The Classifier was earlier trained and stored in self.forest
        # self.forest is used in predictBagging
        prediction = self.predictBagging([features])
        move = self.convertNumberToMove(prediction)
        # Get the actions we can try.
        legal = api.legalActions(state)
        # STOP is removed from legal options
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        if move in legal:
            # If move obtained was legal, it is made
            return api.makeMove(move, legal)
        else:
            # Else, a random legal move is made
            return api.makeMove(random.choice(legal), legal)


### Citations
# Theory of Random Forest
# https://builtin.com/data-science/random-forest-algorithm
# https://towardsdatascience.com/understanding-random-forest-58381e0602d2
