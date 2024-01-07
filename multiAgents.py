# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent
from pacman import GameState


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()


def better_Evaluation_Function(currentGameState: GameState):

    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()

    capsule_count = len(currentGameState.getCapsules())
    closest_food_distance = 1
    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)

    game_score = currentGameState.getScore()

    if food_count > 0:
        food_distances = []
        for food_position in food_list:
            food_distances.append(manhattanDistance(pacman_position, food_position))

        closest_food_distance = min(food_distances)

    for ghost_position in ghost_positions:
        ghost_distance = manhattanDistance(pacman_position, ghost_position)

        if ghost_distance < 2:
            closest_food_distance = float('inf')

    features = [1.0 / closest_food_distance, game_score, food_count, capsule_count]

    weights = [10, 200, -100, -10]

    features = np.array(features)
    weights = np.array(weights)

    return np.dot(features, weights)



class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):

        best_eval, best_action = self.minimax(gameState, 5, float('-inf'), float('inf'), True)

        chosen_action = random.choice(best_action)

        return chosen_action

    def minimax(self, gameState: GameState, depth, alpha, beta, maximizing_player):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return better_Evaluation_Function(gameState), None

        if maximizing_player:
            max_eval = float('-inf')
            best_action = []
            for action in gameState.getLegalActions(0):
                if action == "Stop":
                    continue
                evaluation, _ = self.minimax(gameState.generateSuccessor(0, action),
                                             depth, alpha, beta, False)
                if evaluation > max_eval:
                    max_eval = evaluation
                    temp_list = []
                    temp_list.append(action)
                    best_action = temp_list
                elif evaluation == max_eval:
                    best_action.append(action)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = float('inf')
            best_action = None
            for action in gameState.getLegalActions(1):
                evaluation, _ = self.minimax(gameState.generateSuccessor(1, action),
                                             depth - 1, alpha, beta, True)
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_action = action
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return min_eval, best_action
