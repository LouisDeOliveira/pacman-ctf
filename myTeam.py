# myTeam.py
# ---------
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


import random
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import game
import util
from capture import COLLISION_TOLERANCE, GameState
from captureAgents import CaptureAgent
from distanceCalculator import manhattanDistance
from game import Directions
from models import SimpleModel

#################
# Team creation #
#################


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="NNTrainingAgent",
    second="NNTrainingAgent",
    numTraining=0,
):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        """
		Make sure you do not delete the following line. If you would like to
		use Manhattan distances instead of maze distances in order to save
		on initialization time, please take a look at
		CaptureAgent.registerInitialState in captureAgents.py.
		"""
        CaptureAgent.registerInitialState(self, gameState)

        """
		Your initialization code goes here, if you need any.
		"""

    def chooseAction(self, gameState: GameState):
        """
        Picks among actions randomly.
        """
        print("I am agent " + str(self.index))
        print("My current position is " + str(gameState.getAgentPosition(self.index)))
        print("My current score is " + str(gameState.getScore()))
        print("My current state is " + str(gameState.isOver()))
        print(
            "My current legal actions are " + str(gameState.getLegalActions(self.index))
        )
        actions = gameState.getLegalActions(self.index)

        """
		You should change this in your own agent.
		"""

        return random.choice(actions)


class NNTrainingAgent(CaptureAgent):
    def registerInitialState(self, gameState: GameState):
        """
        Do init stuff in here :)
        """
        self.observation_size = 4
        self.action_size = 5
        self.hidden_size = 64
        self.action_numbers = {"North": 0, "South": 1, "East": 2, "West": 3, "Stop": 4}
        self.action_names = {v: k for k, v in self.action_numbers.items()}
        self.model = SimpleModel(
            self.observation_size, self.action_size, self.hidden_size
        )
        self.last_turn_state = gameState
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState: GameState):
        observation = self.convert_gamestate(gameState)
        action = self.model(observation)
        true_action = self.action_masking(action, gameState)  # action number
        final_action = self.action_names[true_action.item()]  # action name

        # print(f"food reward: {self.eat_food_reward(gameState, self.last_turn_state)}")

        # print(f"score diff reward: {self.score_diff_reward(gameState, next_state)}")
        # print(
        #     f"eaten food reward: {self.food_eaten_reward(gameState, self.last_turn_state)}"
        # )
        # print(gameState.getAgentDistances())
        self.checkDeath(gameState, self.index)

        # update the last known state
        self.last_turn_state = gameState
        return final_action

    def make_food_matrix(self, gameState: GameState):
        """
        Converts food data into numpy array.
        """
        food = np.array(self.getFood(gameState).data).astype(np.float32)
        ownfood = np.array(self.getFoodYouAreDefending(gameState).data).astype(
            np.float32
        )
        matrix = food - ownfood
        return matrix

    def action_masking(self, raw_action_values: torch.Tensor, gameState: GameState):
        legal_actions = gameState.getLegalActions(self.index)
        action_mask = torch.zeros_like(raw_action_values)
        for action in legal_actions:
            action_mask[0, self.action_numbers[action]] = 1

        large = torch.finfo(raw_action_values.dtype).max

        best_legal_action = (
            raw_action_values - large * (1 - action_mask) - large * (1 - action_mask)
        ).argmax()
        return best_legal_action

    def convert_gamestate(self, gameState: GameState) -> torch.Tensor:
        # get noisy distances
        noisy_distances = gameState.getAgentDistances()
        clean_distances = self.cleanup_distances(gameState, noisy_distances)

        print(clean_distances)
        return torch.randn(1, self.observation_size)

    def cleanup_distances(
        self, gameState: GameState, noisy_distances: List[int]
    ) -> List[int]:
        """
        Cleans up the noisy distances for agents that are in range.
        """
        own_pos = gameState.getAgentPosition(self.index)
        # replace noisy distances with their true values if we can see the agent
        for opponent_idx in self.getOpponents(gameState):
            opponent_pos = gameState.getAgentPosition(opponent_idx)
            if opponent_pos is not None:
                distance = self.getMazeDistance(own_pos, opponent_pos)
                noisy_distances[opponent_idx] = distance

        for teammate_idx in self.getTeam(gameState):
            teammate_pos = gameState.getAgentPosition(teammate_idx)
            if teammate_pos is not None:
                distance = self.getMazeDistance(own_pos, teammate_pos)
                noisy_distances[teammate_idx] = distance

        return noisy_distances

    def get_next_state(self, gameState: GameState, action: str) -> GameState:
        """
        Returns the next state of the game given an action
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def score_diff_reward(
        self, current_state: GameState, successor: GameState
    ) -> float:
        return successor.getScore() - current_state.getScore()

    def eat_food_reward(
        self, current_state: GameState, previous_state: GameState
    ) -> float:
        """
        positive reward if ennemy food is eaten, negative reward if ennemy food increases
        """
        current_food = self.getFood(current_state).data
        current_food = np.sum(np.array(current_food).astype(int))

        prev_food = self.getFood(previous_state).data
        prev_food = np.sum(np.array(prev_food).astype(int))
        return prev_food - current_food

    def food_eaten_reward(
        self, current_state: GameState, previous_state: GameState
    ) -> float:
        """
        negative reward if our food is eaten, positive reward if our food increases
        """
        current_food = self.getFoodYouAreDefending(current_state).data
        current_food = np.sum(np.array(current_food).astype(int))

        prev_food = self.getFoodYouAreDefending(previous_state).data
        prev_food = np.sum(np.array(prev_food).astype(int))
        return current_food - prev_food

    def checkDeath(self, state: GameState, agentIndex: int):
        reward = 0
        thisAgentState = state.data.agentStates[agentIndex]
        if state.isOnRedTeam(agentIndex):
            otherTeam = state.getBlueTeamIndices()
        else:
            otherTeam = state.getRedTeamIndices()

        if thisAgentState.isPacman:
            for index in otherTeam:
                otherAgentState = state.data.agentStates[index]
                if otherAgentState.isPacman:
                    continue
                ghostPosition = otherAgentState.getPosition()
                if ghostPosition == None:
                    continue
                if (
                    manhattanDistance(ghostPosition, thisAgentState.getPosition())
                    <= COLLISION_TOLERANCE
                ):
                    # award points to the other team for killing Pacmen
                    if otherAgentState.scaredTimer <= 0:
                        # ghost killed this Pac-man!
                        reward = -1
                    else:
                        # we killed a ghost! Yay! Because this Pac-Man got power powerup
                        reward = 1
        else:  # Agent is a ghost
            for index in otherTeam:
                otherAgentState = state.data.agentStates[index]
                if not otherAgentState.isPacman:
                    continue
                pacPos = otherAgentState.getPosition()
                if pacPos == None:
                    continue
                if (
                    manhattanDistance(pacPos, thisAgentState.getPosition())
                    <= COLLISION_TOLERANCE
                ):
                    # award points to the other team for killing Pacmen
                    if thisAgentState.scaredTimer <= 0:
                        # we killed a Pac-Man!
                        reward = 1
                    else:
                        # The powered up enemy pacman killed us!
                        reward = -1
        if reward != 0:
            print(
                "sdlfsdfksdjfsjfskldfjsdklfjskdlfjsdkfjskdlfjksdjfksdjfklsdjfjklsdjfklsdjfkjsdfkjsd"
            )
        return reward
