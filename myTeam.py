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
from collections import deque
from dataclasses import dataclass
from typing import Generic, Iterable, List, TypeVar

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
T = TypeVar("T")


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float
    done: bool


Episode = list[Transition]


def total_reward(episode: Episode) -> float:
    return sum([t.reward for t in episode])


class Buffer(Generic[T]):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def append(self, obj: T):
        self.memory.append(obj)

    def append_multiple(self, obj_list: list[T]):
        for obj in obj_list:
            self.memory.append(obj)

    def sample(self, batch_size) -> Iterable[T]:
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


class TransitionBuffer(Buffer[Transition]):
    def __init__(self, capacity=100000):
        super().__init__(capacity)

    def append_episode(self, episode: Episode):
        self.append_multiple(episode)

    def get_batch(self, batch_size):
        batch_of_transitions = self.sample(batch_size)
        states = np.array([t.state for t in batch_of_transitions])
        actions = np.array([t.action for t in batch_of_transitions])
        next_states = np.array([t.next_state for t in batch_of_transitions])
        rewards = np.array([t.reward for t in batch_of_transitions])
        dones = np.array([t.done for t in batch_of_transitions])

        return Transition(states, actions, next_states, rewards, dones)


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
    def __init__(self, *args):
        super().__init__(*args)
        self.step = 0
        self.game_step = 0
        self.buffer = TransitionBuffer()

    def registerInitialState(self, gameState: GameState):
        """
        Do init stuff in here :)
        """
        self.game_step = 0
        self.observation_size = 4
        self.action_size = 5
        self.hidden_size = 64
        self.action_numbers = {"North": 0, "South": 1, "East": 2, "West": 3, "Stop": 4}
        self.action_names = {v: k for k, v in self.action_numbers.items()}
        self.model = SimpleModel(
            self.observation_size, self.action_size, self.hidden_size
        )
        self.map_size = gameState.data.layout.width, gameState.data.layout.height
        self.last_turn_state = gameState
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState: GameState):
        self.step += 1
        self.game_step += 1
        print(f"game step: {self.game_step}")
        print(f"step: {self.step}")
        self.make_vision_matrix(gameState)
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
        # print(matrix.shape)
        # print(f"map size: {self.map_size}")

        # update the last known state
        self.last_turn_state = gameState
        return final_action

    def checkEatenFoodAndCapsules(
        self, gameState: GameState, last_turn_state: GameState
    ):
        now_food = self.getFoodYouAreDefending(gameState).asList()
        previous_food = self.getFoodYouAreDefending(last_turn_state).asList()
        previous_capsules = self.getCapsulesYouAreDefending(last_turn_state)
        now_capsules = self.getCapsulesYouAreDefending(gameState)
        eaten_capsules = list(set(previous_capsules) - set(now_capsules))
        eaten_food = list(set(previous_food) - set(now_food))
        return eaten_food + eaten_capsules

    def checkIsScared(self, gameState: GameState, index: int):
        return gameState.getAgentState(index).scaredTimer > 0

    def checkIsPacman(self, gameState: GameState, index: int):
        return gameState.getAgentState(index).isPacman

    def checkWeCanEat(self, gameState: GameState, index: int):
        scared = self.checkIsScared(gameState, index)
        pacman = self.checkIsPacman(gameState, index)

        if scared and not pacman:
            return False
        if not scared and not pacman:
            return True

    def make_vision_matrix(self, gameState: GameState):
        matrix = np.zeros((*self.map_size, 3), dtype=np.uint8)
        owncolor = np.array([0, 0, 200], dtype=np.uint8)
        teammatecolor = np.array([0, 200, 0], dtype=np.uint8)
        offset_teammate_color = np.array([0, 50, 0], dtype=np.uint8)
        offset_own_color = np.array([0, 0, 50], dtype=np.uint8)
        enemycolor = np.array([200, 0, 0], dtype=np.uint8)
        wallcolor = np.array([255, 255, 255], dtype=np.uint8)
        foodcolor = np.array([255, 255, 0], dtype=np.uint8)
        enemyfoodcolor = np.array([255, 128, 0], dtype=np.uint8)
        capsulecolor = np.array([255, 0, 255], dtype=np.uint8)
        enemycapsulecolor = np.array([128, 0, 255], dtype=np.uint8)

        own_pos = gameState.getAgentPosition(self.index)

        matrix[own_pos[0], own_pos[1]] = owncolor
        if self.checkIsPacman(gameState, self.index):
            matrix[own_pos[0], own_pos[1]] = owncolor + offset_own_color
        elif self.checkIsScared(gameState, self.index):
            matrix[own_pos[0], own_pos[1]] = owncolor - offset_own_color

        for ally in self.getTeam(gameState):
            if ally != self.index:
                position = gameState.getAgentPosition(ally)
                if position is not None:
                    matrix[position[0], position[1]] = teammatecolor
                    if self.checkIsPacman(gameState, ally):
                        matrix[position[0], position[1]] = (
                            teammatecolor + offset_teammate_color
                        )
                    elif self.checkIsScared(gameState, ally):
                        matrix[position[0], position[1]] = (
                            teammatecolor - offset_teammate_color
                        )

        for enemy in self.getOpponents(gameState):
            if enemy != self.index:
                position = gameState.getAgentPosition(enemy)
                if position is not None:
                    matrix[position[0], position[1]] = enemycolor

        for wall in gameState.getWalls().asList():
            matrix[wall[0], wall[1]] = wallcolor

        for ownfood in self.getFoodYouAreDefending(gameState).asList():
            matrix[ownfood[0], ownfood[1]] = foodcolor

        for food in self.getFood(gameState).asList():
            matrix[food[0], food[1]] = enemyfoodcolor

        for owncapsule in self.getCapsulesYouAreDefending(gameState):
            matrix[owncapsule[0], owncapsule[1]] = capsulecolor

        for capsule in self.getCapsules(gameState):
            matrix[capsule[0], capsule[1]] = enemycapsulecolor

        for eaten_stuff in self.checkEatenFoodAndCapsules(
            gameState, self.last_turn_state
        ):
            matrix[eaten_stuff[0], eaten_stuff[1]] = enemycolor

        plt.imshow(np.rot90(matrix))
        plt.show()

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
        # print(f"Is pacman: {is_pacman}")
        # print(f"Is red: {is_scared}")
        # plt.imshow(food_matrix)
        # plt.title("food matrix")
        # plt.show()

        # print(clean_distances)
        return torch.randn(1, self.observation_size)

    def cleanup_distances(
        self, gameState: GameState, noisy_distances: List[int], normalize: bool = True
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

        if normalize:
            # normalize distances
            for i in range(len(noisy_distances)):
                noisy_distances[i] = noisy_distances[i] / (
                    self.map_size[0] + self.map_size[1]
                )

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
        positive reward if enemy food is eaten, negative reward if enemy food increases
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
