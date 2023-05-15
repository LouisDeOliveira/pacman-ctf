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
USE_WANDB = False

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Generic, Iterable, List, Tuple, TypeVar

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

if USE_WANDB:
    import wandb

    wandb.init(project="pacman")

import game
import util
from baselineTeam import DefensiveReflexAgent, OffensiveReflexAgent
from capture import COLLISION_TOLERANCE, GameState
from captureAgents import CaptureAgent
from distanceCalculator import Distancer, manhattanDistance
from game import Directions

#################
# Team creation #
#################
T = TypeVar("T")
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(
        self, observation_size: int, action_size: int, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.observation_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.action_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        # add average pooling layer
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(84, 5)

    def forward(self, observation: torch.Tensor):
        x = self.relu(self.conv1(observation))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = torch.mean(x, dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@dataclass
class Transition:
    state: np.ndarray
    action: int
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


def timeit(func):
    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(f"{func.__name__} took {end - start} seconds")
        return result

    return timed


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
        """
        Real init, called only once and not on each reset
        """
        super().__init__(*args)
        self.step = 0
        self.game_step = 0
        self.episode_number = 0
        self.wins = 0
        self.buffer = TransitionBuffer()
        self.total_reward = 0
        self.batch_size = 128
        self.training_frequency = 32
        self.plot_frequency = 5
        self.update_target_frequency = 5
        self.save_checkpoint_frequency = 200
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_numbers = {"North": 0, "South": 1, "East": 2, "West": 3, "Stop": 4}
        self.action_names = {v: k for k, v in self.action_numbers.items()}
        self.policy = CNNPolicy()
        self.target = CNNPolicy()
        self.loss = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.0001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.target.to(self.device)
        # self.load_weights("./policy_2000.pt", "./target_2000.pt")
        self.loss_values = []
        self.total_reward_values = []
        self.episode_lenghts_values = []
        self.refoffensiveagent = OffensiveReflexAgent(self.index)
        self.refdefensiveagent = DefensiveReflexAgent(self.index)

    def registerInitialState(self, gameState: GameState):
        """
        Do init stuff in here :)
        """
        # print("I am agent " + str(self.index))
        # print("Total reward from last game: " + str(self.total_reward))
        if self.episode_number > 0:
            self.total_reward_values.append(self.total_reward)
            if USE_WANDB:
                wandb.log({"Total reward": self.total_reward})
        self.total_reward = 0
        if self.episode_number > 0:
            self.episode_lenghts_values.append(self.game_step)
            if USE_WANDB:
                wandb.log({"Episode length": self.game_step})
        if self.episode_number % self.save_checkpoint_frequency == 0:
            print("Saving checkpoint")
            self.save_policy("./policy_" + str(self.episode_number) + ".pt")
            self.save_target("./target_" + str(self.episode_number) + ".pt")
        print("Episode number: " + str(self.episode_number))
        self.game_step = 0
        self.episode_number += 1
        self.map_size = gameState.data.layout.width, gameState.data.layout.height
        self.last_turn_state = gameState
        self.last_turn_observation = self.upscale_matrix(
            self.convert_gamestate(gameState), desired_size=(64, 128)
        )
        self.last_turn_action = 0
        # if self.episode_number % self.plot_frequency == 0:
        #     self.plot_loss()
        if self.episode_number % self.update_target_frequency == 0:
            # print("Updating target network")
            self.target.load_state_dict(self.policy.state_dict())
        # print("wins: " + str(self.wins) + " out of " + str(self.episode_number))
        # print("Episode number: " + str(self.episode_number))
        self.refdefensiveagent.registerInitialState(gameState)
        self.refoffensiveagent.registerInitialState(gameState)
        CaptureAgent.registerInitialState(self, gameState)

    def game_outcome(self, gameState: GameState):
        """
        returns 1 if we won the game, 0 for a draw and -1 if we lost the game
        """
        if gameState.isOver():
            if self.red:
                if gameState.getScore() > 0:
                    return 1
                elif gameState.getScore() == 0:
                    return 0
                else:
                    return -1
            else:
                if gameState.getScore() < 0:
                    return 1
                elif gameState.getScore() == 0:
                    return 0
                else:
                    return -1

    def chooseAction(self, gameState: GameState):
        self.step += 1
        self.game_step += 1
        observation = self.convert_gamestate(gameState)
        upscaled_observation = self.upscale_matrix(observation, desired_size=(64, 128))
        rand = random.random()
        if rand < self.epsilon:
            # we use actions not from the policy in this block
            # if rand < 0.33:
            #     action = self.refdefensiveagent.chooseAction(gameState)
            #     true_action = self.action_numbers[action]
            # uuse offensive reflex agent action
            # elif rand < 0.66:
            action = self.refoffensiveagent.chooseAction(gameState)
            true_action = self.action_numbers[action]

            # else:
            #     action = random.choice(gameState.getLegalActions(self.index))
            #     true_action = self.action_numbers[action]

        else:
            action = self.policy(
                torch.tensor(upscaled_observation)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )  # convert to tensor and add batch dimension
            true_action = self.action_masking(action, gameState)  # action number
        final_action = self.action_names[true_action]  # action name

        reward = (
            self.eat_food_reward(gameState, self.last_turn_state)
            # + self.score_diff_reward(gameState, self.last_turn_state)
            # + self.food_eaten_reward(gameState, self.last_turn_state)
            # + self.has_moved_reward(gameState, self.last_turn_state)
        )

        # make a transition for the buffer
        transition = self.make_transition(
            self.last_turn_action,
            self.last_turn_observation,
            reward,
            upscaled_observation,
            gameState.isOver(),
        )
        self.buffer.append(transition)

        self.total_reward += reward
        self.last_turn_state = gameState
        self.last_turn_observation = upscaled_observation
        self.last_turn_action = true_action
        # print("Current buffer size: " + str(len(self.buffer)))

        # update policy
        if (
            len(self.buffer) >= self.batch_size
            and self.step % self.training_frequency == 0
        ):
            self.learn_step()
        self.epsilon *= self.epsilon_decay
        if USE_WANDB:
            outcome = self.game_outcome(gameState)
            if outcome is not None:
                wandb.log({"Game outcome": outcome})

        return final_action

    def load_weights(self, policy_path, target_path):
        """
        Load weights from a file, to avoid having to train from scratch
        """
        self.policy.load_state_dict(torch.load(policy_path))
        self.target.load_state_dict(torch.load(target_path))

    def learn_step(self):
        """
        Use a batch from the buffer to update the policy
        """
        batch = self.buffer.sample(self.batch_size)
        states = (
            torch.tensor(np.array([x.state for x in batch]))
            .to(self.device)
            .permute(0, 3, 1, 2)
        )
        next_states = (
            torch.tensor(np.array([x.next_state for x in batch]))
            .to(self.device)
            .permute(0, 3, 1, 2)
        )
        actions = torch.tensor(
            np.array([x.action for x in batch]), dtype=torch.int64
        ).to(self.device)
        rewards = torch.tensor(np.array([x.reward for x in batch])).to(self.device)
        dones = torch.tensor(np.array([x.done for x in batch])).to(self.device)

        # Compute the estimated values
        values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_values = torch.zeros(self.batch_size).to(self.device)
        with torch.no_grad():
            # Theses are the values of the next states according to the target network
            next_values[~dones] = self.target(next_states[~dones]).max(1)[0]

        # Compute the target values
        target_values = rewards + self.gamma * next_values

        loss = self.loss(values, target_values)
        self.optimizer.zero_grad()
        loss.backward()

        # prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.optimizer.step()
        self.loss_values.append(loss.item())
        if USE_WANDB:
            wandb.log({"loss": loss.item()})

    def save_policy(self, path: str = "./policy.pt"):
        torch.save(self.policy.state_dict(), path)

    def save_target(self, path: str = "./target.pt"):
        torch.save(self.target.state_dict(), path)

    def plot_loss(self):
        # 2 plots side by side, loss and reward with smoothing over last 50 episodes
        N = 5
        # overlaid on top of the raw data
        smoothed_loss = np.convolve(self.loss_values, np.ones(N) / N, mode="valid")
        smoothed_reward = np.convolve(
            self.total_reward_values, np.ones(N) / N, mode="valid"
        )
        smoothed_episode_length = np.convolve(
            self.episode_lenghts_values, np.ones(N) / N, mode="valid"
        )
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.plot(self.loss_values, label="Raw", color="orange")
        ax1.plot(smoothed_loss, label="Smoothed")
        ax1.set_title(f"Loss, agent {self.index}")
        ax1.set_xlabel("Training steps")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax2.plot(self.total_reward_values, label="Raw", color="orange")
        ax2.plot(smoothed_reward, label="Smoothed")
        ax2.set_title(f"Total reward, agent {self.index}")
        ax2.set_xlabel("Training episodes")
        ax2.set_ylabel("Total reward")
        ax2.legend()
        ax3.plot(self.episode_lenghts_values, label="Raw", color="orange")
        ax3.plot(smoothed_episode_length, label="Smoothed")
        ax3.set_title(f"Episode lenghts, agent {self.index}")
        ax3.set_xlabel("Training episodes")
        ax3.set_ylabel("Episode length")
        ax3.legend()
        plt.show()

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.plot(self.loss_values)
        # ax1.set_title(f"Loss, agent {self.index}")
        # ax1.set_xlabel("Training steps")
        # ax1.set_ylabel("Loss")
        # ax2.plot(self.total_reward_values)
        # ax2.set_title(f"Total reward, agent {self.index}")
        # ax2.set_xlabel("Training episodes")
        # ax2.set_ylabel("Total reward")
        # plt.show()

    def make_transition(
        self,
        action: int,
        observation: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> Transition:
        return Transition(observation, action, next_observation, reward, done)

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

    def get_own_pos(self, gameState: GameState) -> Tuple[int, int]:
        return gameState.getAgentPosition(self.index)

    def make_vision_vector(self, gameState: GameState):
        """
        Generates an observation for a MLP based policy
        """
        obs = []
        own_pos = self.get_own_pos(gameState)
        # distances to 10 closest food
        foods = self.getFood(gameState).asList()
        food_distances = sorted([self.getMazeDistance(own_pos, food) for food in foods])
        obs += food_distances[:10]
        # distance to powerup
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            dist = self.getMazeDistance(own_pos, capsules[0])
            obs.append(dist)
        else:
            obs.append(0)
        # distance to teaammate
        for ally in self.getTeam(gameState):
            if ally != self.index:
                ally_pos = gameState.getAgentPosition(ally)
                if ally_pos is not None:
                    dist = self.getMazeDistance(own_pos, ally_pos)
                    obs.append(dist)
                else:
                    obs.append(0)
        # distance to enemies
        for enemy in self.getOpponents(gameState):
            if enemy != self.index:
                ennemy_pos = gameState.getAgentPosition(enemy)
                if ennemy_pos is not None:
                    dist = self.getMazeDistance(own_pos, ennemy_pos)
                    obs.append(dist)
                else:
                    # if not observable use a noisy distance
                    dist = gameState.getAgentDistances()[enemy]
                    obs.append(dist)

    def make_vision_matrix(self, gameState: GameState):
        """
        Generates an observation for a CNN based policy
        """
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

        return matrix

    # @timeit
    def upscale_matrix(self, matrix: np.ndarray, desired_size: tuple):
        return cv2.resize(matrix, desired_size, interpolation=cv2.INTER_NEAREST)

    def action_masking(self, raw_action_values: torch.Tensor, gameState: GameState):
        legal_actions = gameState.getLegalActions(self.index)
        action_mask = torch.zeros_like(raw_action_values)
        for action in legal_actions:
            action_mask[0, self.action_numbers[action]] = 1

        large = torch.finfo(raw_action_values.dtype).max

        best_legal_action = (
            (raw_action_values - large * (1 - action_mask) - large * (1 - action_mask))
            .argmax()
            .item()
        )
        return best_legal_action

    def convert_gamestate(self, gameState: GameState) -> np.ndarray:
        """
        Converts the gamestate to a numpy array, representing our world model.
        """
        matrix = self.make_vision_matrix(gameState)
        matrix = matrix.astype(np.float32) / 255
        return matrix

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

    def has_moved_reward(self, current_state: GameState, last_state: GameState):
        current_pos = current_state.getAgentPosition(self.index)
        last_pos = last_state.getAgentPosition(self.index)
        if current_pos == last_pos:
            return -1
        return 0

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
            # print(
            #     "sdlfsdfksdjfsjfskldfjsdklfjskdlfjsdkfjskdlfjksdjfksdjfklsdjfjklsdjfklsdjfkjsdfkjsd"
            # )
            pass
        return reward


class NNPlayingAgent(CaptureAgent):
    """
    Agent use for playing with a pretrained neural network
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.device = torch.device("cpu")
        self.action_numbers = {"North": 0, "South": 1, "East": 2, "West": 3, "Stop": 4}
        self.action_names = {v: k for k, v in self.action_numbers.items()}
        self.policy = CNNPolicy().to(self.device)
        self.policy.load_state_dict(
            torch.load("./policy_2000.pt", map_location=self.device)
        )

    def registerInitialState(self, gameState: GameState):
        """
        Do init stuff in here :)
        """
        # print("I am agent " + str(self.index))
        # print("Total reward from last game: " + str(self.total_reward))
        self.map_size = gameState.data.layout.width, gameState.data.layout.height
        self.last_turn_state = gameState
        CaptureAgent.registerInitialState(self, gameState)

    def action_masking(self, raw_action_values: torch.Tensor, gameState: GameState):
        legal_actions = gameState.getLegalActions(self.index)
        action_mask = torch.zeros_like(raw_action_values)
        for action in legal_actions:
            action_mask[0, self.action_numbers[action]] = 1

        large = torch.finfo(raw_action_values.dtype).max

        best_legal_action = (
            (raw_action_values - large * (1 - action_mask) - large * (1 - action_mask))
            .argmax()
            .item()
        )
        return best_legal_action

    def convert_gamestate(self, gameState: GameState) -> np.ndarray:
        """
        Converts the gamestate to a numpy array, representing our world model.
        """
        matrix = self.make_vision_matrix(gameState)
        matrix = matrix.astype(np.float32) / 255
        return matrix

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

        return matrix

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

    def chooseAction(self, gameState: GameState):
        observation = self.convert_gamestate(gameState)
        upscaled_observation = self.upscale_matrix(observation, desired_size=(64, 128))
        # plt.imshow(upscaled_observation)
        # plt.imsave("test.png", upscaled_observation)
        # plt.show()
        action = self.policy(
            torch.tensor(upscaled_observation)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )  # convert to tensor and add batch dimension
        true_action = self.action_masking(action, gameState)  # action number
        final_action = self.action_names[true_action]  # action name
        self.last_turn_state = gameState
        return final_action

    def upscale_matrix(self, matrix: np.ndarray, desired_size: tuple):
        return cv2.resize(matrix, desired_size, interpolation=cv2.INTER_NEAREST)
