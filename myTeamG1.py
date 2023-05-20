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
from typing import Generic, Iterable, List, Tuple, TypeVar

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import game
import util
from baselineTeam import DefensiveReflexAgent, OffensiveReflexAgent
from capture import COLLISION_TOLERANCE, GameState
from captureAgents import CaptureAgent
from distanceCalculator import manhattanDistance
from game import Directions

USE_WANDB = True

if USE_WANDB:
    import wandb

    wandb.init(project="pacman")


#################
# Team creation #
#################
T = TypeVar("T")


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

    def sample_weighted(self, batch_size):
        """
        Sample experiences weighted by reward
        """
        rewards = [t.reward for t in self.memory]
        weights = [t.reward + abs(min(rewards)) for t in self.memory]
        return random.choices(self.memory, weights=weights, k=batch_size)


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
    second="DefensiveReflexAgent",
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


USE_IMAGE = False


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
        self.buffer = TransitionBuffer(capacity=10000)
        self.total_reward = 0
        self.batch_size = 128
        self.training_frequency = 32
        self.plot_frequency = 5
        self.update_target_frequency = 5
        self.save_checkpoint_frequency = 100
        self.gamma = 0.99
        self.epsilon = 0.1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_numbers = {"North": 0, "South": 1, "East": 2, "West": 3, "Stop": 4}
        self.action_names = {v: k for k, v in self.action_numbers.items()}

        if USE_IMAGE:
            self.policy = CNNPolicy()
            self.target = CNNPolicy()
        else:
            self.policy = SimpleModel(observation_size=75, action_size=5)
            self.target = SimpleModel(observation_size=75, action_size=5)
        self.loss = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.0001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.target.to(self.device)
        # self.load_weights("./policy_450.pt", "./target_450.pt")
        self.refoffensiveagent = OffensiveReflexAgent(self.index)
        self.refdefensiveagent = DefensiveReflexAgent(self.index)

    def registerInitialState(self, gameState: GameState):
        """
        Do init stuff in here :)
        """
        # print("I am agent " + str(self.index))
        # print("Total reward from last game: " + str(self.total_reward))

        # Empty gpu cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Log to wandb
        if USE_WANDB and self.episode_number > 0:
            wandb.log({"Total reward": self.total_reward})
            wandb.log({"Episode length": self.game_step})
        self.total_reward = 0
        self.game_step = 0

        # Save models
        if self.episode_number % self.save_checkpoint_frequency == 0:
            print("Saving checkpoint")
            if USE_IMAGE:
                self.save_policy(
                    "./policy_" + str(self.episode_number) + "_CNN" + ".pt"
                )
                self.save_target(
                    "./target_" + str(self.episode_number) + "_CNN" + ".pt"
                )
            else:
                self.save_policy(
                    "./policy_" + str(self.episode_number) + "_MLP" + ".pt"
                )
                self.save_target(
                    "./target_" + str(self.episode_number) + "_MLP" + ".pt"
                )

        print("Episode number: " + str(self.episode_number))
        self.episode_number += 1

        # Update target network
        if self.episode_number % self.update_target_frequency == 0:
            self.target.load_state_dict(self.policy.state_dict())

        # Reset agents
        self.refdefensiveagent.registerInitialState(gameState)
        self.refoffensiveagent.registerInitialState(gameState)
        CaptureAgent.registerInitialState(self, gameState)
        self.map_size = gameState.data.layout.width, gameState.data.layout.height
        self.MAX_DIST = max(self.distancer._distances.values())
        # initialize the last turn state
        self.last_turn_state = gameState
        self.last_turn_action = 0
        if USE_IMAGE:
            self.last_turn_observation = self.make_CNN_input(gameState)
        else:
            self.last_turn_observation = self.make_vision_vector(gameState)
        self.start_pos = gameState.getAgentPosition(self.index)

    def count_food(self, gameState: GameState):
        return len(self.getFood(gameState).asList())

    def distance_to_start_reward(self, gameState: GameState):
        distance = self.getMazeDistance(
            gameState.getAgentPosition(self.index), self.start_pos
        )
        score = distance / self.MAX_DIST
        return -1 / (score + 1)

    def make_CNN_input(self, gameState: GameState, desired_size=(64, 128)):
        observation = self.convert_gamestate(gameState)
        observation = self.upscale_matrix(observation, desired_size=desired_size)
        return observation

    def chooseAction(self, gameState: GameState):
        self.step += 1
        self.game_step += 1

        if USE_IMAGE:
            observation = self.make_CNN_input(gameState)
        else:
            observation = self.make_vision_vector(gameState)
        # print(observation.shape)
        rand = random.random()
        if rand < self.epsilon:
            # we use actions not from the policy in this block
            final_action = self.refoffensiveagent.chooseAction(gameState)
            true_action = self.action_numbers[final_action]
        else:
            if USE_IMAGE:
                action = self.policy(
                    torch.tensor(observation)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(self.device)
                )  # convert to tensor and add batch dimension
            else:
                action = self.policy(
                    torch.tensor(observation, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )  # convert to tensor and add batch dimension
            true_action = self.action_masking(action, gameState)  # action number
        final_action = self.action_names[true_action]  # action name

        eatFoodRwdd = self.eat_food_reward(gameState, self.last_turn_state)
        scoreDiffRwd = self.score_diff_reward(gameState, self.last_turn_state)
        hasMovedRwd = -1 if final_action == "Stop" else 0
        killedEnemyRwd = self.checkKill(gameState, self.last_turn_state, self.index)
        distanceGoalRwd = self.distance_to_start_reward(gameState)

        reward = (
            eatFoodRwdd + scoreDiffRwd + hasMovedRwd + killedEnemyRwd + distanceGoalRwd
        )

        # make a transition for the buffer
        transition = self.make_transition(
            self.last_turn_action,
            self.last_turn_observation,
            reward,
            observation,
            gameState.isOver(),
        )
        if gameState.isOver():
            print("Game over")
        self.buffer.append(transition)

        self.total_reward += reward
        self.last_turn_state = gameState
        self.last_turn_observation = observation
        self.last_turn_action = true_action

        # update policy
        if (
            len(self.buffer) >= self.batch_size
            and self.step % self.training_frequency == 0
        ):
            self.learn_step()
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

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
        if USE_IMAGE:
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
        else:
            states = torch.tensor(np.array([x.state for x in batch])).to(self.device)
            next_states = torch.tensor(np.array([x.next_state for x in batch])).to(
                self.device
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
        if USE_WANDB:
            wandb.log({"loss": loss.item()})

    def save_policy(self, path: str = "./policy.pt"):
        torch.save(self.policy.state_dict(), path)

    def save_target(self, path: str = "./target.pt"):
        torch.save(self.target.state_dict(), path)

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

    def isVulnerable(self, gameState: GameState, index: int):
        enemy_index = self.getOpponents(gameState)[0]
        if self.checkIsScared(gameState, index):
            return True
        elif self.checkIsPacman(gameState, index) and not self.checkIsScared(
            gameState, enemy_index
        ):
            return True
        else:
            return False

    def checkCanGoUp(self, gameState: GameState, index: int):
        return not gameState.hasWall(
            self.get_own_pos(gameState)[0], self.get_own_pos(gameState)[1] + 1
        )

    def checkCanGoDown(self, gameState: GameState, index: int):
        return not gameState.hasWall(
            self.get_own_pos(gameState)[0], self.get_own_pos(gameState)[1] - 1
        )

    def checkCanGoLeft(self, gameState: GameState, index: int):
        return not gameState.hasWall(
            self.get_own_pos(gameState)[0] - 1, self.get_own_pos(gameState)[1]
        )

    def checkCanGoRight(self, gameState: GameState, index: int):
        return not gameState.hasWall(
            self.get_own_pos(gameState)[0] + 1, self.get_own_pos(gameState)[1]
        )

    def get_own_pos(self, gameState: GameState) -> Tuple[int, int]:
        return gameState.getAgentPosition(self.index)

    def make_vision_vector(self, gameState: GameState):
        """
        Generates an observation for a MLP based policy
        """
        obs = []
        own_pos = self.get_own_pos(gameState)
        # distances to 65 closest food
        foods = self.getFood(gameState).asList()
        food_distances = sorted(
            [self.getMazeDistance(own_pos, food) / self.MAX_DIST for food in foods]
        )
        if len(food_distances) < 65:
            obs += food_distances
            obs += [1] * (10 - len(food_distances))
        else:
            obs += food_distances[:65]
        print(len(obs))
        # distance to powerup
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            dist = self.getMazeDistance(own_pos, capsules[0])
            obs.append(dist / self.MAX_DIST)
        else:
            obs.append(1)
        print(len(obs))
        # distance to teaammate
        for ally in self.getTeam(gameState):
            if ally != self.index:
                ally_pos = gameState.getAgentPosition(ally)
                if ally_pos is not None:
                    dist = self.getMazeDistance(own_pos, ally_pos)
                    obs.append(dist / self.MAX_DIST)
                else:
                    obs.append(1)
        print(len(obs))
        # distance to enemies
        for enemy in self.getOpponents(gameState):
            if enemy != self.index:
                ennemy_pos = gameState.getAgentPosition(enemy)
                if ennemy_pos is not None:
                    dist = self.getMazeDistance(own_pos, ennemy_pos)
                    obs.append(dist / self.MAX_DIST)
                else:
                    # if not observable use a noisy distance
                    dist = gameState.getAgentDistances()[enemy]
                    dist = max(0, dist)
                    obs.append(dist / self.MAX_DIST)
        print(len(obs))
        # is scared
        obs.append(int(self.checkIsScared(gameState, self.index)))
        # is pacman
        obs.append(int(self.checkIsPacman(gameState, self.index)))
        print(len(obs))
        # check if we can go up, down, left, right
        obs.append(int(self.checkCanGoUp(gameState, self.index)))
        obs.append(int(self.checkCanGoDown(gameState, self.index)))
        obs.append(int(self.checkCanGoLeft(gameState, self.index)))
        obs.append(int(self.checkCanGoRight(gameState, self.index)))
        print(len(obs))
        obs_array = np.array(obs, dtype=np.float32)
        return obs_array

    def make_vision_matrix(self, gameState: GameState):
        """
        Generates an observation for a CNN based policy
        """
        matrix = np.zeros((*self.map_size, 3), dtype=np.uint8)
        owncolor = np.array([0, 0, 200], dtype=np.uint8)
        teammatecolor = np.array([0, 200, 0], dtype=np.uint8)
        offset_color = np.array([0, 50, 0], dtype=np.uint8)
        enemycolor = np.array([200, 0, 0], dtype=np.uint8)
        wallcolor = np.array([255, 255, 255], dtype=np.uint8)
        foodcolor = np.array([255, 255, 0], dtype=np.uint8)
        enemyfoodcolor = np.array([255, 128, 0], dtype=np.uint8)
        capsulecolor = np.array([255, 0, 255], dtype=np.uint8)
        enemycapsulecolor = np.array([128, 0, 255], dtype=np.uint8)

        own_pos = gameState.getAgentPosition(self.index)

        matrix[own_pos[0], own_pos[1]] = owncolor
        if self.isVulnerable(gameState, self.index):
            matrix[own_pos[0], own_pos[1]] = owncolor + offset_color

        for ally in self.getTeam(gameState):
            if ally != self.index:
                position = gameState.getAgentPosition(ally)
                if position is not None:
                    matrix[position[0], position[1]] = teammatecolor
                    if self.isVulnerable(gameState, self.index):
                        matrix[position[0], position[1]] = teammatecolor + offset_color

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
        change = successor.getScore() - current_state.getScore()
        if self.red:
            return change
        else:
            return -change

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
        current_carried_food = current_state.getAgentState(self.index).numCarrying
        prev_carried_food = previous_state.getAgentState(self.index).numCarrying

        current_num_returned = current_state.getAgentState(self.index).numReturned
        prev_num_returned = previous_state.getAgentState(self.index).numReturned

        diff_returned = current_num_returned - prev_num_returned
        diff_carried = current_carried_food - prev_carried_food

        if diff_returned > 0:
            return diff_returned
        else:
            return diff_carried

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

    def checkKill(self, state: GameState, prevState: GameState, agentIndex: int):
        reward = 0.0
        ownFoodAppeared = self.food_eaten_reward(state, prevState)
        thisAgentPrevState = prevState.data.agentStates[agentIndex]
        if state.isOnRedTeam(agentIndex):
            otherTeam = state.getBlueTeamIndices()
        else:
            otherTeam = state.getRedTeamIndices()

        if ownFoodAppeared > 0:  # if food has appeared in our field
            if (
                not thisAgentPrevState.isPacman and thisAgentPrevState.scaredTimer <= 0
            ):  # should we check this for the previous state?
                for index in otherTeam:
                    otherAgentPrevState = prevState.data.agentStates[index]
                    if not otherAgentPrevState.isPacman:  # if the enemy is a Ghost,
                        continue
                    enemyPackmanPosition = otherAgentPrevState.getPosition()
                    if enemyPackmanPosition is None:
                        continue
                    if (
                        manhattanDistance(
                            enemyPackmanPosition, thisAgentPrevState.getPosition()
                        )
                        <= COLLISION_TOLERANCE
                    ):
                        reward = ownFoodAppeared
        return reward

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
                if ghostPosition is None:
                    continue
                if (
                    manhattanDistance(ghostPosition, thisAgentState.getPosition())
                    <= COLLISION_TOLERANCE
                ):
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
                if pacPos is None:
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
            torch.load("./policy_100.pt", map_location=self.device)
        )

    def registerInitialState(self, gameState: GameState):
        """
        Do init stuff in here :)
        """
        # print("I am agent " + str(self.index))
        # print("Total reward from last game: " + str(self.total_reward))
        self.map_size = gameState.data.layout.width, gameState.data.layout.height
        self.last_turn_state = gameState
        self.refoffensiveagent = OffensiveReflexAgent(self.index)
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
        """
        Generates an observation for a CNN based policy
        """
        matrix = np.zeros((*self.map_size, 3), dtype=np.uint8)
        owncolor = np.array([0, 0, 200], dtype=np.uint8)
        teammatecolor = np.array([0, 200, 0], dtype=np.uint8)
        offset_color = np.array([0, 50, 0], dtype=np.uint8)
        enemycolor = np.array([200, 0, 0], dtype=np.uint8)
        wallcolor = np.array([255, 255, 255], dtype=np.uint8)
        foodcolor = np.array([255, 255, 0], dtype=np.uint8)
        enemyfoodcolor = np.array([255, 128, 0], dtype=np.uint8)
        capsulecolor = np.array([255, 0, 255], dtype=np.uint8)
        enemycapsulecolor = np.array([128, 0, 255], dtype=np.uint8)

        own_pos = gameState.getAgentPosition(self.index)

        matrix[own_pos[0], own_pos[1]] = owncolor
        if self.isVulnerable(gameState, self.index):
            matrix[own_pos[0], own_pos[1]] = owncolor + offset_color

        for ally in self.getTeam(gameState):
            if ally != self.index:
                position = gameState.getAgentPosition(ally)
                if position is not None:
                    matrix[position[0], position[1]] = teammatecolor
                    if self.isVulnerable(gameState, self.index):
                        matrix[position[0], position[1]] = teammatecolor + offset_color

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

    def isVulnerable(self, gameState: GameState, index: int):
        enemy_index = self.getOpponents(gameState)[0]
        if self.checkIsScared(gameState, index):
            return True
        elif self.checkIsPacman(gameState, index) and not self.checkIsScared(
            gameState, enemy_index
        ):
            return True
        else:
            return False
