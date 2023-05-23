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


import heapq
import random
import sys
import time
from typing import Tuple

import numpy as np

import game
import util
from capture import GameState
from captureAgents import CaptureAgent
from game import Actions, Directions
from MCTS.AI import MCTSfindMove, removeStop
from MCTS.MCTSData import MCTSData
from MCTS.Node import Node

distributions = []
distances = []

np.set_printoptions(threshold=sys.maxsize)

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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




  # ESTIMATING
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

  def registerInitialState(self, gameState:GameState):
    global distributions, distances
    
    """
    This method handles the initial setup of the gameState.getLegalActions(self.data.player)
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    
    self.WALLS = gameState.data.layout.walls
    self.DIR_STR2VEC = {'North':(0,1), 'South':(0,-1), 'East':(1,0), 'West':(-1,0)}
    self.DIR_VEC2STR = {(0,1):'North', (0,-1):'South', (1,0):'East', (-1,0):'West', (0,0):'Stop'}
    
    self.data = MCTSData(gameState, self.index, UCB1=1, sim_time=0.95)
    # self.data.find_choke_points()
    self.data.middle = (self.WALLS.width-1)/2
    # self.move_from_MCTS = False
    # self.start_pos = gameState.getAgentPosition(self.index)

    for index in self.getTeam(gameState):
       if index != self.index: self.friend_index = index

    if self.index in [0, 1]: ### ONLY THE FIRST AGENT SHOULD INITIALIZE
        distances = self.calculate_distances()
        # np.save("distances.npy", distances)
        for opponent in self.getOpponents(gameState):
            position = gameState.getInitialAgentPosition(opponent)
            distributions.append(dict())
            distributions[-1][position] = 1
            
    self.data.distances = distances
    # self.data.distances = np.load("distances.npy")
    self.data.max_distance = np.amax(self.data.distances)
        
    self.enemy_positions = [[None, None],[None,None]] # [enemy][0 = current, 1 = past]
    self.data.distributions = distributions
    

  def chooseAction(self, gameState:GameState) -> str:
    self.my_pos = gameState.getAgentPosition(self.data.player)

    self.update_distributions(gameState)
    
    players = []
    for i in range(4):
        pos = gameState.getAgentPosition(i)
        if not pos: continue
        threshold = -1 if self.data.state.isOnRedTeam(self.index) == self.data.state.isOnRedTeam(i) else 100
        if self.data.player == i or self.data.distances[pos[0]][pos[1]][self.my_pos[0]][self.my_pos[1]] <= threshold: players.append(i)
    players = np.array(players)
    self.data.players = players
    
    # deciding whether to keep tree, unclear if tree should always be discarded or not
    self.discard_tree(gameState)
    # self.handle_tree_keeping(gameState, self.move_from_MCTS)
    
    mcts_move = MCTSfindMove(self.data)
        
    # if MCTS is uncertain of what to do
    # child_visits = [child.visits for child in self.data.root.children]
    # if len(child_visits) > 1 and (max(child_visits) - min(child_visits) < 0.05*self.data.root.visits): 
        # self.moves = []
    #     # if self.index <= 1: self.go_to_nearest_enemy()
    #     if self.moves:
    #         self.move_from_MCTS = False
    #         return self.moves[0]
        
    #     self.data.get_food_locations()
    #     if gameState.getAgentState(self.index).numCarrying >= 10 or \
    #       (gameState.getAgentState(self.index).numCarrying >  0  and len(self.data.food) <= 2): self.go_to_deposit()
    #     else: self.go_to_nearest_food()
    #     self.move_from_MCTS = False
    #     if self.moves: return self.moves[0]
        
    # print("MCTS")
    # self.move_from_MCTS = True
    return mcts_move


  def go_to_nearest_food(self) -> None:
    print("FOOD")
    if self.index <= 1: food = self.data.food[:len(self.data.food)//2]
    else: food = self.data.food[len(self.data.food)//2:]
    
    closest_food_dist, closest_food = 100, ()
    for food_location in food:
        if self.data.distances[self.my_pos[0]][self.my_pos[1]][food_location[0]][food_location[1]] < closest_food_dist and food_location != self.my_pos: 
            closest_food_dist = self.data.distances[self.my_pos[0]][self.my_pos[1]][food_location[0]][food_location[1]]
            closest_food = food_location
    self.moves = self.movesToPoint(self.my_pos, closest_food)


  def go_to_nearest_enemy(self) -> None:
    print("KILL", end=" : ")
    middle = (self.WALLS.width-1)/2
    enemy_dist, enemy_location = 100, ()
    for dist in distributions:
        for pos in dist:
            if (pos[0] > middle and self.data.state.isOnRedTeam(self.index)) or (pos[0] < middle and not self.data.state.isOnRedTeam(self.index)): continue
            if self.data.distances[self.my_pos[0]][self.my_pos[1]][pos[0]][pos[1]] < enemy_dist: 
                enemy_dist = self.data.distances[self.my_pos[0]][self.my_pos[1]][pos[0]][pos[1]]
                enemy_location = pos
    self.moves = self.movesToPoint(self.my_pos, enemy_location)


  def go_to_deposit(self) -> None:
    print("HOME")
    if self.data.state.isOnRedTeam(self.index): home = self.WALLS.width//2 - 1
    else: home = self.WALLS.width//2
    
    closest_home_dist, closest_home = 100, ()
    for y in range(self.WALLS.height):
        if self.WALLS[home][y]: continue
        if self.data.distances[self.my_pos[0]][self.my_pos[1]][home][y] < closest_home_dist: 
            closest_home_dist = self.data.distances[self.my_pos[0]][self.my_pos[1]][home][y]
            closest_home = (home, y)
    self.moves = self.movesToPoint(self.my_pos, closest_home)


  def movesToPoint(self, init_pos:Tuple[int], point:Tuple[int]) -> list[str]:
    checked = set()
    queue = []
    
    heapq.heappush(queue, (0, [init_pos]))
    checked.add(init_pos)

    while queue:
        current_cost, current_path = heapq.heappop(queue)
        current_pos = current_path[-1]
        
        if current_pos == point:
            path = np.array(current_path)
            (path[1:] - path[:-1]).tolist()
            
            return [self.DIR_VEC2STR[(current_path[i+1][0] - current_path[i][0], current_path[i+1][1] - current_path[i][1])] for i in range(len(current_path)-1)]

        for dir in self.getPossibleDirections(current_pos):
            dx, dy = self.DIR_STR2VEC[dir]
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if new_pos not in checked:
                new_path = current_path.copy()
                new_path.append(new_pos)
                heapq.heappush(queue, (current_cost+1, new_path))
                checked.add(new_pos)
        
        
  def getPossibleDirections(self, pos:Tuple[int]) -> list[str]:
    possible = []
    x, y = pos

    for dir, vec in self.DIR_STR2VEC.items():
        if dir == 'Stop': continue
        dx, dy = vec
        next_y = y + dy
        next_x = x + dx
        if not self.WALLS[next_x][next_y]: possible.append(dir)
    return possible


  def handle_tree_keeping(self, gameState:GameState, from_MCTS:bool):
    if self.keeping_tree(): self.keep_tree(gameState, from_MCTS)
    else: self.discard_tree(gameState)
    self.data.last_people_in_tree = self.data.players
    self.data.player_pos = np.array([gameState.getAgentPosition(i) for i in self.data.players])


  def keep_tree(self, gameState:GameState, from_MCTS:bool):
    new_pos = np.array([gameState.getAgentPosition(i) for i in self.data.players])
    moves = new_pos - self.data.player_pos
    index = np.where(self.data.players == self.index)[0][0]
    # print(f"I am player {self.index} and I think players {self.data.players} did the moves {moves}")
    for i, player in enumerate(self.data.players):
        move = moves[index]
        # print(pos_diff)
        # print(f"players: {self.data.players}, this_player: {self.index} index: {index}, move: {move}")
        # print([x.move for x in self.data.root.children])
        self.data.root = next((x for x in self.data.root.children if x.move == self.DIR_VEC2STR[tuple(move)]))
        index = (index + 1) % len(self.data.players)
        
        
        # if from_MCTS: self.data.root = self.data.root.chooseBestChild()
        # else: self.data.root = next((x for x in self.data.root.children if x.move == self.moves[0]))
    self.data.root.player = self.index
    self.data.root.parent = None
    self.data.state = gameState
    self.data.keeps += 1
    # self.data.state = self.data.state.generateSuccessor(self.data.player, self.data.root.move)


  def discard_tree(self, gameState:GameState):
    self.data.state = gameState
    self.data.calculate_threshold()
    self.data.root = Node(self.data.player)
    moves = gameState.getLegalActions(self.data.player)
    if self.index > self.data.defender_threshold: removeStop(moves)
    self.data.root.makeChildren(self.data.player, moves)
    
    
  def keeping_tree(self) -> bool:
    return np.array_equal(self.data.last_people_in_tree, self.data.players)


  def calculate_distances(self) -> np.ndarray:
    distance_matrix = np.zeros((self.WALLS.width, self.WALLS.height, self.WALLS.width, self.WALLS.height), np.int32)

    for y_start in range(self.WALLS.height):
        for x_start in range(self.WALLS.width):
            if self.WALLS[x_start][y_start]: continue
            for y in range(self.WALLS.height):
                for x in range(self.WALLS.width):
                    if (x_start, y_start) == (x, y): continue
                    if self.WALLS[x][y] or distance_matrix[x_start][y_start][x][y]: continue

                    path_length = len(self.movesToPoint((x_start, y_start), (x, y)))
                    distance_matrix[x_start][y_start][x][y] = path_length
                    distance_matrix[x][y][x_start][y_start] = path_length
    
    return distance_matrix
  
  
  def update_distributions(self,gameState:GameState): ### ADD FRIEND
    global distributions
    friend_pos = gameState.getAgentPosition(self.friend_index)
    enemies = self.getOpponents(gameState)

    distances = gameState.getAgentDistances()
    
    ### CONVERTS THEIR 0-3 INDEX TO 0-1
    last_enemy = (self.index-1) % 4
    if last_enemy < 2: enemy_index = 0
    else: enemy_index = 1

    if gameState.getAgentPosition(enemies[enemy_index]): ### THEN WE CAN SEE THEM AND KNOW THEIR EXACT POSITION
        distributions[enemy_index] = dict()
        distributions[enemy_index][gameState.getAgentPosition(enemies[enemy_index])] = 1
    else:  
       self.spread_distribution_for_enemy(gameState, enemy_index, self.my_pos)

    for i_0_1, i in enumerate(self.getOpponents(gameState)):
       self.enemy_positions[i_0_1][1] = self.enemy_positions[i_0_1][0]
       self.enemy_positions[i_0_1][0] = gameState.getAgentPosition(i)
       if self.enemy_positions[i_0_1][1] == None:
          continue
       
       if not self.enemy_positions[i_0_1][0] and util.manhattanDistance(self.enemy_positions[i_0_1][1], self.my_pos) <= 2 and self.my_pos != gameState.getInitialAgentPosition(self.index):
          position = gameState.getInitialAgentPosition(i)
          distributions[i_0_1] =  dict()
          distributions[i_0_1][position] = 1
          self.spread_distribution_for_enemy(gameState, i_0_1, self.my_pos)

    ### NOW GOES THROUGH BOTH ENEMY AGENTS AND PRUNES THEIR POSITIONS BASED ON THE MEASUREMENT
    for i, distribution in enumerate(distributions):
        measured_distance = distances[enemies[i]]
        to_remove = []
        for position in distribution.keys():
            true_distance = int(util.manhattanDistance(self.my_pos, position))
            friend_distance = int(util.manhattanDistance(friend_pos, position))
            if ((true_distance <= 5 or friend_distance <= 5) and gameState.getAgentPosition(enemies[i]) != position) or \
                gameState.getDistanceProb(measured_distance,true_distance) == 0: ### If we can see the square or if the probability of the measurement is zero
                to_remove.append(position)
        for position in to_remove:
            del distributions[i][position]
        
    # drawing = [util.Counter() for _ in range(2)]
    # for i, distribution in enumerate(distributions):
    #     for pos in distribution:
    #         drawing[i][pos] = 1
    
    # self.displayDistributionsOverPositions(drawing)


  def spread_distribution_for_enemy(self,gameState, enemy_0_1,my_pos):
    global distributions
    distances = gameState.getAgentDistances()
    distribution = distributions[enemy_0_1]
    enemies = self.getOpponents(gameState)
    measured_distance = distances[enemies[enemy_0_1]]
    positions_to_add = []
    for position in distribution.keys(): ### IF position IS A POSSIBLE LOCATION FOR THE AGENT
        for direction in self.DIR_VEC2STR:
            new_pos = (position[0]+direction[0], position[1]+direction[1])
            if (new_pos[0]>=0 and new_pos[0] < gameState.data.layout.width) and\
              (new_pos[1]>=0 and new_pos[1] < gameState.data.layout.height) and not gameState.hasWall(new_pos[0], new_pos[1]):  ### NOT OUT OF BOUNDS AND NOT A WALL
                true_distance = int(util.manhattanDistance(my_pos, new_pos))
                if gameState.getDistanceProb(true_distance, measured_distance) > 0:  ### IF THE MEASUREMENT DID NOT MAKE THAT POSITION IMPOSSIBLE     
                    positions_to_add.append(new_pos)
    for newPosition in positions_to_add:
      distributions[enemy_0_1][newPosition] = 1
