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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from game import Actions
from game import Grid
from capture import GameState
from DefensiveAgents import DefensiveAgent
from OffensiveAgents import OffensiveAgent

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DefensiveAgent', second = 'OffensiveAgent'):
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

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
 
    '''
    Your initialization code goes here, if you need any.
    '''
    self.currentPath = []
    self.patrol_positions = []
    self.enemy_pacman = None
    self.get_patrol_positions(gameState)
    self.opponent_indexes = self.getOpponents(gameState)
    enemy1, enemy2 = gameState.getAgentPosition(self.opponent_indexes[0]),gameState.getAgentPosition(self.opponent_indexes[1])
    self.agent_pos = gameState.getAgentPosition(self.index)
    self.currentPath =  self.aStarSearch(self.agent_pos, gameState, [self.patrol_positions.pop(0)], avoidPositions=[], returnPosition=False)
    self.grid = self.getFoodYouAreDefending(gameState)
    self.x_N = self.grid.width
    self.y_N = self.grid.height

  def get_food_indexes(self,gameState):
    result = []
    food_positions = self.getFoodYouAreDefending(gameState)
    for i,row in enumerate(food_positions):
      for j,col in enumerate(row):
        if col == True:
          result.append((i,j))
    return sorted(result, key = lambda x: (x[0],x[1]))
  
  def get_patrol_positions(self,gameState):
     food_positions = self.get_food_indexes(gameState)
     power_capsule = self.getCapsulesYouAreDefending(gameState)
     if len(self.patrol_positions) < 1:
       self.patrol_positions.append(food_positions[-1])
       try:
         self.patrol_positions.append(power_capsule[0])
       except:
         pass
       self.patrol_positions.append(food_positions[-5])

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    '''
    You should change this in your own agent.
    '''
    #print(self.getFoodYouAreDefending(gameState))
    # enemy1, enemy2 = gameState.getAgentPosition(self.opponent_indexes[0]),gameState.getAgentPosition(self.opponent_indexes[1])
    action = Directions.STOP
    #self.enemy_pacman = self.getDisappearingFoodPos(gameState)
    #if gameStateself.opponent_indexes:
    # if len(self.enemy_pacman) > 0:
    #    self.currentPath = self.aStarSearch(self.agent_pos, gameState, [self.enemy_pacman.pop(0)], avoidPositions=[], returnPosition=False)
    #    action = self.currentPath.pop(0)
    self.agent_pos = gameState.getAgentPosition(self.index)
    enemy1 = gameState.getAgentPosition(self.opponent_indexes[0])
    enemy2 = gameState.getAgentPosition(self.opponent_indexes[1])
    if enemy1 != None:
      if enemy1[0] > self.x_N:
        safe_place =  self.get_safe_place(gameState)
        self.currentPath = self.aStarSearch(self.agent_pos, gameState, [safe_place], avoidPositions=[], returnPosition=False)
        action = self.currentPath.pop(0)
      else:
        self.currentPath = self.aStarSearch(self.agent_pos, gameState, [enemy1], avoidPositions=[], returnPosition=False)
        action = self.currentPath.pop(0)
    elif enemy2 != None:
      if enemy2[0] > self.x_N:
        safe_place =  self.get_safe_place(gameState)
        self.currentPath = self.aStarSearch(self.agent_pos, gameState, [safe_place], avoidPositions=[], returnPosition=False)
        action = self.currentPath.pop(0)
      else:
        self.currentPath = self.aStarSearch(self.agent_pos, gameState, [enemy2], avoidPositions=[], returnPosition=False)
        action = self.currentPath.pop(0)
    else:
      if len(self.patrol_positions) > 0:
        self.currentPath = self.aStarSearch(self.agent_pos, gameState, [self.patrol_positions.pop(0)], avoidPositions=[], returnPosition=False)
        action = self.currentPath.pop(0)
      else:
        self.get_patrol_positions(gameState)
        self.currentPath = self.aStarSearch(self.agent_pos, gameState, [self.patrol_positions.pop(0)], avoidPositions=[], returnPosition=False)
        action = self.currentPath.pop(0)
  
    return action
  
  def get_safe_place(self, gameState):
    now = self.grid
    x_value = len(now[0])//2 - 1
    for y_value in range(0,len(now)):
       if gameState.hasWall(x_value, y_value) == False:
         return x_value,y_value
       

  def getDisappearingFoodPos(self, gameState):
    foodPos = []
    if len(self.observationHistory) >2:
      prev = self.getPreviousObservation()
      now = self.getCurrentObservation()
      for i in range(len(prev)):
        for j in range(len(prev[0])):
            if prev[i][j] != self.foodGrid[i][j]:
              foodPos.append((i,j))
    return foodPos




  def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[] ,returnPosition=False):
    """
    Finds the distance between the agent with the given index and its nearest goalPosition
    """
    #print("ASTAR STARTED")
    walls = gameState.getWalls()
    width = walls.width
    height = walls.height
    walls = walls.asList()

    actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
    actionVectors = [Actions.directionToVector(action) for action in actions]
    # Change action vectors to integers so they work correctly with indexing
    actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]
    #print(actionVectors)

    # Values are stored a 3-tuples, (Position, Path, TotalCost)

    currentPosition, currentPath, currentTotal = startPosition, [], 0

    #print(currentPosition, "CURRERERERFAE:SDFA")
    # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
    # def priority(entry):
    #    if entry[0] in avoidPositions:
    #       return entry[2] + width * height
    #    elif entry[0] in closePositions:
    #       return 1
    #    else:
    #       min(util.manhattanDistance(entry[0],endPosition) for )
          
    # queue = util.PriorityQueueWithFunction(
    #     lambda entry: entry[2] + width * height if entry[0] in avoidPositions else 0 + min(
    #         util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))
    
    queue = util.PriorityQueueWithFunction(
      lambda entry: 2 if entry[0] in avoidPositions else 0 + min(
          util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))

    #print(queue,'queueueuueueueueueuueue')
    # Keeps track of visited positions
    visited = {currentPosition}
    while currentPosition not in goalPositions:

        possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for
                              vector, action in zip(actionVectors, actions)]
        legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]
        #print(legalPositions, 'LEEGAALLL    POSSSSSS')

        for position, action in legalPositions:
            if position not in visited:
                visited.add(position)
                
                queue.push((position, currentPath + [action], currentTotal + 1))

        # This shouldn't ever happen...But just in case...
        if len(queue.heap) == 0:
            return None
        else:
            currentPosition, currentPath, currentTotal = queue.pop()


    #print(currentPath)
    if returnPosition:
        return currentPath, currentPosition
    else:
        return currentPath

