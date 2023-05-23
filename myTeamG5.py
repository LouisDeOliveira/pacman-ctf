from captureAgents import CaptureAgent
import random, time, util, math, sys
from game import Directions, Actions
import game, os, json, copy
from util import nearestPoint, PriorityQueue
import distanceCalculator

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ApproximateQLearningAgent', second = 'DefensiveAgent'):
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

######################
#     Our Agents     #
# written by Group 5 #
######################

class ApproximateQLearningAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    self.epsilon = 0.1 # exploration rate
    self.alpha = 0.2 # learning rate
    self.gamma = 0.9 # discount factor
    # self.epsilon = 0.05
    # self.alpha = 0.2
    # self.gamma = 0.8
    self.trainingMode = False
    self.oldGameState = gameState
    self.numTraining = 1000
    self.episodesSoFar = 1
    self.accumTrainingRewards = 0.0
    self.accumExploitRewards = 0.0
    self.dict = util.Counter()

    self.red_ind = gameState.getRedTeamIndices()
    self.blue_ind = gameState.getBlueTeamIndices()
    self.my_team = 'blue' if self.index in self.blue_ind else 'red'

    if self.my_team == 'blue':
      self.team = self.blue_ind
      self.opponent = self.red_ind
    else:
      self.team = self.red_ind
      self.opponent = self.blue_ind
      

    # self.weights = util.Counter()
    # self.getWeights("OptimalWeights.txt")
    
    self.weights = {"closestCapsuleDistance": -7.0443804635095e+205, 
                    "closestFoodDistance": -6.397772875551455e+206, 
                    "bias": 1, 
                    "numGhostsOneStepAway": 7.790394065146265e+207, 
                    "eatsFoodNext": 9.413582780690914e+207, 
                    "numInvaders": 8.1962365449863e+204, 
                    "closestInvaderDistance": 2.844388654939921e+206, 
                    "goHome": -1.95158804160861e+204, 
                    "numCapsules": 8.350328555690777e+206, 
                    "areTrapped": 2.538255086782103e+204}
    
    # self.weights = {"closestInvaderDistance": 22.844388654939921,
    #                 'closestCapsuleDistance': 54.95292664106621, 
    #                 'closestFoodDistance': -2.3545086025412494, 
    #                 'bias': 1.0, 
    #                 "numInvaders": 8.1962365449863,
    #                 'numGhostsOneStepAway': 7.790394065146265, 
    #                 'eatsFoodNext': 32.84029154417439,
    #                 "goHome": -1.95158804160861,
    #                 "numCapsules": 8.350328555690777,
    #                 "areTrapped": 2.538255086782103}
    
    # self.weights = {"closestCapsuleDistance": 0, 
    #                 "successorFoodScore": 100, 
    #                 "onOffense": 0,
    #                 "closestFoodDistance": -1,
    #                 "bias": 1, 
    #                 "numGhostsOneStepAway": -1, 
    #                 "eatsFoodNext": 1,
    #                 "numInvaders": 1000,
    #                 "closestInvaderDistance": -1,
    #                 "goHome": 0,
    #                 "numCapsules": 1000,
    #                 "areTrapped": 1}

    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    
    # self.startEpisode()
    # if self.episodesSoFar == 0:
    #     print('Beginning %d episodes of Training' % (self.numTraining))

  def getWeights(self, filename):
    if os.stat("./weights/"+filename).st_size == 0:
      self.weights = util.Counter()
    else:
      self.weights = util.Counter()
      test = open("./weights/"+filename, 'r').read()
      parsedDict = json.loads(test)
      for features in parsedDict:
        self.weights[features] = parsedDict[features]
    
  def writeWeights(self, filename):
    f = open("./weights/"+filename,"w+")
    dumps = json.dumps(self.weights)
    f.write(dumps)
    f.close()
        
#   def startEpisode(self):
#     self.lastState = None
#     self.lastAction = None
#     self.episodeRewards = 0.0
  
#   def stopEpisode(self):
#     if self.episodesSoFar < self.numTraining:
#         self.accumTrainingRewards += self.episodeRewards
#     else:
#         self.accumExploitRewards += self.episodeRewards
#     self.episodesSoFar += 1
#     if self.episodesSoFar >= self.numTraining:
#         # Take off the training wheels
#         print('----------------EXPLOIT MODE----------------')
#         self.epsilon = 0.0    # no exploration
#         self.alpha = 0.0      # no learning
            
  def getQValue(self, gameState, action):
    # features vector
    features = self.getFeatures(gameState, action)
    Q_value = 0.0
    for feature in features:
      Q_value += features[feature] * self.weights[feature] # Q(state, action) = featureVector * w
    return Q_value

  # Returns best Q-value based on the best action given a state
  def getValue(self, gameState):
    allowedActions = gameState.getLegalActions(self.index)
    if len(allowedActions) == 0:
      return 0.0
    bestAction = self.getPolicy(gameState)
    bestQValue = self.getQValue(gameState, bestAction)
    return bestQValue

  # computes the best action from the Q-values
  def getPolicy(self, gameState):
    allowedActions = gameState.getLegalActions(self.index)
    if len(allowedActions) == 0:
      return None
    actionQValues = {}
    bestQValue = float("-inf")
    for action in allowedActions:
      targetQValue = self.getQValue(gameState, action)
      actionQValues[action] = targetQValue
      if targetQValue > bestQValue:
        bestQValue = targetQValue
    bestActions = [a for a, v in actionQValues.items() if v == bestQValue]
    # random tie-breaking for actions that have the same Q-value
    return random.choice(bestActions)
   
  def chooseAction(self, gameState):
    allowedActions = gameState.getLegalActions(self.index)
    allowedActions.remove("Stop")
    # allowedActions.remove(Directions.STOP)
    if len(allowedActions) == 0:
      return None

    if self.trainingMode:
      for action in allowedActions:
        self.updateWeights(self.oldGameState, gameState, action)
        
    # Compute the action to take in the current state.  With
    # probability self.epsilon, we should take a random action and
    # take the best policy action otherwise.
    action = None
    if (util.flipCoin(self.epsilon)):
        action = random.choice(allowedActions)
    else:
        action = self.getPolicy(gameState)
  
    foodLeft = len(self.getFood(gameState).asList())
    # if gameState.getAgentState(self.index).numCarrying >= 10 or foodLeft <= 2:
    if foodLeft <= 2:
      bestDist = 9999
      for action in allowedActions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      action = bestAction

    self.oldGameState = gameState
    return action

  def updateWeights(self, oldGameState, gameState, action):
    nextState = self.getSuccessor(gameState, action) # s' = successor state
    features = self.getFeatures(gameState, action) # f(s, a) = feature vector
    R = self.getReward(oldGameState, gameState, nextState) # R(s, a) = reward
    Q = self.getQValue(gameState, action) # Q(s, a) = current Q-value
    V = self.getValue(nextState) # V(s') = max_a' Q(s', a')
    
    # (R(s, a) + gamma * V(s')) - Q(s, a)
    correction = (R + self.gamma * V) - Q
    
    # for each feature i
    for feature in features:
      # performs the weight update (aka stochastic gradient descent)
      # w_i = w_i + alpha * correction * f_i(s, a)
      newWeight = self.alpha * correction * features[feature]
      self.weights[feature] += newWeight

  def getReward(self, oldGameState, gameState, nextState):
    agentPosition = gameState.getAgentPosition(self.index)
    agentState = gameState.getAgentState(self.index)
    reward = 0

    # check if I have updated the score
    if self.getScore(nextState) > self.getScore(gameState):
      reward = (self.getScore(nextState) - self.getScore(gameState)) * 10

    # check if food is eaten in nextState
    foodList = self.getFood(gameState).asList()
    minDistanceToFood = min([self.getMazeDistance(agentPosition, food) for food in foodList])
    # I am 1 step away, will I be able to eat food in nextState?
    if minDistanceToFood == 1:
      nextFoods = self.getFood(nextState).asList()
      if len(foodList) - len(nextFoods) == 1:
        reward = 10

    # check if I am eaten by a ghost in nextState
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer == 0 and a.getPosition() != None]
    if len(ghosts) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in ghosts])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I am eaten by a ghost
          reward = -100
          
    # avoid ghosts
    if len(ghosts) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in ghosts])
      if minDistGhost < 3:
        reward = -minDistGhost * 10
        
    # search on food
    food_matrix = self.getFood(gameState)
    if food_matrix[agentPosition[0]][agentPosition[1]]:
        reward = 10
    
    # computes reward for eating food
    returnedFoodReward = agentState.numReturned - oldGameState.getAgentState(self.index).numReturned
    carryingFoodReward = (agentState.numCarrying - oldGameState.getAgentState(self.index).numCarrying)*0.1
    opponentScore = 0
    opponentFood = 0
    for i in self.opponent:
      opponentScore -= gameState.getAgentState(i).numReturned - oldGameState.getAgentState(i).numReturned
      opponentFood -= (gameState.getAgentState(i).numCarrying - oldGameState.getAgentState(i).numCarrying)*0.1
    reward += returnedFoodReward + carryingFoodReward + opponentScore + opponentFood

    return reward

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    food_matrix = self.getFood(gameState)
    wall_matrix = gameState.getWalls()
    agentState = gameState.getAgentState(self.index)
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()  
    numCarrying = successor.getAgentState(self.index).numCarrying
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition() 
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    # ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
    deadlyGhosts = [a for a in enemies if not a.isPacman and a.scaredTimer == 0 and a.getPosition() != None]    
    deadlyGhostPositions = [g.getPosition() for g in deadlyGhosts]
    agentPosition = gameState.getAgentPosition(self.index)
    x, y = agentPosition
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy) 
    # next_2x, next_2y = int(x + 2*dx), int(y + 2*dy)

    # compute the bias feature
    features["bias"] = 1.0

    # features['successorFoodScore'] = -len(foodList)
    features['goHome'] = numCarrying * self.getMazeDistance(self.start, myPos) # this is the distance to the start
    
    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if not a.isPacman and a.scaredTimer == 0 and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders if self.getMazeDistance(myPos, a.getPosition()) != 0]
      features['closestInvaderDistance'] = min(dists)
    if not gameState.getAgentState(self.index).isPacman:
      features['closestInvaderDistance'] = 0

    # count the number of deadly ghosts one step away from our pacman agent
    features["numGhostsOneStepAway"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, wall_matrix) for g in deadlyGhostPositions)
    # features["numGhostsTwoStepsAway"] = sum((next_2x, next_2y) in Actions.getLegalNeighbors(g, wall_matrix) for g in deadlyGhostPositions)
    
    # if there is no danger of ghosts then add the food feature
    if not features["numGhostsOneStepAway"] and food_matrix[next_x][next_y]:
      features["eatsFoodNext"] = 1.0

    # if len(ghosts) > 0:
    #   minDistance = min([self.getMazeDistance(agentPosition, g) for g in ghosts])
    #   if minDistance < 3:
    #     features["closestGhostDistance"] = minDistance
        
    # compute the distance to the nearest deadly ghost enemy that pose a threat to us
    # if len(deadlyGhostPositions) > 0:
    #   minDeadlyGhostDistance = min([self.getMazeDistance(agentPosition, gPos) for gPos in deadlyGhostPositions])
    #   if minDeadlyGhostDistance < 3:
    #     features["closestGhostDistance"] = minDeadlyGhostDistance

    # compute the closest capsule distance feature
    # capsules = self.getCapsules(gameState)
    # if len(capsules) > 0:
    #   closestCap = min([self.getMazeDistance(agentPosition, cap) for cap in capsules])
    #   features["closestCapsuleDistance"] = closestCap
      
    capsules = self.getCapsules(successor)
    if len(capsules) > 0:
      capCount = 0
      for capsule in capsules:
        capCount += 1
      minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
      features['closestCapsuleDistance'] = minDistance
      features['numCapsules'] = capCount
    else:
      features['closestCapsuleDistance'] = 0
      features['numCapsules'] = 0

    # Compute distance to the nearest food
    # foodList = food_matrix.asList()
    # features['successorFoodScore'] = -len(foodList) # self.getScore(successor)
    # if len(foodList) > 0:
    #     minDistance = min([self.getMazeDistance(agentPosition, food) for food in foodList])
    #     features['closestFoodDistance'] = minDistance

    # compute the distance to the nearest food
    dist = self.searchClosestFood((next_x, next_y), food_matrix, wall_matrix)
    if dist is not None:
      features["closestFoodDistance"] = float(dist) / (wall_matrix.width * wall_matrix.height)
    features.divideAll(10.0)
    
    # Avoid enemy heuristic
    currentPos = gameState.getAgentPosition(self.index)
    if not self.red: # we are in the blue team
        enemyIndx = gameState.getRedTeamIndices()
    else:
        enemyIndx = gameState.getBlueTeamIndices()    
    enemyPos1 = gameState.getAgentPosition(enemyIndx[0])
    enemyPos2 = gameState.getAgentPosition(enemyIndx[1])
    nearestEnemyPos = None
    if enemyPos1 != None:
        nearestEnemyPos = enemyPos1
    if enemyPos2 != None:
        if enemyPos1 != None:
            if self.getMazeDistance(myPos,enemyPos1) > self.getMazeDistance(myPos, enemyPos2):
                nearestEnemyPos = enemyPos2
        else:
            nearestEnemyPos = enemyPos2
    if nearestEnemyPos != None:
        nearest_enemy_distance = self.getMazeDistance(myPos, nearestEnemyPos)
        if(nearest_enemy_distance <= 3):
            move_list = self.getPossibleMoves(gameState, currentPos[0], currentPos[1])
            if len(move_list) <= 1:
                features['areTrapped'] = 1 # deadend
        
    return features

  def getPossibleMoves(self, gameState, x, y): 
      move_list = []
      if not gameState.hasWall(x,y+1):
          move_list.append('North')
      if not gameState.hasWall(x,y-1):
          move_list.append('South')
      if not gameState.hasWall(x+1,y):
          move_list.append('East')
      if not gameState.hasWall(x-1,y):
          move_list.append('West')
      return move_list     

  def searchClosestFood(self, position, food_matrix, wall_matrix):
      queue = [(position[0], position[1], 0)]
      visited = set()
      
      while queue:
          x, y, distance = queue.pop(0)
          
          if (x, y) in visited:
              continue
          
          visited.add((x, y))
          
          # if we find a food at this location, return its distance
          if food_matrix[x][y]:
              return distance
          
          # otherwise, add this location's neighbors to the queue
          neighbors = Actions.getLegalNeighbors((x, y), wall_matrix)
          for neighbor_x, neighbor_y in neighbors:
              queue.append((neighbor_x, neighbor_y, distance + 1))
              
      # no food found
      return None

  # def final(self, state):
  #   # call the super-class final method
  #   CaptureAgent.final(self, state)

  #   #if self.episodesSoFar == self.numTraining:
  #   print("myWeights: ", self.weights)
  #   file = open('myWeights.txt', 'w')
  #   file.write(str(self.weights))

#   def final(self, gameState):
#     self.observationHistory = []

#     if not 'episodeStartTime' in self.__dict__:
#         self.episodeStartTime = time.time()
#     if not 'lastWindowAccumRewards' in self.__dict__:
#         self.lastWindowAccumRewards = 0.0
#     self.lastWindowAccumRewards += self.getScore(gameState)

#     NUM_EPS_UPDATE = 100
#     if self.episodesSoFar % NUM_EPS_UPDATE == 0:
#         print('Reinforcement Learning Status:')
#         windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
#         if self.episodesSoFar <= self.numTraining:
#             trainAvg = self.accumTrainingRewards / float(self.episodesSoFar)
#             print('\tCompleted %d out of %d training episodes' % (
#                   self.episodesSoFar,self.numTraining))
#             print('\tAverage Rewards over all training: %.2f' % (
#                     trainAvg))
#         else:
#             testAvg = float(self.accumExploitRewards) / (self.episodesSoFar - self.numTraining)
#             print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
#             print(')\tAverage Rewards over testing: %.2f' % testAvg)
#         print('\tAverage Rewards for last %d episodes: %.2f'  % (
#                 NUM_EPS_UPDATE,windowAvg))
#         print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
#         self.lastWindowAccumRewards = 0.0
#         self.episodeStartTime = time.time()

#     if self.episodesSoFar == self.numTraining:
#         msg = 'Training Done (turning off epsilon and alpha)'
#         print('%s\n%s' % (msg,'-' * len(msg)))
        
#     self.writeWeights("OptimalWeights.txt")


class DefensiveAgent(CaptureAgent):
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman]
        knowninvaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
        self.lastEatenFoodPosition = self.getPosOfLastEatenFood(gameState)

        if len(invaders) == 0 or gameState.getAgentPosition(self.index) == self.lastEatenFoodPosition or len(
                knowninvaders) > 0:
            self.lastEatenFoodPosition = None

        if len(invaders) < 1:
            if gameState.getAgentState(self.index).numCarrying < 3 and len(
                    self.getFood(gameState).asList()) != 0 and not (
                    self.distToGhost(gameState) is not None and self.distToGhost(gameState)[0] < 4 and
                    self.distToGhost(gameState)[1].scaredTimer < 2):
                search = Search(gameState, self, self.index, "searchfood")
                path = aStarSearch(search, gameState, self.heuristic)
                return path[0]
            else:
                search = Search(gameState, self, self.index, "backhome")
                if len(aStarSearch(search, gameState, self.heuristic)) == 0:
                    return 'Stop'
                else:
                    path = aStarSearch(search, gameState, self.heuristic)
                    return path[0]
        else:
            if len(knowninvaders) == 0 and self.lastEatenFoodPosition is not None and gameState.getAgentState(
                    self.index).scaredTimer == 0:
                search = Search(gameState, self, self.index, "searchlasteatenfood")
                path = aStarSearch(search, gameState, self.heuristic)
                return path[0]

            if len(knowninvaders) > 0 and gameState.getAgentState(self.index).scaredTimer == 0:
                search = Search(gameState, self, self.index, "searchinvader")
                path = aStarSearch(search, gameState, self.heuristic)
                return path[0]
        return random.choice(actions)

    def distToGhost(self, gameState):
        myPosition = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
        if len(ghosts) > 0:
            min_dist = float('inf')
            for a in ghosts:
                temp = self.getMazeDistance(myPosition, a.getPosition())
                if temp < min_dist:
                    min_dist = temp
                    ghostState = a
            return [min_dist, ghostState]
        else:
            return None

    def getBoundary(self, gameState):
        if self.red:
            i = (int)(gameState.data.layout.width / 2 - 1)
        else:
            i = (int)(gameState.data.layout.width / 2 + 1)
        boundaries = [(i, j) for j in range(gameState.data.layout.height)]
        validPositions = []
        for i in boundaries:
            if not gameState.hasWall(i[0], i[1]):
                validPositions.append(i)
        return validPositions

    def getPosOfLastEatenFood(self, gameState):
        if len(self.observationHistory) > 1:
            prevState = self.getPreviousObservation()
            prevFoodList = self.getFoodYouAreDefending(prevState).asList()
            currentFoodList = self.getFoodYouAreDefending(gameState).asList()
            if len(prevFoodList) != len(currentFoodList):
                for food in prevFoodList:
                    if food not in currentFoodList:
                        return food
        return None

    def heuristic(self, state, gameState):
        heuristic = 0
        nearest_ghost_distance = -1
        myPosition = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
        if len(ghosts) > 0:
            dists = [self.getMazeDistance(myPosition, a.getPosition()) for a in ghosts]
            nearest_ghost_distance = min(dists)
        if nearest_ghost_distance != -1:
            ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer < 2 and a.getPosition() is not None]

            if ghosts and len(ghosts) > 0:
                ghost_positions = [ghost.getPosition() for ghost in ghosts]
                ghost_distances = [self.getMazeDistance(state, ghost_position) for ghost_position in ghost_positions]
                ghost_dist = min(ghost_distances)

                if ghost_dist < 2:
                    heuristic = 1.0 / (ghost_dist + 0.0001)

        return heuristic

def aStarSearch(search, gameState, heuristic):
    start_state = search.startState
    fringe = PriorityQueue()
    h = heuristic(start_state, gameState)
    g = 0
    f = g + h
    start_node = (start_state, [], g)
    fringe.push(start_node, f)
    explored = []
    while not fringe.isEmpty():
        current_node = fringe.pop()
        state = current_node[0]
        path = current_node[1]
        current_cost = current_node[2]
        if state not in explored:
            explored.append(state)
            if search.isGoalState(state):
                return path
            successors = search.getSuccessors(state)
            for successor in successors:
                current_path = list(path)
                successor_state = successor[0]
                move = successor[1]
                g = successor[2] + current_cost
                h = heuristic(successor_state, gameState)
                if successor_state not in explored:
                    current_path.append(move)
                    f = g + h
                    successor_node = (successor_state, current_path, g)
                    fringe.push(successor_node, f)
    return []
class Search:
    def __init__(self, gameState, agent, agentIndex, goal=""):
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        self.walls = gameState.getWalls()
        self.goal = goal
        self.homeBoundary = agent.getBoundary(gameState)

        self.enemies = [gameState.getAgentState(agentIndex) for agentIndex in agent.getOpponents(gameState)]
        self.invaders = [a for a in self.enemies if a.isPacman and a.getPosition is not None]
        if len(self.invaders) > 0:
            self.invadersPosition = [invader.getPosition() for invader in self.invaders]
        else:
            self.invadersPosition = None

        if goal == "searchlasteatenfood":
            self.lasteatenfoodpos = agent.lastEatenFoodPosition

    def isGoalState(self, state):
        if self.goal == "searchfood":
            return state in self.food.asList()
        elif self.goal == "backhome":
            return state in self.homeBoundary or state in self.capsule
        elif self.goal == "searchinvader":
            return state in self.invadersPosition
        elif self.goal == "searchlasteatenfood":
            return state == self.lasteatenfoodpos

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = 1
                successors.append((nextState, action, cost))
        return successors
