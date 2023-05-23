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
import numpy as np
from util import nearestPoint
from util import manhattanDistance


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'SwitchAgent', second = 'SwitchAgent'):
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


#################
#               #
# Dynamic Agent # 
#               #
#################

DEPTH = int(11) # adversarial search tree depth

class DynamicAgent(CaptureAgent):
  """
  Parent class for multi agent search agents
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.depth = DEPTH
    CaptureAgent.registerInitialState(self, gameState)

    self.currentMission = ""
    self.currentMissionCounter = 0
    self.pointToGoTo = (0, 0)
    self.walls = gameState.getWalls().asList()
    self.middleX = gameState.data.layout.width // 2
    if self.start[0] < self.middleX:
      self.middleX -= 1
    
    self.previouselyExistingFood = \
      self.getFoodYouAreDefending(gameState).asList()
    
    # Switch 
    indices = []
    self.enemyIndices = []
    if (gameState.isOnRedTeam(self.index)):
      indices = gameState.getRedTeamIndices()
      self.enemyIndices = gameState.getBlueTeamIndices()
    
    else:
      indices = gameState.getBlueTeamIndices()
      self.enemyIndices = gameState.getRedTeamIndices()

    defensiveIndex = min(indices)
    offensiveIndex = max(indices)

    self.isDefense = self.index==defensiveIndex
    self.isOffense = self.index==offensiveIndex


  def returnToHome(self, gameState, pointPercentage):
     '''Go back to home side if carrying certain percentage of pellets'''
     threshold = pointPercentage * len(self.getFood(gameState).asList())
     if gameState.getAgentState(self.index).numCarrying >= threshold:
        possibleActions = gameState.getLegalActions(self.index)
        possibleActions = self.removeStopFromActions(possibleActions)
        action_scores = [self.evaluateGoHome(gameState,action) for action in possibleActions]

        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]

        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]
     
     else:
        return None
      

  def evaluateGoHome(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    successor = self.getSuccessor(gameState,action)
    pacmanPosition = successor.getAgentPosition(self.index)
    ghostWeight = 5000.0
    ghostFreeWeight = 9000.0
    homeWeight = 200.0
    
    # Calculate distance to nearest ghost
    ghostScore = 0
    ghostDistances = []
    for agent in self.enemyIndices:
        agentPos = successor.getAgentPosition(agent)
        if agentPos is not None and not successor.getAgentState(agent).isPacman:
            ghostDistances.append(self.getMazeDistance(pacmanPosition, agentPos))

    for ghost in ghostDistances:
        if ghost <= 1.0:
            ghostScore = float('inf')
        else:
            ghostScore += 1.0 / ghost

    # Reward position with no ghosts
    ghostFreeScore = 1.0 / float(len(ghostDistances)+1.0)

    # Calculate distance home
    homeScore = 0
    homeBase = self.findHomeBase(gameState)
    homeDistance = self.getMazeDistance(pacmanPosition,homeBase)
    if homeDistance > 0:
       homeScore = 1.0/homeDistance

    score = homeWeight * homeScore + ghostFreeWeight * ghostFreeScore - \
      ghostWeight * ghostScore

    return score

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    successor = self.getSuccessor(gameState,action)
    newPos = successor.getAgentPosition(self.index)
    ghostWeight = 6000.0
    ghostFreeWeight = 9000.0

    # Calculate distance to nearest ghost
    ghostScore = 0
    scaredGhostScore = 0
    ghostDistances = []
    scaredGhostDistances = []
    for agent in self.enemyIndices:
        agentPos = successor.getAgentPosition(agent)
        if agentPos is not None and not \
          successor.getAgentState(agent).isPacman: 
            if successor.getAgentState(agent).scaredTimer <= 3:
              ghostDistances.append(self.getMazeDistance(newPos, agentPos))
            else:
               scaredGhostDistances.append(\
                  self.getMazeDistance(newPos,agentPos))
    
    # calculate ghost score
    for ghost in ghostDistances:
        if ghost <= 1.0:
            ghostScore = 1000
        else:
            ghostScore += 1.0 / ghost

    # calculate scared ghost score
    for scaredGhost in scaredGhostDistances:
       if scaredGhost <= 3.0:
            scaredGhostScore = 10000

    # Reward position with no ghosts
    ghostFreeScore = 1.0 / float(len(ghostDistances)+1.0)

    score = features * weights + ghostFreeScore * ghostFreeWeight \
      + ghostWeight * scaredGhostScore - ghostWeight * ghostScore
    return score
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': 10000}
  
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
    
  def removeStopFromActions(self, actions):
      if 'Stop' in actions:
        actions.remove('Stop')
      return actions
  
  def bestActionToGetToPoint(self, gameState, point):
      bestDistance = float('inf')
      bestAction = ""
      for action in gameState.getLegalActions(self.index):
        d = self.getMazeDistance(gameState.generateSuccessor(self.index, action).getAgentState(self.index).getPosition(), point)
        if d < bestDistance:
          bestDistance = d
          bestAction = action

      return bestAction
  
  def findHomeBase(self, gameState):
    ownSide = self.getOwnSide(gameState)
    currentPos = gameState.getAgentPosition(self.index)
    minDist = float('inf')
    closestPoint = self.start

    for point in ownSide:
        dist = self.getMazeDistance(currentPos,point)
        if dist < minDist:
            minDist = dist
            closestPoint = point

    #self.debugDraw(closestPoint,[0.0, 0.0, 1.0],True)
    return closestPoint
  
  def getOwnSide(self, gameState):
    """
    Returns the list of positions on the side of the board belonging to the agent.
    """
    walls = gameState.getWalls()
    ownSide = []
    # Blue team
    if not self.red:
        x_range = range(1 + walls.width // 2, walls.width)
    
    # Red team
    else:
        x_range = range(-1 + walls.width // 2)
    for x in x_range:
        for y in range(walls.height):
            if not walls[x][y]:
                ownSide.append((x, y))
    return ownSide


              
  
################
#              #
# Switch Agent # 
#              #
################

# inherets from Dynamic Agent but can switch between defensive/offensive
class SwitchAgent(DynamicAgent):

  def chooseAction(self, gameState):
    # Check offense or defense
    return self.chooseActionDefensiveBehaviour(gameState)
    


  def chooseActionIsOffense(self, gameState):
     # Expectimax/Alpha-beta offense for pacman
      if (gameState.getAgentState(self.index).isPacman):
        action = self.returnToHome(gameState, \
                                   self.determinePointPercentage(gameState))
        if action is not None:
           return action
        else:
           return self.chooseActionAlphaBeta(gameState)
           #return self.chooseActionExpectimax(gameState)

      # Reflex offense for ghost
      else:
        return self.chooseActionReflex(gameState)
  
  def determinePointPercentage(self, gameState):
      enemies = [gameState.getAgentState(i) for i in \
                self.getOpponents(gameState)]
      seenGhosts = [enemy.getPosition() for enemy in enemies if \
                   not enemy.isPacman and enemy.getPosition() != None]
     #for agent in self.enemyIndices:
        # greedy food search
      if gameState.getAgentState(self.enemyIndices[0]).scaredTimer > 0 \
        and gameState.getAgentState(self.enemyIndices[1]).scaredTimer > 0:
         return 0.75
      if len(seenGhosts)==0:
         return 0.3
      else:
         return 0.05
      
      
  #####################
  #                   #
  # ALPHA-BETA ACTION # 
  #                   #
  #####################

  def chooseActionAlphaBeta(self, gameState):
    
    """
    Chooses pacman action based on alpha-beta pruning expectimax with DEPTH
    """
    possibleActions = gameState.getLegalActions(self.index)
    possibleActions = self.removeStopFromActions(possibleActions)
    action_scores = [self.alpha_beta(0, 0, self.getSuccessor(gameState,action),float('inf'),-float('inf')) for action in possibleActions]

    max_action = max(action_scores)
    max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]

    chosenIndex = random.choice(max_indices)

    return possibleActions[chosenIndex]
  
  def alpha_beta(self, agent, depth, gameState, alpha, beta):
    if gameState.isOver() or depth == self.depth:
        actions = self.getOrderedActions(gameState)
        #actions = self.removeStopFromActions(actions)
        max_score = -float('inf')
        for action in actions:
            score = self.evaluate(gameState, action)
            max_score = max(max_score, score)
            successor = self.getSuccessor(gameState,action)
            actionPos = gameState.getAgentPosition(self.index)
            #self.debugDraw(actionPos,[0.0, 1.0, 0.0],True)
        return max_score

    if agent == self.index:  # maximize for our team
        actions = self.getOrderedActions(gameState)
        #actions = self.removeStopFromActions(actions)
        max_score = -float('inf')
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            score = self.alpha_beta((agent + 1) % gameState.getNumAgents(), depth, successor, alpha, beta)
            max_score = max(max_score, score)
            alpha = max(alpha, max_score)
            if beta <= alpha:
                break  # beta cut-off
        return max_score

    else:  # minimize for other team
        if (not gameState.getAgentState(agent).isPacman) and (gameState.getAgentPosition(agent) is tuple):  # minimize for ghosts
            #actions = gameState.getLegalActions(agent)
            #actions = self.removeStopFromActions(actions)
            actions = self.getMinimumOrderedActions(gameState,agent)
            min_score = float('inf')
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                score = self.alpha_beta((agent + 1) % gameState.getNumAgents(), depth, successor, alpha, beta)
                min_score = min(min_score, score)
                beta = min(beta, min_score)
                if beta <= alpha:
                    break  # alpha cut-off
            return min_score

        else:  # ignore Pacman agents
            if gameState.getAgentPosition(agent) is tuple:
                #actions = gameState.getLegalActions(agent)
                #actions = self.removeStopFromActions(actions)
                actions = self.getOrderedActions(gameState)
                avg_score = 0
                for action in actions:
                    actionPos = gameState.getAgentPosition(self.index)
                    #self.debugDraw(actionPos,[0.0, 1.0, 0.0],True)
                    successor = self.getSuccessor(gameState, action)
                    score = self.alpha_beta((agent + 1) % gameState.getNumAgents(), depth, successor, alpha, beta)
                    avg_score += score
                return avg_score / len(actions)
            
            else:  # ghost is scared or has just died
                #actions = gameState.getLegalActions(self.index)
                #actions = self.removeStopFromActions(actions)
                actions = self.getOrderedActions(gameState)
                max_score = -float('inf')
                for action in actions:
                    actionPos = gameState.getAgentPosition(self.index)
                    #self.debugDraw(actionPos,[0.0, 1.0, 0.0],True)
                    successor = self.getSuccessor(gameState, action)
                    score = self.alpha_beta((self.index + 1) % gameState.getNumAgents(), depth+1, successor, alpha, beta)
                    max_score = max(max_score, score)
                    alpha = max(alpha, max_score)
                    if beta <= alpha:
                        break  # beta cut-off
                return max_score


  #####################
  #                   #
  # EXPECTIMAX ACTION #
  #                   #
  #####################
  def chooseActionExpectimax(self, gameState):
    
    """
    Chooses pacman action based on expectimaz with DEPTH
    """
    possibleActions = gameState.getLegalActions(self.index)
    action_scores = [self.expectimax(0, 0, self.getSuccessor(gameState,action)) for action in possibleActions]

    max_action = max(action_scores)
    max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]

    chosenIndex = random.choice(max_indices)

    return possibleActions[chosenIndex]

  def expectimax(self, agent, depth, gameState):
    if gameState.isOver() or depth == self.depth:
        actions = gameState.getLegalActions(self.index)
        max_score = -float('inf')
        for action in actions:
            score = self.evaluate(gameState,action)
            max_score = max(max_score, score)

        return max_score

    if agent == self.index:  # maximize for our team
        actions = gameState.getLegalActions(agent)
        max_score = -float('inf')
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            score = self.expectimax((agent + 1) % gameState.getNumAgents(), depth, successor)
            max_score = max(max_score, score)

        return max_score

    else:  # minimize for other team
        if (not gameState.getAgentState(agent).isPacman) and (gameState.getAgentPosition(agent) is tuple):  # minimize for ghosts
            actions = gameState.getLegalActions(agent)
            num_actions = len(actions)
            min_score = float('inf')
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                score = self.expectimax((agent + 1) % gameState.getNumAgents(), depth, successor)
                min_score = min(min_score, score)

            return min_score

        else:  # ignore Pacman agents
            if gameState.getAgentPosition(agent) is tuple:
              avg_score = 0
              for action in actions:
                  successor = self.getSuccessor(gameState, action)
                  score = self.expectimax((agent + 1) \
                                          % gameState.getNumAgents(), depth,  successor)
                  avg_score += score

              return avg_score / num_actions
            
            else: 
                actions = gameState.getLegalActions(self.index)
                max_score = -float('inf')
                for action in actions:
                    successor = self.getSuccessor(gameState, action)
                    score = self.expectimax((self.index + 1) % gameState.getNumAgents(),depth+1, successor)
                    max_score = max(max_score, score)

                return max_score
            
  def getMinimumOrderedActions(self, gameState, agent):
    actions = gameState.getLegalActions(agent)
    pacmanPosition = gameState.getAgentPosition(self.index)
    pacmanDistances = {action: [agent.getMazeDistance(\
       agent.getSuccessorPosition(gameState,action), pacmanPosition)] for \
        action in actions}
    orderedActions = sorted(actions, key=lambda action: pacmanDistances[action])
    return orderedActions

  def getOrderedActions(self, gameState):
    actions = gameState.getLegalActions(self.index)
    foodPositions = self.getFood(gameState).asList()
    if len(foodPositions)>0:
      actionDistances = {action: min([self.getMazeDistance(\
        self.getSuccessorPosition(gameState,action), food) for food in \
          foodPositions]) for action in actions}
    else:
       actionDistances = {action: min([self.getMazeDistance(\
        self.getSuccessorPosition(gameState,action), \
          self.findHomeBase(gameState))]) for action in actions}
    orderedActions = sorted(actions, key=lambda action: actionDistances[action])
    return orderedActions
  
  
  def getSuccessorPosition(self, gameState, action):
     successor = self.getSuccessor(gameState,action)
     return successor.getAgentPosition(self.index)
            

  
  #################
  #               #
  # REFLEX ACTION #
  #               #
  #################
  def chooseActionReflex(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)
  
  def evaluateReflex(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}
  
  
  ###########################
  #                         #
  # DEFENSIVE BAHVIOUR TREE # 
  #                         #
  ###########################

  def point_exists(self, p1, p2):
    point_exists = False
    try:
      self.getMazeDistance(p1, p2)
      point_exists = True
    except:
        point_exists = False
    return point_exists
     

  def patrolFood(self, gameState):
      if self.currentMission != "PATROL":
        self.currentMission = "PATROL"
        self.currentMissionCounter = 0

      existingFoodPositions = self.getFoodYouAreDefending(gameState).asList()

      if self.isDefense and not gameState.getAgentState(self.index).isPacman and not self.checkForLead(gameState):
        if len(existingFoodPositions) > 0:
          max_y = gameState.data.layout.height - 5
          min_y = 5

          pointToGoTo = [self.middleX, min_y]
          while not self.point_exists(tuple(pointToGoTo), gameState.getAgentPosition(self.index)):
            if self.start[0] < self.middleX:
              pointToGoTo[0] -= 1
            else:
              pointToGoTo[0] += 1

          self.pointToGoTo = tuple(pointToGoTo)
          if gameState.getAgentPosition(self.index) == self.pointToGoTo:
            pointToGoTo = [self.middleX, max_y]
            while not self.point_exists(tuple(pointToGoTo), gameState.getAgentPosition(self.index)):
              if self.start[0] < self.middleX:
                pointToGoTo[0] -= 1
              else:
                pointToGoTo[0] += 1


          self.pointToGoTo = tuple(pointToGoTo)
      
      else:
        if len(existingFoodPositions) > 0:
          if self.isOffense:
            max_y = gameState.data.layout.height - 5
            min_y = gameState.data.layout.height // 2
          else:
            max_y = (gameState.data.layout.height // 2) - 1
            min_y = 5
             

          pointToGoTo = [self.middleX, min_y]
          while not self.point_exists(tuple(pointToGoTo), gameState.getAgentPosition(self.index)):
            if self.start[0] < self.middleX:
              pointToGoTo[0] -= 1
            else:
              pointToGoTo[0] += 1

          self.pointToGoTo = tuple(pointToGoTo)
          if gameState.getAgentPosition(self.index) == self.pointToGoTo:
            pointToGoTo = [self.middleX, max_y]
            while not self.point_exists(tuple(pointToGoTo), gameState.getAgentPosition(self.index)):
              if self.start[0] < self.middleX:
                pointToGoTo[0] -= 1
              else:
                pointToGoTo[0] += 1


          self.pointToGoTo = tuple(pointToGoTo)

      


  def chooseActionDefensiveBehaviour(self, gameState):
      startTime = time.time()
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      invaders = [enemy.getPosition() for enemy in enemies if enemy.isPacman]
      seenInvaders = [enemy.getPosition() for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
      seenGhosts = [enemy.getPosition() for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
      foodEaten = list(set(self.previouselyExistingFood) - set(self.getFoodYouAreDefending(gameState).asList()))

      """ ------------------- Behavior Tree ------------------- """
      if gameState.getAgentState(self.index).scaredTimer > 0:
        # If scared, invade!
          return self.chooseActionIsOffense(gameState)
           

      # We are losing, one attack
      if self.isOffense and not self.checkForLead(gameState):
         return self.chooseActionIsOffense(gameState)
      
      # We are under attack, defend
      if len(invaders)!=0:
        if len(seenInvaders) != 0: # Invader exists, and is visible -> chase invader
          if self.currentMission != "CHASE":
            self.currentMission = "CHASE"
            self.currentMissionCounter = 0
          if self.isOffense and len(seenInvaders)==2:
            self.pointToGoTo = seenInvaders[1]
          else:
            self.pointToGoTo = seenInvaders[0]
        
        else: 
          if len(foodEaten) != 0: # Invader exists, is not visible, but food has been eaten -> go to eaten food
            if self.currentMission != "GO TO EATEN FOOD":
              self.currentMission = "GO TO EATEN FOOD"
              self.currentMissionCounter = 0
            self.pointToGoTo = foodEaten[0]
            
          else:
            if len(self.getCapsulesYouAreDefending(gameState)) != 0: # Invader exists, is not visible -> try go guard capsule
              if self.currentMission != "GUARD CAPSULE":
                self.currentMission = "GUARD CAPSULE"
                self.currentMissionCounter = 0
              capsules = self.getCapsulesYouAreDefending(gameState)
              capsules.sort(key=lambda x: abs(x[0] - self.middleX))
              if self.isOffense and len(capsules)==2:
                self.pointToGoTo = capsules[1]
              
              elif self.isOffense and len(capsules)==1:
                if self.currentMission != "PATROL" or \
                    gameState.getAgentPosition(self.index) == self.pointToGoTo: # No need for both to guard the same capsule -> guard food
                    return self.chooseActionIsOffense(gameState)
                  

              else:
                self.pointToGoTo = capsules[0]
              

            elif self.currentMission != "PATROL" or  gameState.getAgentPosition(self.index) == self.pointToGoTo: # Invader exists, is not visible, capsule don't exist -> guard food
              self.patrolFood(gameState)
      
      elif self.currentMission != "PATROL" or  gameState.getAgentPosition(self.index) == self.pointToGoTo:
        self.patrolFood(gameState)


      """ ----------------------------------------------------- """

      self.currentMissionCounter += 1
      self.previouselyExistingFood = self.getFoodYouAreDefending(gameState).asList()
      return self.bestActionToGetToPoint(gameState, self.pointToGoTo)
  
  # Check if we are winning with strong lead to play defense
  def checkForLead(self,gameState):
     if (self.red and self.getScore(gameState)<=10) or \
        (not self.red and self.getScore(gameState)>=-10) or \
          gameState.getAgentState(self.index).scaredTimer>0:
          return False
     else:
        return True

