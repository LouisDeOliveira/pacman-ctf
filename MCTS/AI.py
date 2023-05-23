import time

import numpy as np

from capture import GameState
from MCTS.Node import Node
from MCTS.MCTSData import MCTSData


def MCTSfindMove(data:MCTSData) -> str:
    moves = data.state.getLegalActions(data.player)
    removeStop(moves)
    if not moves: return None
    
    # starting_position = data.state.getAgentPosition(data.player)
    # furthest_away_distance = 0
    # furthest_away_position = starting_position
    # max_depth = 0
    
    startTime = time.process_time()
    for _ in range(data.sim_number):
        if time.process_time() - startTime > data.sim_time:
            # distance = [x-y for x, y in zip(furthest_away_position, starting_position)]
            # team = "Red" if currentState.isOnRedTeam(data.player) else "Blue"
            # role = "Defensive" if data.player <=  data.defender_threshold else "Offensive"
            # print(f"maximum depth: {max_depth}, checked {distance} from starting position in {starting_position}, in team {team} as {role}")
            # print(f"Hypothisys is {np.round(evaluationHeuristic(GameState(data.state), data, data.player),2)}")
            # printData(data.root)
            return data.root.chooseBestChild().move

        currentState = GameState(data.state)
        current = data.root
        # current_depth = 0
        
        # Tree traverse
        while len(current.children) > 0:
            # current_depth += 1
            current = current.selectChild(data.UCB1)
            currentState = currentState.generateSuccessor(current.player, current.move)

            # returns a move if visits exceeds half of total simulations
            if current.visits >= 0.5*data.sim_number:
                # printData(data.root)
                return current.move

        # if (current_depth > max_depth): max_depth = current_depth

        # Expand tree if current has been visited and isn't a terminal node
        if current.visits > 0 and not currentState.isOver():
            moves = currentState.getLegalActions(current.nextPlayer(data.players))
            if data.state.isOnRedTeam(data.player) == data.state.isOnRedTeam(current.nextPlayer(data.players)): removeStop(moves) # and current.nextPlayer(data.players) > data.defender_threshold:
            current.makeChildren(current.nextPlayer(data.players), moves)
            current = current.selectChild(data.UCB1)
            currentState = currentState.generateSuccessor(current.player, current.move)
            
            # if (current.player == data.player):
            #     furthest_away_distance, furthest_away_position = calculate_depth(
            #         currentState, starting_position, furthest_away_distance, furthest_away_position, data.player)

        # Rollout
        result = evaluationHeuristic(currentState, data, current.player)

        # Backpropagate
        current.backpropagate(data.state, result)

    # printData(data.root)
    return data.root.chooseBestChild().move


def evaluationHeuristic(gameState: GameState, data:MCTSData, current_player:int) -> tuple[float]:
    ### REASONABLE HEURISTIC. Maximize your score. Maximize how much you're carrying but less so than how much you deposited.
    ### Minimize how much food your opponent has captured but it's harder so dont spend to much time on it.
    my_pos = gameState.getAgentPosition(current_player)
    # carried and scored food
    foodCapturedByRed  = gameState.data.layout.totalFood/2 - len(gameState.getBlueFood().asList())
    foodCapturedByBlue = gameState.data.layout.totalFood/2 - len(gameState.getRedFood().asList())    
    score = gameState.getScore()
    
    home_penalty_red, home_penalty_blue = home_penalty(gameState)
        
    # summing
    heuristic_red  =  score*2 + foodCapturedByRed/2 - foodCapturedByBlue/2 - home_penalty_red + home_penalty_blue
    heuristic_blue = -score*2 - foodCapturedByRed/2 + foodCapturedByBlue/2 + home_penalty_red - home_penalty_blue

    # if enemy: return here
    if gameState.isOnRedTeam(data.player) != gameState.isOnRedTeam(current_player):
        return heuristic_red, heuristic_blue

    team = f"Red{current_player<=data.defender_threshold}" if gameState.isOnRedTeam(current_player) else f"Blue{current_player<=data.defender_threshold}"
    match team:
        case "RedTrue": ### DEFENSIVE
            offensive_enemy, _ = enemies_distances(gameState, data, my_pos, current_player)
            heuristic_red  += (1-offensive_enemy/data.max_distance)*10
        case "RedFalse": ### OFFENSIVE
            food = closest_food(data, my_pos)
            num_carrying = gameState.getAgentState(current_player).numCarrying
            home = 0
            if num_carrying > 2 or (num_carrying > 0 and len(data.food) <= 2): 
                home_dist = distance_home(gameState, data, my_pos, current_player)
                home = (1-home_dist/data.max_distance)/2
            heuristic_red  += (1-food/data.max_distance)*3 + home
        case "BlueTrue":
            offensive_enemy, _ = enemies_distances(gameState, data, my_pos, current_player)
            heuristic_blue += (1-offensive_enemy/data.max_distance)*10
        case "BlueFalse":
            food = closest_food(data, my_pos)
            num_carrying = gameState.getAgentState(current_player).numCarrying
            home = 0
            if num_carrying > 2 or (num_carrying > 0 and len(data.food) <= 2): 
                home_dist = distance_home(gameState, data, my_pos, current_player)
                home = (1-home_dist/data.max_distance)/2
            heuristic_blue += (1-food/data.max_distance)*3 + home
        case _:
            raise Exception
    return heuristic_red, heuristic_blue


def home_penalty(gameState:GameState):
    # penalty if you are on home row
    home_penalty_red, home_penalty_blue = 0, 0
    for i in range(4):
        pos = gameState.getAgentPosition(i)
        if not pos: continue
        if gameState.isOnRedTeam(i)     and pos[0] == 1: home_penalty_red += 7
        if not gameState.isOnRedTeam(i) and pos[0] == gameState.data.layout.width-2: home_penalty_blue += 7
    return home_penalty_red, home_penalty_blue

def closest_food(data:MCTSData, my_pos):
    food = data.max_distance
    data.get_food_locations()
    for food_location in data.food:
        food = min(food, data.distances[my_pos[0]][my_pos[1]][food_location[0]][food_location[1]])
    return food

def enemies_distances(gameState:GameState, data:MCTSData, my_pos, current_player):
    offensive_enemy = defensive_enemy = data.max_distance
    ys = [gameState.data.layout.walls.height//2]*2
    min_x = [data.max_distance]*2
    for i, distribution in enumerate(data.distributions):
        for pos in distribution:
            if (pos[0] < data.middle and gameState.isOnRedTeam(data.player)) or (pos[0] > data.middle and not gameState.isOnRedTeam(data.player)):
                offensive_enemy = min(offensive_enemy, data.distances[my_pos[0]][my_pos[1]][pos[0]][pos[1]])
            else:
                # defensive_enemy = min(defensive_enemy, data.distances[my_pos[0]][my_pos[1]][pos[0]][pos[1]])
                if abs(data.middle - pos[0]) < min_x[i]:
                    min_x[i] = abs(data.middle - pos[0])
                    ys[i] = pos[1]
                    
    # if there is no offensive enemy
    if offensive_enemy == data.max_distance:
        # go to middle column on your side 
        if gameState.isOnRedTeam(current_player): home_col = gameState.data.layout.walls.width//2 - 1
        else: home_col = gameState.data.layout.walls.width//2
        # go to same rank as closest opponent if only one defender, and otherwise shadow one opponent each
        if data.defender_threshold == 1: y = ys[min_x.index(min(min_x))]
        else: y = ys[data.index_mapping[current_player]]
        
        offensive_enemy = 0
        # while loop here so (home_col, y) is not wall
        while offensive_enemy == 0:
            offensive_enemy = data.distances[my_pos[0]][my_pos[1]][home_col][y]
            y = (y+1) % gameState.data.layout.walls.height
    return offensive_enemy, defensive_enemy

def distance_home(gameState:GameState, data:MCTSData, my_pos, current_player):
    home_dist = data.max_distance
    if gameState.isOnRedTeam(current_player): home_col = gameState.data.layout.walls.width//2 - 1
    else: home_col = gameState.data.layout.walls.width//2

    for y in range(gameState.data.layout.walls.height):
        if gameState.data.layout.walls[home_col][y]: continue
        if data.distances[my_pos[0]][my_pos[1]][home_col][y] < home_dist: 
            home_dist = data.distances[my_pos[0]][my_pos[1]][home_col][y]
    return home_dist

def capsule_distance(gameState:GameState, data:MCTSData, my_pos):
    own_capsule = data.max_distance
    own_cap = gameState.getRedCapsules() if gameState.isOnRedTeam(data.player) else gameState.getBlueCapsules()
    if own_cap: own_capsule = data.distances[my_pos[0]][my_pos[1]][own_cap[0][0]][own_cap[0][1]]
    return own_capsule


def enemy_heuristic(gameState:GameState, data:MCTSData, my_pos, current_player):
    ### IF ENEMY IS NOT GHOST YOU ARE ON THE OTHER SIDE THEY SHOULD CHASE YOU
    root_pos = gameState.getAgentPosition(data.player)
    heuristic_red = heuristic_blue = 0
    if gameState.getAgentState(data.player).isPacman and gameState.getAgentState(current_player).scaredTimer <= 0:
        distance_to_root = data.distances[my_pos[0]][my_pos[1]][root_pos[0]][root_pos[1]]
        if gameState.isOnRedTeam(current_player):
            heuristic_red += (1-distance_to_root/data.max_distance)*5
            heuristic_blue -= (1-distance_to_root/data.max_distance)*5
        else:
            heuristic_red -= (1-distance_to_root/data.max_distance)*5
            heuristic_blue + (1-distance_to_root/data.max_distance)*5
    return heuristic_red, heuristic_blue


def removeStop(list:list) -> None:
    try: list.remove('Stop')
    except ValueError: pass
    
    
def calculate_depth(gameState:GameState, starting_position, furthest_away_distance:int, furthest_away_position:int, rootPlayer:int):
    new_pos = gameState.getAgentPosition(rootPlayer)
    difference = (new_pos[0]-starting_position[0],new_pos[1]-starting_position[1])
    dist = np.sqrt(difference[0]**2+difference[1]**2)
    if(dist>furthest_away_distance):
        furthest_away_position = gameState.getAgentPosition(rootPlayer)
        furthest_away_distance = dist
    
    return furthest_away_distance, furthest_away_position



def printData(root: Node) -> None:
    visits, val, p = root.visits, root.value, root.player
    print(
        f'root; player: {p}, rollouts: {visits}, value: {round(val*1, 2)}, vinstprocent: {round((visits+val)*50/visits, 2)}%')
    print('children;')
    print('visits:', end=' ')
    childVisits = [child.visits.tolist() for child in root.children]
    childVisits.sort(reverse=True)
    print(childVisits)
    print('')
    
    
