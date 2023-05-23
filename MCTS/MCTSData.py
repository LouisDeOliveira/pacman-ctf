import numpy as np

from capture import GameState
from MCTS.Node import Node

class MCTSData:    
    def __init__(self, state:GameState, player:int, UCB1:float, sim_time:float = np.inf, sim_number:int = 1_000_000, cutoff:int = 0) -> None:
        self.state = state
        self.player = player
        self.UCB1 = UCB1
        self.sim_time = sim_time
        self.sim_number = sim_number
        self.cutoff = cutoff
        
        self.root:Node
        self.players:np.ndarray
        self.distances: np.ndarray
        self.food:list
        self.distributions:list[dict]
        self.max_distance:int
        self.defender_threshold:int
        self.choke_points:list[list[int]]
        
        self.index_mapping = {0: 0, 1: 0, 2: 1, 3: 1}
        

    def get_food_locations(self) -> None:
        got_food = self.state.getBlueFood() if self.state.isOnRedTeam(self.player) else self.state.getRedFood()
        self.food = []
        for x in range(got_food.width):
            for y in range(got_food.height):
                if got_food[x][y]: self.food.append((x,y))
        
        
    def calculate_threshold(self) -> None:
        if self.state.isOnRedTeam(self.player): score = self.state.getScore()
        else: score = -self.state.getScore()

        self.defender_threshold = 1
        if score >= 3:
            self.defender_threshold = 10


    def find_choke_points(self):
        self.choke_points = []
        list = []
        xb, xr = self.state.data.layout.walls.width//2-1, self.state.data.layout.walls.width//2
        for y in range(self.state.data.layout.walls.height):
            if not self.state.data.layout.walls[xr][y] and not self.state.data.layout.walls[xb][y]: 
                list.append(y)
            else:
                if list: self.choke_points.append(sum(list)//len(list))
                list = []