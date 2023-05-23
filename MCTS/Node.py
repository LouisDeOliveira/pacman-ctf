import random
import numpy as np
from capture import GameState


class Node:
    def __init__(self, player: int, move:str=None, parent=None) -> None:
        self.value = np.array(0, np.float32)
        self.visits = np.array(0, np.int32)
        self.parent:int = parent
        self.children:list["Node"] = []
        self.move:str = move
        self.player:int = player                    # self.player makes self.move


    def makeChildren(self, player: int, moves: list) -> None:
        """Makes a child node for every possible move"""
        for move in moves:
            child = Node(player, move, parent=self)
            self.children.append(child)

        random.shuffle(self.children)


    def selectChild(self, C: float) -> "Node":
        """Uses UCB1 to pick child node"""
        # if node doesn't have children, return self
        if len(self.children) == 0:
            return self

        UCB1values = np.zeros(len(self.children))
        for i, child in enumerate(self.children):
            # returns child if it hasn't been visited before
            if child.visits == 0:
                return child

            # calculates UCB1
            v = child.value
            mi = child.visits
            mp = child.parent.visits
            UCB1values[i] = v/mi + C * np.sqrt(np.log(mp)/mi)

        # return child that maximizes UCB1
        maxIndex = np.argmax(UCB1values)
        return self.children[maxIndex]


    def backpropagate(self, gameState:GameState, result: tuple) -> None:
        """Updates value and visits according to result"""
        instance = self
        while instance != None:
            instance.visits += 1
            instance.value += result[0] if gameState.isOnRedTeam(instance.player) else result[1]
            instance = instance.parent


    def chooseBestChild(self) -> "Node":
        """Chooses most promising move from the list of children"""
        # if node doesn't have children, make no move
        if len(self.children) == 0:
            return self

        # finds child with most visits and returns it
        visits = [child.visits for child in self.children]
        maxVisits = max(visits)
        maxIndex = visits.index(maxVisits)

        chosenChild = self.children[maxIndex]
        return chosenChild


    def nextPlayer(self, players:np.ndarray) -> int:
        player_index = np.where(players == self.player)
        return players[(player_index[0][0]+1) % players.size]
    