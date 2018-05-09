from player import Player
from chest import Chest
from obstacle import Obstacle
from freeTile import FreeTile
from forest import Forest
from goldMine import GoldMine
import operator
import numpy as np

# Coordinates corresponding to the domain displayed during the presentation
PLAYER = (5,5)
CHEST = (4,5)
GOLD_MINES = [(0,1), (6,1)]
FORESTS = [(3,7), (7,4)]
OBSTACLES = [(2, i) for i in range(4)] + [(i, 2) for i in range(5,8)] + \
            [(6, i) for i in range(4,7)]

# Resources constants
NOTHING, GOLD, WOOD = 0, 1, 2
RESOURCES = [GOLD, WOOD]
NB_RESOURCES = 2
NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
ACTIONS = {NORTH: [-1, 0], SOUTH: [1, 0], EAST: [0, 1], WEST: [0, -1]}
TARGET_QUOTA = 3
PRINT_RES = {NOTHING: "nothing", WOOD: "wood", GOLD: "gold"}

class Game(object):

    def __init__(self, n, stocha, obs):

        self.gold_mines = []
        self.forests = []
        self.obstacles = []
        self.board = []
        self.quotas = [False for k in range(NB_RESOURCES)]
        self.n= n
        self.time = 0
        self.reward = 0
        self.stocha = stocha

        # Board instantiation
        for i in range(n):
            for j in range(n):
                if (i, j) in OBSTACLES and obs:
                    obstacle = Obstacle([i, j])
                    self.board.append(obstacle)
                    self.obstacles.append(obstacle)
                elif (i,j) in GOLD_MINES:
                    gold_mine = GoldMine([i,j])
                    self.board.append(gold_mine)
                    self.gold_mines.append(gold_mine)
                elif (i,j) in FORESTS:
                    forest = Forest([i,j])
                    self.board.append(forest)
                    self.forests.append(forest)
                elif (i,j) == PLAYER:
                    self.player = Player([i,j], NOTHING)
                    self.board.append(FreeTile([i,j]))
                elif (i,j) == CHEST:
                    self.chest = Chest([i,j])
                    self.chest_next = True
                    self.board.append(self.chest)
                else:
                    self.board.append(FreeTile([i,j]))

        self.gold_mines_next = [False for k in self.gold_mines]
        self.forests_next = [False for k in self.forests]

    def get_reward(self):
        return self.reward

    def get_time(self):
        return self.time

    def get_n(self):
        return self.n

    def get_chest_position(self):
        return tuple(self.chest.get_position())

    def get_forest_position(self):
        return [tuple(forest.get_position()) for forest in self.forests]

    def get_mines_position(self):
        return [tuple(mine.get_position()) for mine in self.gold_mines]

    def get_state(self):
        pos = self.player.get_position()
        row, col = pos[0], pos[1]
        res = self.player.get_resource()
        wood_quota = TARGET_QUOTA-self.chest.get_wood()
        gold_quota = TARGET_QUOTA-self.chest.get_gold()
        wood_left = tuple([forest.get_capacity() for forest in self.forests])
        good_left = tuple([mine.get_capacity() for mine in self.gold_mines])

        return row, col, res, tuple([wood_quota, gold_quota]), \
               wood_left, good_left

    def move(self, direction):

        self.time += 1
        self.reward = -1

        # Stochastic dynamics
        if self.stocha:
            noise = np.random.uniform()
            if noise > 0.7 and noise < 0.8:
                direction = np.mod(direction+1, 4)
            elif noise > 0.8 and noise < 0.9:
                direction = np.mod(direction+2, 4)
            elif noise > 0.9 and noise < 1:
                direction = np.mod(direction+3, 4)


        action = ACTIONS[direction]
        destination = map(operator.add, self.player.get_position(), action)
        row, col = destination[0], destination[1]
        objects = self.obstacles + self.forests + self.gold_mines + \
                  [self.chest]

        # Check if the move is legal
        if row >= 0 and row < self.n and col >= 0 and col < self.n:
            for obj in objects:
                if obj.distance(destination) == 0:
                    self.reward -= 1
                    return False

            self.player.set_position(destination)

            # Update adjacency booleans
            for k in range(len(self.gold_mines)):
                mine = self.gold_mines[k]
                self.gold_mines_next[k] = mine.is_next(destination)
            for k in range(len(self.forests)):
                forest = self.forests[k]
                self.forests_next[k] = forest.is_next(destination)
            self.chest_next = self.chest.is_next(destination)

            return True

        else:
            self.reward -= 1
            return False

    def chop(self):

        self.time += 1
        self.reward = -1

        if self.player.get_resource() is not NOTHING:
            self.reward -= 1
            return False

        for k in range(len(self.forests)):
            if self.forests_next[k] and self.forests[k].exploit():
                self.player.set_resource(WOOD)
                return True

        self.reward -= 1
        return False

    def harvest(self):

        self.time += 1
        self.reward = -1

        if self.player.get_resource() is not NOTHING:
            return False

        for k in range(len(self.gold_mines)):
            if self.gold_mines_next[k] and self.gold_mines[k].exploit():
                self.player.set_resource(GOLD)
                return True

        self.reward -= 1
        return False

    def deposit(self):

        self.time +=1
        self.reward = -1

        player_res = self.player.get_resource()
        if player_res is NOTHING or not self.chest_next:
            self.reward -= 1
            return False

        if player_res == WOOD:
            self.chest.store_wood()
            if self.chest.get_wood() == TARGET_QUOTA:
                self.reward += 50
                self.quotas[WOOD-1] = True
        else:
            self.chest.store_gold()
            if self.chest.get_gold() == TARGET_QUOTA:
                self.reward += 50
                self.quotas[GOLD-1] = True

        self.player.set_resource(NOTHING)
        return True

    def is_win(self):

        for quota in self.quotas:
            if not quota:
                return False

        return True

    def print_board(self):

        hor_offset = ''
        a = hor_offset + (' ___' * self.n)

        c = []
        for i in range(self.n):
            b = []
            for j in range(self.n):
                b.append('|')
                if (i,j) == tuple(self.player.get_position()):
                    b.append("P")
                else:
                    tile = self.board[i*self.n+j]
                    if isinstance(tile, FreeTile):
                        b.append(" ")
                    elif isinstance(tile, Obstacle):
                        b.append("O")
                    elif isinstance(tile, GoldMine):
                        b.append("G")
                    elif isinstance(tile, Forest):
                        b.append("W")
                    else:
                        b.append("C")
            b.append('|')
            b = ' '.join(b)
            c.extend([a, b])
        c.append(a)
        print('\n'.join(tuple(c)))

        print("Player is carrying "+str(PRINT_RES[self.player.get_resource()]))+"."
        print("Chest contains "+str(self.chest.get_gold())+" gold(s) and "+
              str(self.chest.get_wood())+" wood(s).")
