#! /usr/bin/env python
from gameObject import GameObject

class Chest(GameObject):

    def __init__(self, position):

        super(Chest, self).__init__(position)
        self.gold_amount = 0
        self.wood_amount = 0

    def store_gold(self):
        self.gold_amount += 1

    def store_wood(self):
        self.wood_amount += 1

    def get_gold(self):
        return self.gold_amount

    def get_wood(self):
        return self.wood_amount

    def is_next(self, position):
        return super(Chest, self).is_next(position)

    def distance(self, position):
        return super(Chest, self).distance(position)

    def is_reached(self, position):
        return self.distance(position) == 0