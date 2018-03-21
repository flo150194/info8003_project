from functools import reduce
import operator

class GameObject(object):

    def __init__(self, position):
        self.position = position


    def get_position(self):
        return self.position


    def is_next(self, position):
        return self.distance(position) == 1


    def distance(self, position):
        return reduce(operator.add, map(abs, map(operator.sub,
                                                 self.position,
                                                 position)))
