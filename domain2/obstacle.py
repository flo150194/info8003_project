from gameObject import GameObject

class Obstacle(GameObject):

    def __init__(self, position):
        super(Obstacle, self).__init__(position)

    def distance(self, position):
        return super(Obstacle, self).distance(position)