from gameObject import GameObject

MAX_CAPACITY = 2

class Deposit(GameObject):

    def __init__(self, position):

        super(Deposit, self).__init__(position)
        self.capacity = MAX_CAPACITY

    def get_capacity(self):
        return self.capacity

    def exploit(self):

        if self.capacity > 0:
            self.capacity -= 1
            return True

        else:
            return False

    def is_next(self, position):
        return super(Deposit, self).is_next(position)


    def distance(self, position):
        return super(Deposit, self).distance(position)

    def is_reached(self, position):
        return self.distance(position) == 0