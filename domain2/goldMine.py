from deposit import Deposit

class GoldMine(Deposit):

    def __init__(self, position):
        super(GoldMine, self).__init__(position)

    def get_capacity(self):
        return super(GoldMine, self).get_capacity()

    def exploit(self):
        return super(GoldMine, self).exploit()

    def is_next(self, position):
        return super(GoldMine, self).is_next(position)

    def distance(self, position):
        return super(GoldMine, self).distance(position)

    def is_reached(self, position):
        return super(GoldMine, self).is_reached(position)
