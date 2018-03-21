from deposit import Deposit

class Forest(Deposit):

    def __init__(self, position):
        super(Forest, self).__init__(position)

    def get_capacity(self):
        return super(Forest, self).get_capacity()

    def exploit(self):
        return super(Forest, self).exploit()

    def is_next(self, position):
        return super(Forest, self).is_next(position)

    def distance(self, position):
        return super(Forest, self).distance(position)