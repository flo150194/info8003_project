class Player(object):

    def __init__(self, position, resource):
        self.position = position
        self.resource = resource

    def get_position(self):
        return self.position

    def get_resource(self):
        return self.resource

    def set_position(self, position):
        self.position = position

    def set_resource(self, resource):
        self.resource = resource
