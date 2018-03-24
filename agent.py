from domain.game import *
from max_q import max_q

if __name__ == '__main__':

    value_function, completion_function = {}, {}

    eps = 1

    for i in range(50):
        game = Game(8)
        r = max_q(game.get_state(), "root", game, value_function, completion_function, eps)
        eps = eps**0.99
        print r
        print game.get_state()

    good, val = [], []
    for key in value_function.keys():
        if value_function[key] != 0:
            good.append(key)
            val.append(value_function[key])
    print len(good)

    good, val = [], []
    for key in completion_function.keys():
        if completion_function[key] != 0:
            good.append(key)
            val.append(completion_function[key])
    print len(good)

    print completion_function


