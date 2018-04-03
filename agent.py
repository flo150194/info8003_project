from domain2.game import *
from max_q import max_q
from prbs_max_q import *


def constant_policy():
    game = Game(8, False)
    reward = eval_pbrs_maxq(game.get_state(), "root", game, None, None, 0)
    print(game.get_state())
    print(np.sum(reward))


def test_prbs_maxq():

    value_function, completion_function = {}, {}
    epsilon = np.linspace(0.5, 0.1, num=20, endpoint=True)
    for eps in epsilon:
        print(eps)
        game = Game(8, False)
        seq = pbrs_maxq(game.get_state(), "root", game, value_function,
                        completion_function, eps)

        game = Game(8, False)
        reward = eval_pbrs_maxq(game.get_state(), "root", game, value_function,
                                completion_function, eps)
        print(game.get_state())
        print(np.sum(reward))

if __name__ == '__main__':

    constant_policy()








