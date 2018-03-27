from domain2.game import *
from max_q import max_q, pbrs_maxq, eval_pbrs_maxq

if __name__ == '__main__':

    value_function, completion_function = {}, {}
    game = Game(8)
    seq = pbrs_maxq(game.get_state(), "root", game, value_function,
                    completion_function, 1)

    """

    epss = np.linspace(1, 0.2, 10, endpoint=True)

    for eps in epss:
        print eps
        game = Game(8)
        seq = pbrs_maxq(game.get_state(), "root", game, value_function,
                  completion_function, eps)
        eps = eps*0.8
        print game.get_state()

    game = Game(8)
    rewards = eval_pbrs_maxq(game.get_state(), "root", game, value_function,
                  completion_function, 0.1)
    print rewards
    print np.sum(rewards)
    """







