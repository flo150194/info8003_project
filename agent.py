from domain2.game import *
from max_q import max_q
from prbs_max_q import *

if __name__ == '__main__':

    value_function, completion_function = {}, {}

    for i in range(3):
        print i
        game = Game(8)
        seq = pbrs_maxq(game.get_state(), "root", game, value_function,
                        completion_function, 0.8)

    game = Game(8)
    action, target = "root", None
    for i in range(5):
        n_action, n_target = select_best_action(game.get_state(), action, target,
                                                value_function,
                                                completion_function, pr=True)
        if n_action in PRIMITIVE_TASKS:
            game = primitive_action(game, n_action)
            state = game.get_state()
            print (state[0], state[1], n_action, n_target)
        else:
            action, target = n_action, n_target
            state = game.get_state()
            print (state[0], state[1], action, target)


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







