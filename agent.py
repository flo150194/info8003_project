from max_q import max_q
from prbs_max_q import *
from cmplot import *
from domain2.game import *
import seaborn as sns
sns.set()
sns.set_color_codes()


def constant_policy(policy):
    """
    Evaluates a constant policy with one episode.

    :param policy: the contanst policy to follow
    :return: the cumulated reward
    """
    game = Game(8, False, False, False)
    reward = eval_pbrs_maxq(game.get_state(), "root", game, None, policy, 0)
    return (np.sum(reward))


def test1_prbs_maxq():
    """
    Learns a policy following pbrs-maxq algorithm
    :return the list of rewards corresponding to policy evaluations
    """

    value_function, completion_function = {}, {}
    epsilon, gamma, alpha = 1, 0.7, 0.01
    plt_rewards = []
    while epsilon > 0.05 and gamma < 1:
        print gamma
        game = Game(8, False, False, False)
        seq, reward = pbrs_maxq(game.get_state(), "root", game, value_function,
                                completion_function, eps=epsilon, gamma=gamma,
                                alpha=alpha)
        print game.get_state()

        rewards = []
        for k in range(50):
            game = Game(8, False, False, False)
            reward = eval_pbrs_maxq(game.get_state(), "root", game,
                                    value_function, completion_function,
                                    eps=epsilon, gamma=gamma)
            rewards.append(np.sum(reward))

        plt_rewards.append(np.mean(rewards))
        epsilon *= 0.98
        alpha *= 0.97
        gamma *= 1.005

    return plt_rewards



def test2_prbs_maxq(stocha, obstacle, custom):
    """
    Learns a policy following pbrs-maxq algorithm
    :return the list of rewards corresponding to policy evaluations
    """

    # Tables definition
    value_function, completion_function = {}, {}

    # Learning parameters
    epsilon = np.append(np.geomspace(1, 0.1, num=100, endpoint=True),
                        np.linspace(0, 0, num=100, endpoint=True))
    gamma = np.geomspace(0.99, 0.99, num=200, endpoint=True)
    alpha = np.geomspace(0.01, 0.00001, num=200, endpoint=True)

    # Learning episodes
    plt_rewards = []
    for k in range(len(epsilon)):
        print alpha[k]
        print epsilon[k]
        game = Game(8, stocha, obstacle, custom)
        seq, reward = pbrs_maxq(game.get_state(), "root", game, value_function,
                        completion_function, eps=epsilon[k], gamma=gamma[k],
                                alpha=alpha[k])

        # Policy evaluation after k episodes
        rewards = []
        for i in range(10):
            game = Game(8, stocha, obstacle, custom)
            reward = eval_pbrs_maxq(game.get_state(), "root", game,
                                    value_function, completion_function,
                                    eps=epsilon[k], gamma=gamma[k])
            rewards.append(np.sum(reward))

        plt_rewards.append(np.mean(rewards))

    return plt_rewards


if __name__ == '__main__':

    r = test2_prbs_maxq(False, False, True)
    chop = constant_policy("chop")
    chop = [chop for i in range(len(r))]
    plot_graph(np.arange(start=1, stop=len(r)+1), [r, chop], x_label='Episodes',
               y_label='Episodic reward', colors=['b', 'r'], grid=True,
               names=['PBRS-MAXQ', '$\pi(x)$ = \'chop\''],
               legend=True)
    #plt.savefig('pbrs_eps_3.eps')

    plt.show()









