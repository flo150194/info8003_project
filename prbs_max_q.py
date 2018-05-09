from domain2.game import *
from copy import deepcopy

TIME_HORIZON = 300
PRIMITIVE_TASKS = ["north", "south", "east", "west", "chop", "harvest",
                   "deposit"]
NON_PRIMITIVE_TASKS = ["root", "get_wood", "get_gold", "unload", "navigate"]
CHILDREN_TASKS = {"root": ["get_wood", "get_gold", "unload"],
                  "get_wood": ["chop", "navigate"],
                  "get_gold": ["harvest", "navigate"],
                  "unload": ["deposit", "navigate"],
                  "navigate": ["north", "south", "east", "west"]}
TARGETS = {"get_wood": FORESTS, "get_gold": GOLD_MINES, "unload": [CHEST]}

INIT_CMP = 0
MAX_Q_0 = False

from matplotlib import pyplot as plt

def flatten(array):
    res = []
    for el in array:
        if isinstance(el, (list, tuple)):
            res.extend(flatten(el))
            continue
        res.append(el)
    return tuple(res)


def manhattan_distance(position, destination):
    return reduce(operator.add, map(abs, map(operator.sub,
                                             destination,
                                             position)))
def is_root_completed(game):
    return game.is_win()

def is_get_wood_completed(game):
    _, _, res, _, _, _ = game.get_state()
    return res == WOOD

def is_get_gold_completed(game):
    _, _, res, _, _, _ = game.get_state()
    return res == GOLD

def is_unload_completed(game):
    _, _, res, _, _, _ = game.get_state()
    return res == NOTHING

def is_navigate_completed(game, destination):
    row, col, _, _, _, _ = game.get_state()
    return manhattan_distance([row, col], list(destination)) == 0

def primitive_action(game, action):
    """
    Executes a primitive action.

    :param game: (Game) Instance of the game
    :param action: (string) Action to be perfomed
    :return: (Game) the updated instance of the game
    """
    if action == "north":
        game.move(NORTH)
    elif action == "south":
        game.move(SOUTH)
    elif action == "east":
        game.move(EAST)
    elif action == "west":
        game.move(WEST)
    elif action == "chop":
        game.chop()
    elif action == "harvest":
        game.harvest()
    elif action == "deposit":
        game.deposit()

    return game

def check_termination(game, action, destination=None):
    """
    Check the termination of a given action.

    :param game: (Game) Instance of the game
    :param action: (string) Action for which the termination is checked
    :param destination: (tuple)  In case of 'navigate' action, location to reach
    :return: True if the action is terminated, False otherwise
    """
    if action =="root":
        return is_root_completed(game)
    elif action == "get_wood":
        return is_get_wood_completed(game)
    elif action == "get_gold":
        return is_get_gold_completed(game)
    elif action == "unload":
        return is_unload_completed(game)
    elif action == "navigate":
        return is_navigate_completed(game, destination)


def evaluate(state, action, v_fct, c_fct, game, target=None, pr=False):
    """
    Recursively compute the value function of a tuple (state, action, target)
    following the max_q algorithm.

    :param state: (tuple) State for which the value function is computed
    :param action: (string) Action for which the value function is computed
    :param v_fct: (dict) Value functions dictionary
    :param c_fct: (dict) Completion functions dictionary
    :param target: (tuple)  In case of 'navigate' action, location to reach
    :return: (float) the required value function
    """

    # Base case
    if action in PRIMITIVE_TASKS:
        key = flatten((state, action))
        if key not in v_fct:
            v_fct[key] = 0
        max_v = v_fct[key]
        return max_v, None, None

    # Apply recursive definition
    else:
        values = []
        actions = []
        targets = []

        for sub_task in CHILDREN_TASKS[action]:
            if sub_task is "navigate":
                for sub_target in TARGETS[action]:
                    max_v = 0
                    key = flatten((action, state, sub_task, sub_target))
                    if key not in c_fct:
                        c_fct[key] = INIT_CMP
                    max_v += c_fct[key]
                    val, _, _ = evaluate(state, sub_task, v_fct, c_fct, game,
                                         sub_target)
                    max_v += val
                    max_v += pbrs(state, action, sub_task, game, sub_target,
                                  max_q=MAX_Q_0)
                    values.append(max_v)
                    actions.append(sub_task)
                    targets.append(sub_target)
            else:
                max_v = 0
                key = flatten((action, state, sub_task, target))
                if key not in c_fct:
                    c_fct[key] = INIT_CMP
                max_v += c_fct[key]
                val, _, _ = evaluate(state, sub_task, v_fct, c_fct, game,
                                     target)
                max_v += val
                max_v += pbrs(state, action, sub_task, game, target,
                              max_q=MAX_Q_0)
                values.append(max_v)
                actions.append(sub_task)
                if action == 'navigate':
                    targets.append(target)
                else:
                    targets.append(None)

        idx = np.argmax(values)

        if pr:
            print idx
            print actions
            print targets
            print values

        return values[idx], actions[idx], targets[idx]


def pbrs(state, task, sub_task, game, destination=None, max_q=False):
    if max_q:
        return 0

    phi = 0
    cur_pos = tuple([state[0], state[1]])

    if sub_task == 'navigate':

        if task == 'get_gold':
            mines = game.get_mines()
            for mine in mines:
                if mine.distance(destination) == 0:
                    if mine.get_capacity > 0 \
                            and cur_pos not in GOLD_MINES \
                            and state[2] == NOTHING and state[3][1] > 0:
                        phi = 5
                    else:
                        phi = -5
                    break

        elif task == 'get_wood':
            forest = game.get_forest()
            for forest in forest:
                if forest.distance(destination) == 0:
                    if forest.get_capacity > 0 \
                            and cur_pos not in FORESTS \
                            and state[2] == NOTHING and state[3][0] > 0:
                        phi = 5
                    else:
                        phi = -5
                    break

    elif sub_task in CHILDREN_TASKS['navigate'] and not game.obs:
        cur_dist = manhattan_distance(cur_pos, destination)

        if sub_task == "north":
            next_pos = tuple([cur_pos[0]-1, cur_pos[1]])
        elif sub_task == "south":
            next_pos = tuple([cur_pos[0] + 1, cur_pos[1]])
        elif sub_task == "east":
            next_pos = tuple([cur_pos[0], cur_pos[1]+1])
        else:
            next_pos = tuple([cur_pos[0], cur_pos[1]-1])

        next_dist = manhattan_distance(next_pos, destination)

        if next_dist < cur_dist:
            phi = 2
        else:
            phi = -2

    """
    elif sub_task == 'unload':
        if state[2] == NOTHING:
            phi = -5
        else:
            phi = 5

    elif sub_task == 'get_wood':
        if state[2] == NOTHING and state[4] > 0:
            phi = 5
        else:
            phi = -5

    elif sub_task == 'get_gold':
        if state[2] == NOTHING and state[5] > 0:
            phi = 5
        else:
            phi = -5

    return phi

    elif sub_task == 'chop' and cur_pos in FORESTS:
        for forest in game.get_forest():
            if forest.distance(cur_pos) == 0:
                if state[2] == NOTHING and forest.get_capacity() > 0:
                    phi = 7
                else:
                    phi = -5
                break


    elif sub_task == 'harvest' and cur_pos in GOLD_MINES:
        for mine in game.get_mines():
            if mine.distance(cur_pos) == 0:
                if state[2] == NOTHING and mine.get_capacity() > 0:
                    phi = 7
                else:
                    phi = -5
                break

    elif sub_task == 'unload':
        if state[2] == NOTHING:
            phi = -5
        else:
            phi = 5
    """

    return phi


def pbrs_maxq(state, task, game, value_fct, completion_fct, eps,
              destination=None, gamma=0.99, alpha=0.01):
    """
    Generates an episode following prbs-max_q algorithm and an epsilon-greedy
    policy.

    :param state: (tuple) Current state of the game
    :param task: (string) Current task to be performed
    :param game: (Game) Instance of the game
    :param value_fct: (dict) Value functions dictionary
    :param completion_fct: (dict) Completion function dictionary
    :param eps: (float) epsilon value for epsilon-greedy policy
    :param destination: (tuple) In case of 'navigate' task, location to reach
    :param gamma: (float) discount factor
    :param alpha: (float) learning rate
    :return: (float) the total reward of performing the task in the current state
                     according to max_q algorithm
    """

    seq = []
    rewards = []

    # Base case
    if task in PRIMITIVE_TASKS:
        game = primitive_action(game, task)
        next_state, reward = game.get_state(), game.get_reward()
        key = flatten((next_state, task))
        if key not in value_fct:
            value_fct[key] = 0
        value_fct[key] = (1 - alpha) * value_fct[key] + alpha * reward
        seq.insert(0, state)
        rewards.append(gamma ** game.get_time() * reward)

    else:
        while not check_termination(game, task, destination) \
                and game.get_time() < TIME_HORIZON:

            sub_tasks = CHILDREN_TASKS[task]
            rd = np.random.uniform()
            target = None

            # Action choice
            if rd < eps:
                action = np.random.choice(sub_tasks)
                if action == "navigate":
                    idx = np.random.randint(0, len(TARGETS[task]))
                    target = TARGETS[task][idx]
            else:
                action, target = select_best_action(state, task, destination,
                                                    value_fct, completion_fct,
                                                    game)

            if task == "navigate":
                child_seq, reward = pbrs_maxq(state, action, game, value_fct,
                                              completion_fct, eps, destination,
                                              gamma=gamma, alpha=alpha)
            elif action == 'navigate':
                child_seq, reward = pbrs_maxq(state, action, game, value_fct,
                                              completion_fct, eps, target,
                                              gamma=gamma, alpha=alpha)
            else:
                child_seq, reward = pbrs_maxq(state, action, game , value_fct,
                                              completion_fct, eps,
                                              gamma=gamma, alpha=alpha)

            # Observe results of sub-task execution
            next_state = game.get_state()
            max_v, best_action, best_target = evaluate(next_state, task,
                                                       value_fct,
                                                       completion_fct, game,
                                                       destination)

            # Iterative updates of tables
            n = 1
            for sub_state in child_seq:

                if action == 'navigate':
                    key = flatten((task, sub_state, action, target))
                    cur_pbrs = pbrs(sub_state, task, action, game, target,
                                    max_q=MAX_Q_0)

                else:
                    key = flatten((task, sub_state, action, destination))
                    cur_pbrs = pbrs(sub_state, task, action, game, destination,
                                    max_q=MAX_Q_0)

                new_pbrs = pbrs(next_state, task, best_action, game,
                                best_target, max_q=MAX_Q_0)
                new_key = flatten([task, next_state, best_action, best_target])
                if new_key not in completion_fct:
                    completion_fct[new_key] = INIT_CMP
                new_cmp = completion_fct[new_key]
                if key not in completion_fct:
                    completion_fct[key] = INIT_CMP
                completion_fct[key] = (1 - alpha) * completion_fct[key] + \
                                      alpha * (gamma ** n * (
                                      max_v + new_cmp + new_pbrs) - cur_pbrs)
                n += 1

            seq = child_seq + seq
            rewards = reward + rewards
            state = next_state

    return seq, rewards


def eval_pbrs_maxq(state, task, game, value_fct, completion_fct, eps,
                   destination=None, gamma=0.99):
    """
    Generates an episode following prbs max_q algorithm and an epsilon-greedy
    policy.

    :param state: (tuple) Current state of the game
    :param task: (string) Current task to be performed
    :param game: (Game) Instance of the game
    :param value_fct: (dict) Value functions dictionary
    :param completion_fct: (dict) Completion function dictionary
    :param eps: (float) epsilon value for epsilon-greedy policy
    :param destination: (tuple) In case of 'navigate' task, location to reach
    :param gamma: (float) discount factor
    :return: (float) the total reward of performing the task in the current state
                     according to max_q algorithm
    """

    seq = []

    # Base case
    if task in PRIMITIVE_TASKS:
        game = primitive_action(game, task)
        next_state, reward = game.get_state(), game.get_reward()
        seq.append(gamma ** game.get_time() * reward)

    else:
        while not check_termination(game, task, destination) \
                and game.get_time() < TIME_HORIZON:

            sub_tasks = CHILDREN_TASKS[task]
            rd = np.random.uniform()
            target = None

            # Action choice
            if rd < eps:
                action = np.random.choice(sub_tasks)
                if action == "navigate":
                    idx = np.random.randint(0, len(TARGETS[task]))
                    target = TARGETS[task][idx]
            else:
                action, target = select_best_action(state, task, destination,
                                                    value_fct, completion_fct,
                                                    game)
            if task == "navigate":
                child_seq = eval_pbrs_maxq(state, action, game, value_fct,
                                           completion_fct, eps, destination,
                                           gamma=gamma)
            elif action == 'navigate':
                child_seq = eval_pbrs_maxq(state, action, game, value_fct,
                                           completion_fct, eps, target,
                                           gamma=gamma)
            else:
                child_seq = eval_pbrs_maxq(state, action, game, value_fct,
                                           completion_fct, eps,
                                           gamma=gamma)

            # Observe results of sub-task execution
            next_state = game.get_state()

            seq = child_seq + seq
            state = next_state

    return seq


def select_best_action(state, task, destination, value_fct, completion_fct,
                       game, pr=False):
    """

    :param state:
    :param task:
    :param destination:
    :param value_fct:
    :param completion_fct:
    :return:
    """
    # Constant policy
    if value_fct is None:
        if task == "root":
            action, target = "get_wood", None
        elif task == "get_wood" and completion_fct == "chop":
            action, target = "chop", None
        elif task == "get_wood" and completion_fct == "north":
            action, target = "navigate", (0,0)
        else:
            action, target = "north", (0,0)

    # PBRS-Max-q policy
    else:
        val, action, target = evaluate(state, task, value_fct, completion_fct,
                                       game, destination, pr=pr)


    # Case when navigate is a subtask
    """
    if task in ["unload", "get_wood", "get_gold"]:
        best_q, best_action, best_target = -np.inf, None, None
        for sub_action in sub_tasks:
            if sub_action == "navigate":
                for sub_target in TARGETS[task]:
                    q, _, _ = evaluate(state, sub_action, value_fct, completion_fct, game, sub_target)
                    if q > best_q:
                        best_q, best_action, best_target = q, sub_action, sub_target
            else:
                q, _, _ = evaluate(state, sub_action, value_fct, completion_fct, game, destination)
                if q > best_q:
                    best_q, best_action = q, sub_action
        action, target = best_action, best_target

    # When navigate is not a sub-task
    else:
        max_values = []
        for action in sub_tasks:
            q, _, _ = evaluate(state, action, value_fct, completion_fct, game, destination)
            max_values.append(q)

        action = sub_tasks[np.argmax(max_values)]
    """

    return action, target

if __name__ == '__main__':
    a = 1

    x = np.linspace(1, 10, 10, endpoint=True)

    plt.figure()
    plt.plot(x, x ** 2, label='$x^2$')
    plt.plot(x, x, label='$x$')
    plt.plot(x, x ** 3, label='$x^3$')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)

    plt.show()




