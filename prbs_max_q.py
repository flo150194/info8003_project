from max_q import *
from domain2.game import *
from copy import deepcopy

MAX_TRAINING_EPISODE = 100
MAX_EVAL_EPISODE = 1000

def evaluate(state, action, v_fct, c_fct, game, target=None):
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
            max_v = 0
            if sub_task is "navigate":
                for sub_target in TARGETS[action]:
                    key = flatten((action, state, sub_task, sub_target))
                    if key not in c_fct:
                        c_fct[key] = INIT_CMP
                    max_v += c_fct[key]
                    val, _, _ = evaluate(state, sub_task, v_fct, c_fct, game, sub_target)
                    max_v += val
                    max_v += pbrs(state, action, sub_task, game, sub_target)
                    values.append(max_v)
                    actions.append(sub_task)
                    targets.append(sub_target)
            else:
                key = flatten((action, state, sub_task, target))
                if key not in c_fct:
                    c_fct[key] = INIT_CMP
                max_v += c_fct[key]
                val, _, _ = evaluate(state, sub_task, v_fct, c_fct, game,
                                     target)
                max_v += val
                max_v += pbrs(state, action, sub_task, game, target)
                values.append(max_v)
                actions.append(sub_task)
                if action == 'navigate':
                    targets.append(target)
                else:
                    targets.append(None)

        idx = np.argmax(values)

        return values[idx], actions[idx], targets[idx]


def pbrs(state, task, sub_task, game, destination=None):

    phi = 0
    if sub_task == 'navigate':

        if task == 'get_gold':
            mines = game.get_mines()
            for mine in mines:
                if mine.distance(destination) == 0:
                    if mine.get_capacity > 0:
                        phi = 5
                    else:
                        phi = -5
                    break

        elif task == 'get_wood':
            forest = game.get_forest()
            for forest in forest:
                if forest.distance(destination) == 0:
                    if forest.get_capacity > 0:
                        phi = 5
                    else:
                        phi = -5
                    break

    elif sub_task in CHILDREN_TASKS['navigate']:
        cur_pos = tuple([state[0], state[1]])
        cur_dist = manhattan_distance(cur_pos, destination)

        copy = deepcopy(game)
        copy = primitive_action(copy, sub_task)
        next_state = copy.get_state()
        next_pos = tuple([next_state[0], next_state[1]])
        next_dist = manhattan_distance(next_pos, destination)

        if next_dist < cur_dist:
            phi = 2
        else:
            phi = -2

    return phi




def pbrs_maxq(state, task, game, value_fct, completion_fct, eps, destination=None):
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
    :return: (float) the total reward of performing the task in the current state
                     according to max_q algorithm
    """

    seq = []

    # Base case
    if task in PRIMITIVE_TASKS:
        game = primitive_action(game, task)
        next_state, reward = game.get_state(), game.get_reward()
        key = flatten((next_state, task))
        if key not in value_fct:
            value_fct[key] = 0
        value_fct[key] = (1 - ALPHA) * value_fct[key] + ALPHA * reward
        seq.insert(0, state)

    else:
        while not check_termination(game, task, destination) and game.get_time() < MAX_TRAINING_EPISODE:

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
                action, target = select_best_action(state, task, destination, value_fct, completion_fct)

            if task == "navigate":
                child_seq = pbrs_maxq(state, action, game, value_fct,
                                      completion_fct, eps, destination)
            elif action == 'navigate':
                child_seq = pbrs_maxq(state, action, game, value_fct,
                                      completion_fct, eps, target)
            else:
                child_seq = pbrs_maxq(state, action, game, value_fct,
                                      completion_fct, eps)

            # Observe results of sub-task execution
            next_state = game.get_state()
            max_v, best_action, best_target = evaluate(next_state, task, value_fct, completion_fct, game, destination)

            # Iterative updates of tables
            n = 1
            for sub_state in child_seq:
                if action == 'navigate':
                    key = flatten((task, sub_state, action, target))
                    cur_pbrs = pbrs(sub_state, task, action, game, target)
                else:
                    key = flatten((task, sub_state, action, destination))
                    cur_pbrs = pbrs(sub_state, task, action, game, destination)
                new_pbrs = pbrs(next_state, task, best_action, game, best_target)
                new_key = flatten([task, next_state, best_action, best_target])
                if new_key not in completion_fct:
                    completion_fct[new_key] = INIT_CMP
                new_cmp = completion_fct[new_key]
                if key not in completion_fct:
                    completion_fct[key] = INIT_CMP
                completion_fct[key] = (1-ALPHA)*completion_fct[key] + \
                                      ALPHA* (DISCOUNT**n *(max_v+new_cmp+new_pbrs) - cur_pbrs)
                n += 1

            seq = child_seq + seq
            state = next_state

    return seq


def eval_pbrs_maxq(state, task, game, value_fct, completion_fct, eps, destination=None):
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
    :return: (float) the total reward of performing the task in the current state
                     according to max_q algorithm
    """

    seq = []

    # Base case
    if task in PRIMITIVE_TASKS:
        game = primitive_action(game, task)
        next_state, reward = game.get_state(), game.get_reward()
        seq.append(DISCOUNT**game.get_time() * reward)

    else:
        while not check_termination(game, task, destination) and game.get_time() < MAX_EVAL_EPISODE:

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
                # Case when navigate is a subtask
                action, target = select_best_action(state, task, destination, value_fct, completion_fct)
            if task == "navigate":
                child_seq = eval_pbrs_maxq(state, action, game, value_fct,
                                      completion_fct, eps, destination)
            elif action == 'navigate':
                child_seq = eval_pbrs_maxq(state, action, game, value_fct,
                                      completion_fct, eps, target)
            else:
                child_seq = eval_pbrs_maxq(state, action, game, value_fct,
                                      completion_fct, eps)

            # Observe results of sub-task execution
            next_state = game.get_state()

            seq = child_seq + seq
            state = next_state

    return seq

def select_best_action(state, task, destination, value_fct, completion_fct, pr=False):
    """

    :param state:
    :param task:
    :param destination:
    :param value_fct:
    :param completion_fct:
    :return:
    """
    sub_tasks = CHILDREN_TASKS[task]
    target = None

    # Case when navigate is a subtask
    if task in ["unload", "get_wood", "get_gold"]:
        best_q, best_action, best_target = -np.inf, None, None
        for sub_action in sub_tasks:
            if sub_action == "navigate":
                for sub_target in TARGETS[task]:
                    q = get_max_value_function(state, sub_action,
                                               value_fct,
                                               completion_fct,
                                               sub_target)
                    if q > best_q:
                        best_q, best_action, best_target = q, sub_action, sub_target
            else:
                q = get_max_value_function(state, sub_action,
                                           value_fct,
                                           completion_fct,
                                           destination)
                if q > best_q:
                    best_q, best_action = q, sub_action
        action, target = best_action, best_target

    # When navigate is not a sub-task
    else:
        q = [get_max_value_function(state, action, value_fct,
                                    completion_fct, destination)
             for action in sub_tasks]

        action = sub_tasks[np.argmax(q)]
        if pr and task == 'navigate':
            print sub_tasks
            print q
            print action

    return action, target





