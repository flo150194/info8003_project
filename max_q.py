from domain.game import *

DISCOUNT = 0.999
ALPHA = 0.3
PRIMITIVE_TASKS = ["north", "south", "east", "west", "chop", "harvest", "deposit"]
NON_PRIMITIVE_TASKS = ["root", "get_wood", "get_gold", "unload", "navigate"]
CHILDREN_TASKS = {"root": ["get_wood", "get_gold", "unload"],
                  "get_wood": ["chop", "navigate"],
                  "get_gold": ["harvest", "navigate"],
                  "unload": ["deposit", "navigate"],
                  "navigate": ["north", "south", "east", "west"]}
TARGETS = {"get_wood": FORESTS, "get_gold": GOLD_MINES, "unload": [CHEST]}

INIT_CMP = 0

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
    return manhattan_distance([row, col], list(destination)) == 1

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


def get_max_value_function(state, action, v_fct, c_fct, target=None):
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

    # Apply recursive definition
    else:
        values = []
        for sub_task in CHILDREN_TASKS[action]:
            max_v = 0
            if sub_task is "navigate":
                for target in TARGETS[action]:
                    key = flatten((action, state, sub_task, target))
                    if key not in c_fct:
                        c_fct[key] = INIT_CMP
                    max_v += c_fct[key]
                    max_v += get_max_value_function(state, sub_task, v_fct,
                                                    c_fct, target)
                    values.append(max_v)
            else:
                key = flatten((action, state, sub_task, target))
                if key not in c_fct:
                    c_fct[key] = INIT_CMP
                max_v += c_fct[key]
                max_v += get_max_value_function(state, sub_task, v_fct, c_fct,
                                                target)
                values.append(max_v)

        max_v = max(values)

    return max_v


def max_q(state, task, game, value_fct, completion_fct, eps, destination=None):
    """
    Generates an episode following max_q algorithm and an epsilon-greedy
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

    total_reward, t_elapsed = 0, 0
    t_init = game.get_time()
    k = 0

    while t_elapsed < 1000 and not check_termination(game, task, destination):
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
            if task in ["unload", "get_wood", "get_gold"]:
                best_q, best_action, best_target = -np.inf, None, None
                for sub_action in sub_tasks:
                    if sub_action == "navigate":
                        for sub_target in TARGETS[task]:
                            q = get_max_value_function(state, sub_action, value_fct,
                                                       completion_fct, sub_target)
                            if q > best_q:
                                best_q, best_action, best_target = q, sub_action, sub_target
                    else:
                        q = get_max_value_function(state, sub_action, value_fct,
                                                   completion_fct, destination)
                        if q > best_q:
                            best_q, best_action = q, sub_action
                action, target = best_action, best_target

            # When navigate is not a sub-task
            else:
                q = [get_max_value_function(state, action, value_fct,
                                            completion_fct, destination)
                     for action in sub_tasks]
                action = sub_tasks[np.argmax(q)]

        # Action execution
        if action in PRIMITIVE_TASKS:
            game = primitive_action(game, action)
            reward = game.get_reward()
        elif action == "navigate":
            reward = max_q(state, action, game, value_fct, completion_fct, eps,
                           target)
        else:
            reward = max_q(state, action, game, value_fct, completion_fct, eps)

        # Observe action result
        next_state = game.get_state()
        total_reward += reward
        t_elapsed = game.get_time()-t_init

        # Update value or completion function
        # Primitive action case
        if action in PRIMITIVE_TASKS:
            key = flatten((state, action))
            if key not in value_fct:
                value_fct[key] = 0
            value_fct[key] = (1-ALPHA)*value_fct[key] + ALPHA*reward

        # Navigate task case
        elif task == "navigate":
            key = flatten((task, state, action, destination))
            values = []
            # Navigate cannot be a sub-task
            for sub_task in sub_tasks[task]:
                max_v = get_max_value_function(next_state, sub_task, value_fct,
                                               completion_fct)
                sub_key = flatten((task, state, sub_task, destination))
                if sub_key not in completion_fct:
                    completion_fct[sub_key] = INIT_CMP
                cmp = completion_fct[sub_key]
                values.append(max_v + cmp)
            if key not in completion_fct:
                completion_fct[key] = INIT_CMP
            completion_fct[key] = (1 - ALPHA) * completion_fct[key] + \
                                  ALPHA * DISCOUNT**k * max(values)

        # Other task case
        else:
            key = flatten((task, state, action, target))
            values = []
            # Navigate can be a sub-task
            for sub_task in CHILDREN_TASKS[task]:
                if sub_task == "navigate":
                    for sub_target in TARGETS[task]:
                        max_v = get_max_value_function(next_state, sub_task,
                                                       value_fct,
                                                       completion_fct, sub_target)
                        sub_key = flatten((task, state, sub_task, sub_target))
                        if sub_key not in completion_fct:
                            completion_fct[sub_key] = INIT_CMP
                        cmp = completion_fct[sub_key]
                        values.append(max_v + cmp)
                else:
                    max_v = get_max_value_function(next_state, sub_task,
                                                   value_fct, completion_fct)
                    sub_key = flatten((task, state, sub_task, destination))
                    if sub_key not in completion_fct:
                        completion_fct[sub_key] = INIT_CMP
                    cmp = completion_fct[sub_key]
                    values.append(max_v + cmp)

            if key not in completion_fct:
                completion_fct[key] = INIT_CMP
            completion_fct[key] = (1-ALPHA)*completion_fct[key] + \
                                  ALPHA* DISCOUNT**k *max(values)


        k += 1
        state = next_state

    return total_reward





