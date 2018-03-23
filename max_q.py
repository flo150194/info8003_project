from domain.game import *

DISCOUNT = 0.99
ALPHA = 0.05
PRIMITIVE_TASKS = ["north", "south", "east", "west", "chop", "harvest", "deposit"]
NON_PRIMITIVE_TASKS = ["root", "get_wood", "get_gold", "unload", "navigate"]
CHILDREN_TASKS = {"root": ["get_wood", "get_gold", "unload"],
                  "get_wood": ["chop", "navigate"],
                  "get_gold": ["harvest", "navigate"],
                  "unload": ["deposit", "navigate"],
                  "navigate": ["north", "south", "east", "west"]}

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
    return destination == (row, col)

def primitive_action(game, action):
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


def get_max_value_function(state, action, v_fct, c_fct, parent=None, target=None):

    if action in PRIMITIVE_TASKS:
        max_v = v_fct[(state, action)]
    else:
        max_v = 0
        for sub_task in CHILDREN_TASKS[action]:
            if sub_task in PRIMITIVE_TASKS:
                max_v += v_fct[(state, action)]
            elif sub_task == "navigate":
                max_v += c_fct[(parent, state, action, target)]
            else:
                max_v += c_fct[(parent, state, action)]

    return max_v


def max_q(state, task, game, value_fct, completion_fct, eps, destination=None):

    total_reward = 0

    targets = []
    if task == "unload":
        targets = [game.get_chest_position()]
    elif task == "get_wood":
        targets = [game.get_forest_position()]
    elif task == "get_gold":
        targets = [game.get_mines_position()]

    while not check_termination(game, task, destination):
        sub_tasks = CHILDREN_TASKS[task]
        rd = np.random.uniform()
        target = None

        # Action choice
        if rd < eps:
            action = np.random.choice(sub_tasks)
            if action == "navigate":
                target = np.random.choice(targets)
        else:

            # Case when navigate is a subtask
            if task in ["unload", "get_wood", "get_gold"]:
                best_q, best_action, best_target = -np.inf, None, None
                for action in sub_tasks:
                    if action == "navigate":
                        for target in targets:
                            q = get_max_value_function(state, action, value_fct,
                                                       completion_fct, task, target)
                            if q > best_q:
                                best_q, best_action, best_target = q, action, target
                    else:
                        q = get_max_value_function(state, action, value_fct,
                                                   completion_fct, task)
                        if q > best_q:
                            best_q, best_action = q, action
                action = best_action, target = best_target

            # When navigate is not a sub-task
            else:
                q = [get_max_value_function(state, action, value_fct, completion_fct, task)
                     for action in sub_tasks]
                action = sub_tasks[np.argmax(q)]

        # Action execution
        if action in PRIMITIVE_TASKS:
            game = primitive_action(game, action)
            reward = game.get_reward()
        elif action == "navigate":
            reward = max_q(state, action, game, value_fct, completion_fct, eps, target)
        else:
            reward = max_q(state, action, game, value_fct, completion_fct, eps)

        # Observe action result
        total_reward += reward
        next_state = game.get_state()

        # Update value or completion function
        if action in PRIMITIVE_TASKS:
            key = (state, action)
            value_fct[key] = (1-ALPHA)*value_fct[key] + ALPHA*reward
        elif task == "navigate":
            key = (task, state, action, destination)
            values = []
            for sub_task in sub_tasks[task]:
                max_v = get_max_value_function(next_state, sub_task, value_fct,
                                               completion_fct, task)
                cmp = completion_fct[(task, state, sub_task)]
                values.append(max_v + cmp)
            completion_fct[key] = (1 - ALPHA) * completion_fct[key] + \
                                  ALPHA * max(values)
        else:
            # TODO: TAKE navigate into account  !!!
            key = (task, state, action)
            values = []
            for sub_task in sub_tasks[task]:
                max_v = get_max_value_function(next_state, sub_task, value_fct, completion_fct, task)
                cmp = completion_fct[(task, state, sub_task)]
                values.append(max_v + cmp)
            completion_fct[key] = (1-ALPHA)*completion_fct[key] + \
                                  ALPHA*max(values)



    return total_reward





