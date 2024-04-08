# Sheng-Hung Kuan
from random import randint
from random import random
from random import choices

def approach(n):
    q_table = [[random() / 100.0, random() / 100.0] for i in range(n + 1)]  # 0: roll, 1: hold
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.8
    print("q_table before:")
    print(q_table)
    print("q_table after:")

    for i in range(100000):
        p1 = 0
        path = []
        p2 = 0
        reward = 0
        while p1 <= n - 6:  # initializing p1 (5, 6, 7, 8, 9, 10)
            p1 += randint(1, 6)
        while p1 <= n:  # p1 <= 10
            if q_table[p1][0] >= q_table[p1][1]:
                decision = choices([0, 1], weights=[1 - epsilon, epsilon], k=1)
            else:
                decision = choices([0, 1], weights=[epsilon, 1 - epsilon], k=1)
            if decision[0] == 1 :
                path.append((p1, decision[0]))
                break
            else:
                path.append((p1, decision[0]))
            p1 += randint(1, 6)

        if p1 <= n:
            while p2 <= p1:
                p2 += randint(1, 6)
            if p2 > n:
                reward = 1
        best_action = 0
        for action in path[::-1]:  # s: action[1], a: action[0], action: (9, hold)
            q_table[action[0]][action[1]] = q_table[action[0]][action[1]] \
                                            + alpha * (reward + gamma * best_action - q_table[action[0]][action[1]])
            best_action = q_table[action[0]][action[1]]

    print(q_table)
    print("optimal state")
    for index, state in enumerate(q_table):
        if index <= n - 6:  # 5, 6, 7, 8, 9, 10
            continue
        if state[0] > state[1]:
            print("state: ", index, "action: roll")
        else:
            print("state: ", index, "action: hold")



        # Select an initial state.
        # Take the best move with p=epsilon, and the worst move with p=1-epsilon.
        # Continue playing until the game is done.
        # If you win, reward = 1.
        # If you lose, reward = 0.
        ## Use Q-learning to update the q-table for each state-action pair visited.

    ## After 100000 iterations, print out your q-table.
