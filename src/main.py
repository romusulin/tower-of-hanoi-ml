import os
import pickle
from agent import QLearningAgent
from tower_of_hanoi import TowerOfHanoi

num_disks = 6

agent = QLearningAgent(table_row_size=num_disks * 1000)

filename = os.path.join(os.getcwd(), 'qtable-6.qta')
print('Saving q-table to: ', filename)

try:
    with open(filename, 'rb') as file:
        agent.q_table = pickle.load(file)
except FileNotFoundError:
    print('Starting fresh')


def run_simulation():
    episodes = 100_000
    total_moves = 0
    moves = []

    env = TowerOfHanoi(num_disks)
    for episode in range(1, episodes):
        state = env.reset()
        total_reward = 0
        while not env.is_done():
            actions = env.get_possible_actions()
            action = agent.choose_action(state, actions)
            from_peg, to_peg = action

            move = env.move_disk(from_peg, to_peg)

            reward = env.get_reward() if move else -100
            total_reward += reward

            if env.get_counter() > 200:
                total_reward -= 2000
                break

            next_state = env.get_state()
            agent.learn(state, actions.index(action), reward, next_state, actions)
            state = next_state

        total_moves += env.get_counter()
        moves.append(env.get_counter())
        if episode % 1000 == 0 or episode == episodes:
            print(
                f"\rEpisode {episode + 1}/{episodes}, Total Reward: {total_reward}, Avg moves: {str(total_moves / episode)}",
                end='')


run_simulation()

with open(filename, 'wb') as file:
    pickle.dump(agent.q_table, file)
print("Q-table saved")
