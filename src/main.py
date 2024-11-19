import numpy as np

from constants.constants import *

from sharedMemoryManager.sharedMemoryManager import SharedMemoryManager
from DQNetwork.DQnetwork import Agent

lastScore = 0

def main():
    # Init shared memory manager
    SharedMemoryManager()

    agent = Agent(gamma = GAMMA, epsilon = EPSILON, lr = LR, input_dims = (INPUT_DIMS,), batch_size = BATCH_SIZE, n_actions = NB_ACTIONS, eps_end = EPS_MIN, eps_dec = EPS_DEC)

    # Init game loop
    scores = []
    for episode in range(NB_GAMES):
        observation = get_initial_game_state()

        done = False
        score = 0

        while not done:
            # choose an action based on the current state
            action = agent.choose_action(observation)

            # send the action to the game and get the new state
            new_observation, reward, done = send_action_and_get_state(action)
            
            # update agent's memory with this transition
            agent.store_transition(observation, action, reward, new_observation, done)

            # update the current state
            observation = new_observation

            # update the score
            score += reward

            # train the agent if the batch size is reached
            agent.learn()

        # stock and print the score at each episode
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        
        print(f"Episode {episode}, Score: {score}, Average Score: {avg_score}, Epsilon: {agent.epsilon:.2f}")

        sharedMemoryManager = SharedMemoryManager()
        sharedMemoryManager.writeAt(1199, 10)

    sharedMemoryManager = SharedMemoryManager()
    del sharedMemoryManager

def get_initial_game_state():
    """
    Function that returns the initial state of the game.
    """

    sharedMemoryManager = SharedMemoryManager()
    while(not sharedMemoryManager.isDataReady()): pass

    # return sharedMemoryManager.listBuffer
    return sharedMemoryManager.matrixBuffer

def send_action_and_get_state(action):
    """
    Function that simulates sending an action to the game and receiving the next state.
    """

    sharedMemoryManager = SharedMemoryManager()
    sharedMemoryManager.writeAt(1199, action + 1)

    while(not sharedMemoryManager.isDataReady()): pass

    done = True if sharedMemoryManager.parsedBuffer["player"]["alive"] == 0 else False

    global lastScore
    newScore = (sharedMemoryManager.parsedBuffer["score"] - lastScore) * 10
    if newScore <= 0: newScore = -1

    reward = newScore if not done else -5
    lastScore = sharedMemoryManager.parsedBuffer["score"]

    return sharedMemoryManager.matrixBuffer, reward, done

if __name__ == '__main__':
    main()