import numpy as np
import os
from constants.constants import *

from sharedMemoryManager.sharedMemoryManager import SharedMemoryManager
from DQNetwork.DQnetwork import Agent
from DQNetwork.model import save_model, get_latest_generation, load_model, load_latest_model
from utils.dataPloting import plotLearning

lastScore = 0

def main():
    # create a checkpoint directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    
    # Init shared memory manager
    SharedMemoryManager()



    agent = Agent(gamma = GAMMA, epsilon = EPSILON, lr = LR, input_dims = INPUT_DIMS, batch_size = BATCH_SIZE, n_actions = NB_ACTIONS, eps_end = EPS_MIN, eps_dec = EPS_DEC)

    current_generation = load_latest_model(agent)

    # Init game loop
    scores = []
    epsilons = []
    #modifier à la main entre les différents training?
    max_avg_score = 0 #22.557000000000013
    cpt_increase = 1
    cpt_episode = 0
    total_avg_score = 0
    episode = 0
    last_avg_score = -10
    while agent.epsilon != 0.01: #for episode in range(NB_GAMES):
        episode += 1
        observation = get_initial_game_state()

        done = False
        score = 0
        # signal_save = False

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
        epsilons.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        
        print(f"Episode {episode}, Score: {score}, Average Score: {avg_score}, Epsilon: {agent.epsilon:.2f}")

        cpt_episode += 1
        total_avg_score += avg_score


        #wait for the best time to save the model
        if (avg_score > max_avg_score):
            max_avg_score = avg_score
            last_avg_score = avg_score
            cpt_increase += 1
            cpt_episode = 0
            print(f"Max average score increased to {max_avg_score}, cpt_increase: {cpt_increase}")
            current_generation += 1
            save_model(agent, current_generation)
        else:
            #if no maximum is reached, take a score close to the maximum or take a average peak in the last 10 games
            if (((cpt_episode > 50) and ((max_avg_score - avg_score) < 0.5)) or ((cpt_episode > 10) and ((avg_score > (total_avg_score/cpt_episode)) and (avg_score > last_avg_score)))):
                last_avg_score = avg_score
                cpt_increase += 1
                cpt_episode = 0
                print(f"Average score increased to {avg_score}, cpt_increase: {cpt_increase}")
                current_generation += 1
                save_model(agent, current_generation)


        sharedMemoryManager = SharedMemoryManager()
        sharedMemoryManager.writeAt(1199, 10)
    
    sharedMemoryManager = SharedMemoryManager()
    del sharedMemoryManager

    x = [i+1 for i in range(episode)]
    filename = 'crossyroad.png'
    plotLearning(x, scores,epsilons, filename)

    print("Training finished")


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
        # print the shared memory matrix's size
    sharedMemoryManager.writeAt(1199, action + 1)

    while(not sharedMemoryManager.isDataReady()): pass

    done = True if sharedMemoryManager.parsedBuffer["player"]["alive"] == 0 else False

    global lastScore
    reward, lastScore = calculate_reward(sharedMemoryManager, lastScore, done)
    # newScore = (sharedMemoryManager.parsedBuffer["score"] - lastScore) * 10
    # if newScore <= 0: newScore = -1

    # reward = newScore if not done else -5
    # lastScore = sharedMemoryManager.parsedBuffer["score"]

    return sharedMemoryManager.matrixBuffer, reward, done


def calculate_reward(sharedMemoryManager, lastScore, done):
    """
    Calculate the reward for the agent's actions.
    """
    newScore = (sharedMemoryManager.parsedBuffer["score"] - lastScore) * 10
    if newScore <= 0:
        progress_reward = -1  
    else:
        progress_reward = newScore 

    # small reward for each step survived
    survival_reward = 0.1
    
    # death penalty if the agent dies
    death_penalty = -5 - (newScore * 0.2) if done else 0
    
    
    reward = progress_reward + survival_reward + death_penalty

    return reward, sharedMemoryManager.parsedBuffer["score"]

if __name__ == '__main__':
    main()

