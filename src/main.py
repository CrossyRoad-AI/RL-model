import numpy as np
import os
from constants.constants import *

from sharedMemoryManager.sharedMemoryManager import SharedMemoryManager
from DQNetwork.DQnetwork import Agent
from DQNetwork.model import save_model, get_latest_generation, load_model, load_latest_model
from utils.dataPloting import plotLearning



def main():
    # create a checkpoint directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    
    # Init shared memory manager
    sharedMemoryManager=SharedMemoryManager()

    agent = Agent(gamma = GAMMA, epsilon = EPSILON, lr = LR, input_dims = INPUT_DIMS, batch_size = BATCH_SIZE, n_actions = NB_ACTIONS, eps_end = EPS_MIN, eps_dec = EPS_DEC)

    current_generation = load_latest_model(agent)

    # Init game loop
    scores = []
    epsilons = []
    #to modify for the general loop
    max_avg_score = -10 #22.557000000000013 for gen 701
    cpt_increase = 1
    cpt_episode = 0
    total_avg_score = 0
    last_avg_score = -10
    last_player_position = None
    current_player_position = None
    step_count = 0
    for episode in range(NB_GAMES):
        observation = get_initial_game_state()
        done = False
        score = 0
        lastScore = 0

        while not done:
            # choose an action based on the current state
            action = agent.choose_action(observation)
                # Update current player position for new observation

            # send the action to the game and get the new state
            new_observation, done = send_action_and_get_state(action)
            for row_idx, row in enumerate(new_observation):
                if 1 in row:  # Player position marker
                    current_player_position = (row_idx, row.index(1))
                    break
            reward, lastScore = calculate_reward(sharedMemoryManager, lastScore, done, last_player_position)
            last_player_position = current_player_position
            # update agent's memory with this transition
            agent.store_transition(observation, action, reward, new_observation, done)

            # update the current state
            observation = new_observation
            # update the score
            score += reward
            step_count += 1

            # train the agent if the batch size is reached
            agent.learn()

        # stock and print the score at each episode
        scores.append(score)
        epsilons.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        
        print(f"Episode {episode}, Score: {score}, Average Score: {avg_score}, Epsilon: {agent.epsilon:.2f}")
        cpt_episode += 1
        total_avg_score += avg_score


        if agent.epsilon < 0.60 :
            if (avg_score > max_avg_score):
                max_avg_score = avg_score
                last_avg_score = avg_score
                cpt_increase = 1
                cpt_episode = 0
                print(f"Max average score increased to {max_avg_score}, cpt_increase = {cpt_increase}")
                current_generation += 1
                save_model(agent, current_generation)
            else:
                if (((cpt_episode > 50) and ((max_avg_score - avg_score) < 0.5)) or ((cpt_episode > 10) and ((avg_score > (total_avg_score/cpt_episode)) and (avg_score > last_avg_score)))):
                    last_avg_score = avg_score
                    cpt_increase += 1
                    cpt_episode = 0
                    print(f"Average score increased to {avg_score}, cpt_increase: {cpt_increase}")
                    current_generation += 1
                    save_model(agent, current_generation)



        cpt_episode += 1
        total_avg_score += avg_score


        #wait for the best time to save the model
        if agent.epsilon < 0.75 :
            if (avg_score > max_avg_score):
                max_avg_score = avg_score
                last_avg_score = avg_score
                cpt_increase += 1
                cpt_episode = 0
                print(f"Max average score increased to {max_avg_score}, cpt_increase: {cpt_increase}")
                current_generation += 1
                save_model(agent, current_generation)
            else:
                #if no maximum is reached, take a score close to the maximum or take an average peak in the last 10 games
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


    return sharedMemoryManager.matrixBuffer, done





def calculate_reward(sharedMemoryManager, lastScore, done, last_player_position=None):
    currentScore = sharedMemoryManager.parsedBuffer["score"]
    score_diff = currentScore - lastScore
    
    matrix = sharedMemoryManager.matrixBuffer
    
    
    # Find the player's current row
    current_player_position = None
    player_row = None
    for row_idx, row in enumerate(matrix):
        if 1 in row:
            current_player_position = (row_idx, row.index(1))
            break
    
    
    # Detect water lily bonus
    water_lily_bonus = 0
    if current_player_position:
        current_row, current_col = current_player_position
        
        water_lily_positions = [
            (current_row, current_col),
            (current_row-1, current_col),  
            (current_row+1, current_col),  
            (current_row, current_col-1),  
            (current_row, current_col+1)   
        ]
        
        # Check if any of these positions contain a water lily
        for check_row, check_col in water_lily_positions:
            if (0 <= check_row < len(matrix) and 
                0 <= check_col < len(matrix[0])):
                if matrix[check_row][check_col] == 0.8:
                    water_lily_bonus = 5  # Bonus for being near/on a water lily
                    break
        
        # bonus for moving to/through water lilies
        if last_player_position:
            last_row, last_col = last_player_position
            # Check if the path between last and current position contained a water lily
            for r in range(min(last_row, current_row), max(last_row, current_row) + 1):
                for c in range(min(last_col, current_col), max(last_col, current_col) + 1):
                    if matrix[r][c] == 0.8:
                        water_lily_bonus += 3
                        break
    
    progress_reward = score_diff * 20
    survival_reward = 0.5
    
    lateral_movement_penalty = 0
    if last_player_position and current_player_position:
        last_row, last_col = last_player_position
        current_row, current_col = current_player_position
        if last_row == current_row and last_col != current_col:
            lateral_movement_penalty = -0.2
        
        if abs(last_col - current_col) > 1:
            lateral_movement_penalty = -0.5
    
    death_penalty = -10 - (score_diff * 3) if done else 0
    stuck_penalty = -0.5 if score_diff == 0 else 0
    
    total_reward = (
        progress_reward + 
        survival_reward + 
        lateral_movement_penalty + 
        death_penalty + 
        stuck_penalty +
        water_lily_bonus
    )
    
    return total_reward, currentScore


if __name__ == '__main__':
    main()

