import numpy as np

from sharedMemoryManager.sharedMemoryManager import *
from DQNetwork.DQnetwork import Agent

def main():
    initSharedMemoryReader()

    # Hyperparameteres
    lr = 0.001
    n_games = 500  # Nbr of games to play
    gamma = 0.99
    epsilon = 1.0
    batch_size = 64
    eps_min = 0.01
    eps_dec = 1e-4
    input_dims = (4,)  # Change selon la dimension de l'état
    n_actions = 4  # Nbr of possible actions

    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr, input_dims=input_dims, batch_size=batch_size, n_actions=n_actions, eps_end=eps_min, eps_dec=eps_dec)

    # init game loop
    scores = []
    for episode in range(n_games):
        observation = get_initial_game_state()  # get initial state of game (from Unity canal)

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

    closeSharedMemory()

def get_initial_game_state():
    """
    Function that returns the initial state of the game.
    Should be replaced by the real communication with Unity.
    """

    print("--> Waiting for memory ready state")
    while(isDataReady() == 1):
        print("------> Data is ready")

    pass

def send_action_and_get_state(action):
    """
    Function that simulates sending an action to the game and receiving the next state.
    Should be replaced by the real communication with Unity.
    """
    pass

if __name__ == '__main__':
    main()



# Position de l'Agent
# Informations sur l'Environnement Local
# ->Positions des obstacles à proximité (voitures, trains, rivières, bûches)
# ->Types d’obstacles
# Vitesse et Direction des Obstacles
# Distance de l'Agent aux Objectifs
# État de l’Agent (Vivant ou Mort)
# Vision Limitée :
# -> limiter ce que l'agent voit, seulement les 5 cases devant lui


# observation = [x, y, obs1_x, obs1_y, obs1_type, obs1_speed, obs1_dir, obs2_x, obs2_y, ..., distance_to_obj, alive] 