import numpy as np
import time

from sharedMemoryManager.sharedMemoryManager import *
from globalKeyEventGenerator.globalKeyEventGenerator import *
from DQNetwork.DQnetwork import Agent

import win32gui
import win32process as wproc
import win32api as wapi

def winEnumHandler( hwnd, ctx ):
    if win32gui.IsWindowVisible( hwnd ):
        print(hex(hwnd), win32gui.GetWindowText(hwnd))
        if win32gui.GetWindowText(hwnd) == "Crossy road": win32gui.SetFocus(hwnd)

def main():
    # win32gui.EnumWindows(winEnumHandler, None)

    window_name = 'Crossy road'

    handle = win32gui.FindWindow(None, window_name)
    print("Window `{0:s}` handle: 0x{1:016X}".format(window_name, handle))

    if not handle:
        print("Invalid window handle")
        return
    
    remote_thread, _ = wproc.GetWindowThreadProcessId(handle)
    wproc.AttachThreadInput(wapi.GetCurrentThreadId(), remote_thread, True)
    prev_handle = win32gui.SetFocus(handle)
    
    PressKey(0x57)
    time.sleep(2)
    ReleaseKey(0x57) # Alt~

    SharedMemoryManager()

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

    memoryManager = SharedMemoryManager()
    del memoryManager

def get_initial_game_state():
    """
    Function that returns the initial state of the game.
    """

    while(not SharedMemoryManager().isDataReady()): pass
    return SharedMemoryManager().parsedBuffer

def send_action_and_get_state(action):
    """
    Function that simulates sending an action to the game and receiving the next state.
    """
    
    while(not SharedMemoryManager().isDataReady()): pass
    return SharedMemoryManager().parsedBuffer

if __name__ == '__main__':
    main()