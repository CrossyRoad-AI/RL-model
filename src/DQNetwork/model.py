import torch as T
import os

def save_model(agent,generation):
    model_path = f'checkpoints/model_gen_{generation}.pth'
    T.save(agent.Q_eval.state_dict(), model_path)
    print(f"Model saved as generation {generation}")


def get_latest_generation():
    files = os.listdir('checkpoints')
    generations = [int(f.split('_')[2].split('.')[0]) for f in files if f.startswith('model_gen_')]
    return max(generations) if generations else 0

def load_model(agent, filename):
    """Loads the model's state dictionary from a specified file."""
    agent.Q_eval.load_state_dict(T.load(filename))
    agent.Q_eval.eval()
    print(f"Model loaded from {filename}")


def load_latest_model(agent):
    generation = get_latest_generation()
    if generation > 0:
        model_path = f'checkpoints/model_gen_{generation}.pth'
        load_model(agent, model_path)
        print(f"Model generation {generation} loaded successfully")
    return generation