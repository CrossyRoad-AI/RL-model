import torch as T

def save_model(agent, filename):
    """Saves the model's state dictionary to a specified file."""
    T.save(agent.Q_eval.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(agent, filename):
    """Loads the model's state dictionary from a specified file."""
    agent.Q_eval.load_state_dict(T.load(filename))
    agent.Q_eval.eval()
    print(f"Model loaded from {filename}")


