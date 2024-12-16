# test_network.py

import torch
from agent.network import Actor, Critic

def test_actor():
    batch_size = 128
    input_dim = 16
    output_dim = 8
    actor = Actor(input_dim, output_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    output = actor(dummy_input)
    assert output.shape == (batch_size, output_dim), f"Expected output shape {(batch_size, output_dim)}, got {output.shape}"
    print("Actor network test passed.")

def test_critic():
    batch_size = 128
    state_dim = 64
    action_dim = 64
    critic = Critic(state_dim, action_dim)
    dummy_state = torch.randn(batch_size, state_dim)
    dummy_action = torch.randn(batch_size, action_dim)
    output = critic(dummy_state, dummy_action)
    assert output.shape == (batch_size, 1), f"Expected output shape {(batch_size, 1)}, got {output.shape}"
    print("Critic network test passed.")


if __name__ == "__main__":
    test_actor()
    test_critic()
