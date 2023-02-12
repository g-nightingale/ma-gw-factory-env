import numpy as np
import matplotlib.pyplot as plt
import torch
import utils as ut
import time

def add_noise(env_dims):
    """
    Add noise to stabilise training.
    """
    return np.random.rand(*env_dims).astype(np.float16)/100.0

def plot_rewards(training_metrics, avg_period=100, figname='rewards.jpg'):
    """"
    Plot rewards.
    """
    plt.figure(figsize=(8, 6))
    plt.title('sum of rewards')
    plt.plot(training_metrics['score_history'], label='episode score', c='darkorange', linewidth=1.0)
    plt.plot([np.mean(training_metrics['score_history'][::-1][i:i+avg_period]) for i in range(len(training_metrics['score_history']))][::-1], \
                      label=f'average reward (last {avg_period} episodes)', c='green')
    plt.xlabel("episodes")
    plt.ylabel("reward")
    # plt.ylim([-200, ])
    plt.legend()
    plt.savefig(figname)

def test_model(env, agent, max_steps=50, device='cpu'):
    """
    Test trained model.
    """

    env.reset()
    done = False
    episode_step_count = 0
    score = 0
    step_count = 0
    local_dims, global_dims = env.get_env_dims()

    while not done and step_count < max_steps: 
        step_count += 1
        episode_step_count += 1

        # Collect actions for each agent
        actions = []
        for agent_idx in np.arange(env.n_agents):
            # Get global and local states
            state_local_ = env.get_flattened_state(agent_idx=agent_idx).reshape(*local_dims) + ut.add_noise(local_dims)
            state_local = torch.from_numpy(state_local_).float().to(device)

            action, prob = agent.choose_action(state_local)

            # Append actions and probs
            actions.append(action)

        # Step the environment
        _, rewards, done = env.step(actions)
        score += sum(rewards)

        # Render the environment and sleep
        env.render()
        print()
        time.sleep(0.5)

    print(f'Total steps: {step_count} Total score: {score} Done: {done}')