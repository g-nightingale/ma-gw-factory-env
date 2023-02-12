from simple_ma_gw import MultiAgentGridWorld
import numpy as np
import torch
from matplotlib import pyplot as plt
import utils as ut
from mappo import PPOMemory, ActorNetwork, CriticNetwork, MAPPOAgent
import train_mappo as tppo_mappo
import wandb

def main(test_agent=False):
    SEED = 3
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = MultiAgentGridWorld(max_step_count=500,
                             delivery_win_count=1)
    local_dims, global_dims = env.get_env_dims()

    log_to_wand = False

    run_description = ""
    n_actions = env.action_space
    n_episodes = 500
    learning_steps = 256
    max_steps = 512
    batch_size = 128
    n_epochs = 3 # Too many epochs and model can overfit
    alpha = 0.00003
    policy_clip = 0.2
    gae_lambda = 0.95
    softmax_temp = 0.9
    verbose_episodes = 1
    use_normalized_values = False

    if log_to_wand:
        wandb.init(
            project="simple_ma_gw",
            config = {
                "run_description":run_description,
                "n_episodes":n_episodes,
                "learning_steps":learning_steps,
                "max_steps":max_steps,
                "batch_size":batch_size,
                "n_epochs":n_epochs,
                "alpha":alpha,
                "policy_clip":policy_clip,
                "gae_lambda":gae_lambda,
                "softmax_temp":softmax_temp,
                "verbose_episodes":verbose_episodes,
                "use_normalized_values":use_normalized_values
            }
        )

    mappo_agent = MAPPOAgent(n_actions=n_actions, 
                            input_size_actor=local_dims[1],
                            input_size_critic=global_dims[1],
                            batch_size=batch_size,
                            alpha=alpha, 
                            policy_clip=policy_clip,
                            gae_lambda=gae_lambda,
                            n_epochs=n_epochs,
                            softmax_temp=softmax_temp,
                            use_normalized_values=use_normalized_values)
    
    training_metrics = tppo_mappo.train_ppo(env,
                            mappo_agent,
                            n_episodes,
                            learning_steps=learning_steps,
                            max_steps=max_steps,
                            verbose_episodes=verbose_episodes,
                            device='cpu')
    
    # Plot rewards
    ut.plot_rewards(training_metrics, avg_period=100)

    # Test trained model
    if test_agent:
        ut.test_model(env, mappo_agent, device='cpu')

if __name__ == '__main__':    
    main()