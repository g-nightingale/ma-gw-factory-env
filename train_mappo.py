import numpy as np
from collections import defaultdict
import torch
from IPython.display import clear_output
import utils as ut
import wandb

def train_ppo(env,
            agent,
            n_episodes,
            learning_steps=20,
            max_steps=1000,
            verbose_episodes=5,
            max_no_score_improvement=20,
            device='cpu',
            log_to_wand=False):
    """
    Train PPO agent.
    """

    step_count = 0
    done_count = 0
    no_score_improvement = 0

    training_metrics = {
        "score_history": [], 
        "losses": [],
        "episode_step_counts": [],
    }

    local_dims, global_dims = env.get_env_dims()

    for i in range(n_episodes):
        env.reset()
        done = False
        episode_step_count = 0
        score = 0
        
        while not done: 
            step_count += 1
            episode_step_count += 1

            # Collect actions for each agent
            states = []
            actions = []
            probs = []
            vals = []
            for agent_idx in np.arange(env.n_agents):
                # Get global and local states
                state_local_ = env.get_flattened_state(agent_idx=agent_idx).reshape(*local_dims) + ut.add_noise(local_dims)
                state_local = torch.from_numpy(state_local_).float().to(device)

                action, prob = agent.choose_action(state_local)

                # Append actions and probs
                states.append(state_local)
                actions.append(action)
                probs.append(prob)

            # Step the environment
            _, rewards, done = env.step(actions)

            # Increment score
            prev_score = score
            score += sum(rewards)

            # Create the global metadata state: state + actions
            state_global_ = env.get_flattened_state(actions=actions).reshape(*global_dims) + ut.add_noise(global_dims)
            state_global = torch.from_numpy(state_global_).float().to(device)

            for agent_idx in np.arange(env.n_agents):
                val = agent.get_state_value(state_global)
                vals.append(val)

            # Store each agent experiences
            for agent_idx in np.arange(env.n_agents):
                # Append replay buffer
                agent.store_memory(states[agent_idx],
                                    state_global,  
                                    actions[agent_idx], 
                                    probs[agent_idx],
                                    vals[agent_idx],
                                    rewards[agent_idx], 
                                    done)

            # Learning
            if step_count % learning_steps == 0 and step_count > agent.batch_size:
                loss = agent.learn().detach().numpy()
            else:
                loss = 0.0

            # Termination -> Append metrics

            if done or episode_step_count > max_steps:
                training_metrics['losses'].append(loss)   

                done_count += 1 * done
                done = True
                
        training_metrics['score_history'].append(score)
        training_metrics['episode_step_counts'].append(episode_step_count)
        if log_to_wand:
            wandb.log({"score": score})
            wandb.log({"episode_step_count": episode_step_count})
            wandb.log({"avg_score_10": np.mean(training_metrics['score_history'][-10:])})
            wandb.log({"avg_score_25": np.mean(training_metrics['score_history'][-25:])})
            wandb.log({"avg_score_100": np.mean(training_metrics['score_history'][-100:])})
        

        if i % verbose_episodes == 0:
            clear_output(wait=True)
            print(f"ep: {i+1} \ttsc: {step_count} \tesc: {episode_step_count} \tscr: {score} \tavg scr: {np.mean(training_metrics['score_history'][-100:])}") 

        # if avg_score > best_score:
        #     best_score = avg_score
        #     agent.save_models()
        # if score <= prev_score:
        #     no_score_improvement += 1
        # else:
        #     no_score_improvement = 0

        # if no_score_improvement >= max_no_score_improvement:
        #     return training_metrics

    return training_metrics