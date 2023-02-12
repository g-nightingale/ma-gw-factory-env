import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size, device='cpu'):
        self.states_grid_local = []
        self.states_grid_global = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size
        self.device = device

    def generate_batches(self):
        n_states = len(self.states_grid_local)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return T.cat([x for x in self.states_grid_local]).to(self.device), \
                T.cat([x for x in self.states_grid_global]).to(self.device), \
                np.array(self.probs), \
                np.array(self.vals), \
                np.array(self.actions), \
                np.array(self.rewards), \
                np.array(self.dones), \
                batches

    def store_memory(self, 
                    state_grid_local, 
                    state_grid_global, 
                    action, 
                    probs, 
                    vals, 
                    reward, 
                    done):
        self.states_grid_local.append(state_grid_local)
        self.states_grid_global.append(state_grid_global)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states_grid_local = []
        self.states_grid_global = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, 
                n_actions,
                input_size,
                alpha=0.0003,
                name='ppo_actor',
                fc1_output_dim=256,
                fc2_output_dim=128,
                chkpt_dir='saved_models',
                device='cpu'):
        super().__init__()

        # Save parameters
        self.input_size = input_size
        self.fc1_output_dim = fc1_output_dim
        self.fc2_output_dim = fc2_output_dim

        # Create layers
        self.fc1 = nn.Linear(input_size, self.fc1_output_dim)
        self.fc2 = nn.Linear(self.fc1_output_dim, self.fc2_output_dim)
        self.fc3 = nn.Linear(self.fc2_output_dim, n_actions)
        self.sm = nn.Softmax(dim=1)
        
        self.device = device
        if device is not None:
            self.to(device)

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.n_actions = n_actions

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def softmax_with_temp(self, vals, temp):
        """
        Softmax policy - taken from Deep Reinforcement Learning in Action.
        """
        scaled_qvals = vals/temp
        norm_qvals = scaled_qvals - scaled_qvals.max() 
        soft = T.exp(norm_qvals) / T.sum(T.exp(norm_qvals))
        return soft

    def forward(self, state_grid, temp=1.0):
        x1 = T.relu(self.fc1(state_grid))
        x2 = T.relu(self.fc2(x1))
        x3 = self.fc3(x2)

        # x3 = self.sm(x3)
        x3 = self.softmax_with_temp(x3, temp=temp)
        dist = Categorical(x3)

        return dist, x3

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class CriticNetwork(nn.Module):
    def __init__(self, 
                input_size,
                alpha=0.0003,
                name='ppo_critic',
                fc1_output_dim=256,
                fc2_output_dim=128,
                chkpt_dir='saved_models',
                device='cpu'):
        super().__init__()
        
        # Save parameters
        self.input_size = input_size
        self.fc1_output_dim = fc1_output_dim
        self.fc2_output_dim = fc2_output_dim

        # Create layers
        self.fc1 = nn.Linear(self.input_size, self.fc1_output_dim)
        self.fc2 = nn.Linear(self.fc1_output_dim, self.fc2_output_dim)
        self.fc3 = nn.Linear(self.fc2_output_dim, 1)
        
        self.device = device
        if device is not None:
            self.to(device)

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state_grid):
        x1 = T.relu(self.fc1(state_grid))
        x2 = T.relu(self.fc2(x1))
        x3 = self.fc3(x2)

        return x3

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class MAPPOAgent:
    def __init__(self, 
                 n_actions, 
                 input_size_actor,
                 input_size_critic,
                 gamma=0.99, 
                 alpha=0.0003, 
                 gae_lambda=0.95,
                 policy_clip=0.2, 
                 batch_size=64, 
                 n_epochs=10,
                 softmax_temp=1.0,
                 use_normalized_values=False):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.softmax_temp = softmax_temp
        self.use_normalized_values = use_normalized_values

        self.actor = ActorNetwork(n_actions=n_actions, input_size=input_size_actor, alpha=alpha)
        self.critic = CriticNetwork(input_size=input_size_critic, alpha=alpha)
        self.memory = PPOMemory(batch_size)
       
    def store_memory(self, 
                    state_grid_local,
                    state_grid_global, 
                    action, 
                    probs, 
                    vals, 
                    reward, 
                    done):
        self.memory.store_memory(state_grid_local,
                                    state_grid_global,
                                    action,
                                    probs, 
                                    vals, 
                                    reward, 
                                    done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, 
                        state_grid_local):
        dist, _ = self.actor(state_grid_local, self.softmax_temp)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()

        return action, probs

    def get_state_value(self,
                        state_grid_global):
        value = self.critic(state_grid_global)
        value = T.squeeze(value).item()

        return value

    def get_entropy(self, states):
        # Calculate the entropy of the policy given the states
        logits = self.actor(states)
        probs = T.softmax(logits, dim=1)
        log_probs = T.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        return entropy

    def normalize_values(self, values):
        C = 1e-8
        mean = np.mean(values)
        std = np.std(values)
        normalized_values = (values - mean) / (std + C)
        return normalized_values

    def learn(self):
        for _ in range(self.n_epochs):
            state_grid_arr_local, \
            state_grid_arr_global, \
            old_prob_arr, \
            vals_arr,\
            action_arr, \
            reward_arr, \
            dones_arr, \
            batches = self.memory.generate_batches()

            values = vals_arr
            if self.use_normalized_values:
                values = self.normalize_values(values)
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Normalize the rewards
            reward_arr = (reward_arr - reward_arr.mean()) / (reward_arr.std() + 1e-5)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                grid_states_local = state_grid_arr_local[batch]
                grid_states_global = state_grid_arr_global[batch]

                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist, sm = self.actor(grid_states_local, self.softmax_temp)
                critic_value = self.critic(grid_states_global)

                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)

                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                                                 1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # entropy_bonus = T.sum(-sm * (T.log(sm + 1e-5)))

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss #+ 0.01*entropy_bonus

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()     

        return total_loss          
