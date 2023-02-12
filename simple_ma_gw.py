import numpy as np


class MultiAgentGridWorld:
    """
    A simple multi-agent grid world class.

    Tile values:
    ------------
    0: Open space
    1: Block tile
    2: Agent
    3: Goal tile
    """

    def __init__(self,
                game_mode='static',
                size=5,
                n_agents=2,
                delivery_win_count=1000,
                use_global_rewards=True,
                max_step_count=500):
        self.game_mode = game_mode
        self.size = size
        self.n_agents = n_agents
        self.action_space = 3
        self.delivery_win_count = delivery_win_count
        self.use_global_rewards = use_global_rewards
        self.max_step_count = max_step_count
        self.REWARD_STEP = -1
        self.REWARD_WIN = 100
        self.OPEN_TILE = 0
        self.BLOCK_TILE = 1
        self.AGENT_TILE = 2
        self.GOAL_TILE = 3
        self.reset()

    def reset(self):
        """
        Reset the environment.
        """
        self.step_count = 0
        self.done = False
        self.win_count = 0
        
        if self.game_mode=='static':
            # Generate grid
            self.size = 5
            self.n_agents = 2
            self.grid = np.zeros((self.n_agents, self.size), dtype=np.uint8)
        
            # Set agent positions
            self.agent_positions = {
                0: (0, 3),
                1: (1, 1)
            }

            # Populate blocks
            self.grid[0, self.size-1] = self.GOAL_TILE
            self.grid[0, 2] = self.BLOCK_TILE
            
        elif self.game_mode=='random':
            # Generate grid
            self.grid = np.zeros((self.n_agents, self.size), dtype=np.uint8)

            # Set blocks and agent positions
            self.agent_positions = {}
            col_prev = self.size//2
            for agent_idx in range(self.n_agents):
                if agent_idx==0:
                    block_col = np.random.randint(col_prev, self.size - 2)
                    agent_col = np.random.randint(block_col + 1, self.size - 1)
                else:
                    block_col = np.random.randint(0, col_prev)
                    agent_col = np.random.randint(block_col + 1, self.size - 1)

                # Set agent positions
                self.agent_positions[agent_idx] = (agent_idx, agent_col)
                
                # Populate block tiles
                if agent_idx < self.n_agents - 1:
                    self.grid[agent_idx, block_col] = self.BLOCK_TILE
                
                col_prev = block_col + 1

            # Populate goal block
            self.grid[0, self.size-1] = self.GOAL_TILE

        else:
            raise ValueError("Game mode not recognised. Must be static or random.")

        # Populate agents
        for position in self.agent_positions.values():
            self.grid[position] = self.AGENT_TILE

        self.has_goal_tile = {agent_idx:0 for agent_idx in range(self.n_agents)}
        
    def render(self):
        """
        Render the environment.
        """
        print(self.grid)

    def step(self, actions):
        """
        Step the environment.
        """
        self.step_count += 1
        rewards = []
        for agent_idx, action in enumerate(actions):
            reward = self.movement_handler(agent_idx, action)
            rewards.append(reward)
        
        if self.use_global_rewards:
            rewards = [sum(rewards) for _ in rewards]

        if self.step_count >= self.max_step_count:
            self.done - True

        return self.grid, rewards, self.done

    def movement_handler(self, agent_idx, action):
        """
        Handle agent movement.
        """
        # Determine new positions for each action
        current_pos = self.agent_positions[agent_idx]
        left_pos = (current_pos[0], current_pos[1] - 1)
        right_pos = (current_pos[0], current_pos[1] + 1)
        down_pos = (current_pos[0] + 1, current_pos[1])
        reward = 0

        #----------------------------------------------------------------------
        # Move left
        #----------------------------------------------------------------------
        if action==0 and current_pos[1] > 0:
            if self.grid[left_pos]==self.OPEN_TILE:
                self.grid[left_pos] = self.AGENT_TILE
                self.grid[current_pos] = self.OPEN_TILE
                self.agent_positions[agent_idx] = left_pos
                # Check for the win condition
                if left_pos[1]==0 and self.has_goal_tile[agent_idx]==1:
                    reward += self.REWARD_WIN 
                    self.win_count += 1
                    self.has_goal_tile[agent_idx] = 0
                    # Move agent if it occupies the goal spawn tile
                    if self.grid[0, self.size-1] == self.AGENT_TILE:
                        self.grid[0, self.size-2] = self.AGENT_TILE
                    # Spawn new goal tile
                    self.grid[0, self.size-1] = self.GOAL_TILE
                    # Check for termination
                    if self.win_count >= self.delivery_win_count:
                        self.done = True                   
            elif self.grid[left_pos]==self.GOAL_TILE:
                self.grid[left_pos] = self.AGENT_TILE
                self.grid[current_pos] = self.OPEN_TILE
                self.agent_positions[agent_idx] = left_pos
                self.has_goal_tile[agent_idx] = 1
                
        #----------------------------------------------------------------------
        # Move right
        #----------------------------------------------------------------------
        elif action==1 and current_pos[1] < self.size - 1:
            if self.grid[right_pos]==self.OPEN_TILE:
                self.grid[right_pos] = self.AGENT_TILE
                self.grid[current_pos] = self.OPEN_TILE
                self.agent_positions[agent_idx] = right_pos
            elif self.grid[right_pos]==self.GOAL_TILE:
                self.grid[right_pos] = self.AGENT_TILE
                self.grid[current_pos] = self.OPEN_TILE
                self.agent_positions[agent_idx] = right_pos
                self.has_goal_tile[agent_idx] = 1

        #----------------------------------------------------------------------
        # Pass box down
        #----------------------------------------------------------------------
        elif action==2 and current_pos[0] < self.n_agents - 1:
            # Pass into an empty square
            if self.has_goal_tile[agent_idx] and self.grid[down_pos]==self.OPEN_TILE:
                self.grid[down_pos] = self.GOAL_TILE
                self.has_goal_tile[agent_idx] = 0
            # Passing directly to another agent
            if self.has_goal_tile[agent_idx] and self.grid[down_pos]==self.AGENT_TILE:
                self.has_goal_tile[down_pos[0]] = 1
                self.has_goal_tile[agent_idx] = 0

        # Step reward gets added every single step
        reward += self.REWARD_STEP

        return reward

    def get_flattened_state(self, agent_idx=None, actions=None):
        """
        Get the flattened state.
        """

        flattened_state = self.grid.flatten()
        has_goal_tile_np = np.array(list(self.has_goal_tile.values()), dtype=np.uint8)
        if agent_idx is not None:
            agent_flags = np.zeros(self.n_agents, dtype=np.uint8)
            agent_flags[agent_idx] = 1
            flattened_state = np.concatenate((agent_flags, flattened_state, has_goal_tile_np), axis=0, dtype=np.uint8)
        
        # Add actions to the vector
        if actions is not None:
            actions_np = np.zeros(self.n_agents*self.action_space, dtype=np.uint8)
            for i, action in enumerate(actions):
                actions_np[(i*self.action_space-1)+action] = 1
            flattened_state = np.concatenate((flattened_state, has_goal_tile_np, actions_np), axis=0, dtype=np.uint8)

        return flattened_state

    def get_env_dims(self):
        """
        Get local and global dims.
        """
        local_dims = (1, self.n_agents * self.size + (self.n_agents*2))
        global_dims = (1, (self.n_agents * self.size) + self.n_agents + (self.n_agents * self.action_space))
        return local_dims, global_dims


def move_randomly(env, max_steps=5):
    """"
    Test function to test random movement.
    """
    env.render()
    print()
    done = False
    steps = 0
    while done is False and steps < max_steps:
        actions = np.random.randint(3, size=env.n_agents)
        print(actions)
        _, rewards, done = env.step(actions)
        env.render()
        steps += 1
        print(rewards, done)
        print(env.get_flattened_state())
        print(steps)
        print()


if __name__ == '__main__':
    env = MultiAgentGridWorld()
    env.render()

    local_state = env.get_flattened_state()
    global_state = env.get_flattened_state(actions=[0, 1])

    print(local_state, '\n')
    print(local_state.shape, '\n')
    print(global_state, '\n')
    print(global_state.shape, '\n')

    print(env.get_env_dims())
    move_randomly(env, max_steps=100)
