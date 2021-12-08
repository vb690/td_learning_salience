import numpy as np

from matplotlib import colors
import matplotlib.pyplot as plt


class GridWorld:
    """
    """
    def __init__(self, file_path='grid_worlds\\the_wall.txt',
                 grid_dictionary=None):
        """
        """
        self.read_reward_grid_from_file(
            file_path=file_path,
            grid_dictionary=grid_dictionary
        )
        self.values_grid = np.random.uniform(
            0,
            1,
            size=self.rewards_grid.shape
        )
        self.current_state = self.start_state

        self.boundaries = [bound - 1 for bound in self.rewards_grid.shape]

    def read_reward_grid_from_file(self, file_path, grid_dictionary=None):
        """
        """
        if grid_dictionary is None:
            grid_dictionary = {
                '#': 0,
                ' ': 0,
                '*': -1,
                'S': 0,
                'R': 1.5,
                'T': 1
            }
            setattr(self, 'grid_dictionary', grid_dictionary)
        else:
            setattr(self, 'grid_dictionary', grid_dictionary)

        with open(file_path) as grid_file:

            grid_file = grid_file.read()

        grid = []
        for row in grid_file.split('\n'):

            row = list(row)
            if len(row) > 0:
                grid.append(row)

        grid = np.array(grid)
        setattr(self, 'start_state', np.argwhere(grid == 'S').flatten())
        setattr(self, 'terminal_state', np.argwhere(grid == 'T').flatten())
        setattr(self, 'trans_rew',  np.argwhere(grid == 'R'))
        grid = np.vectorize(grid_dictionary.get)(grid)
        setattr(self, 'rewards_grid', grid)
        return None

    def is_terminal(self):
        """
        """
        terminal = all(self.current_state == self.terminal_state)
        return terminal

    # ######################### REWARD RELATED FUNCTIONS ######################

    def get_reward(self, state=None):
        """
        """
        if state is None:
            y, x = self.current_state
        else:
            y, x = state
        reward = self.rewards_grid[y, x]
        return reward

    def update_reward(self, reward):
        """
        """
        y, x = self.current_state
        self.rewards_grid[y, x] = reward
        return None

    def reset_reward(self):
        """
        """
        for rew_location in self.trans_rew:

            y, x = rew_location
            self.rewards_grid[y, x] = self.grid_dictionary['R']

        return None

    # ######################### VALUE RELATED FUNCTIONS #######################

    def get_value(self, state=None):
        """
        """
        if state is None:
            y, x = self.current_state
        else:
            y, x = state
        value = self.values_grid[y, x]
        return value

    def update_value(self, value):
        """
        """
        y, x = self.current_state
        self.values_grid[y, x] = value
        return None

    def reset_value(self):
        """
        """
        self.values_grid = np.random.random(self.values_grid.shape)
        return None

    def save_value(self, value_path):
        """
        """
        np.save(value_path, self.values_grid)
        return None

    def load_value(self, value_path):
        """
        """
        self.values_grid = np.load(value_path)
        return None

    # ######################### STATE RELATED FUNCTIONS ######################

    def get_state(self, action):
        """
        """
        y, x = self.current_state
        if action == "up":
            next_state = (y + 1, x)
        elif action == "down":
            next_state = (y - 1, x)
        elif action == "left":
            next_state = (y, x - 1)
        elif action == 'right':
            next_state = (y, x + 1)
        elif action == 'stay':
            next_state = self.current_state
        else:
            raise Exception(f'Invalid Action {action}')

        # check legality
        y, x = next_state
        legal_y, legal_x = self.boundaries
        if ((x >= 0) and (x <= legal_x)) and ((y >= 0) and (y <= legal_y)):
            return True, next_state
        else:
            return False, self.current_state

    def update_state(self, state):
        """
        """
        self.current_state = state
        if any([all(self.current_state == rew_state) for
               rew_state in self.trans_rew]) and \
           self.rewards_grid[self.current_state] == self.grid_dictionary['R']:
            self.rewards_grid[self.current_state] = 0
        return None

    def reset_state(self):
        """
        """
        self.current_state = self.start_state
        return None

    # ######################### GRIDS RELATED FUNCTIONS #######################

    def get_grid(self, type_grid='reward'):
        """
        """
        if type_grid == 'reward':
            return self.rewards_grid
        elif type_grid == 'value':
            return self.values_grid
        elif type_grid == 'state':
            y, x = self.current_state[0], self.current_state[1]
            current_state_grid = np.zeros(self.rewards_grid.shape)
            current_state_grid[y, x] = 1
            return current_state_grid

    def show_grid(self, episode, errors, step, save_path, error_buffer=20):
        """
        """
        errors = errors[-20:]
        errors = errors - np.mean(errors)

        rewards_grid = self.get_grid(type_grid='reward')
        values_grid = self.get_grid(type_grid='value')
        current_state_grid = self.get_grid(type_grid='state')

        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        axs = axs.flatten()

        reward_cmap_mapper = {
            self.grid_dictionary['T']: 'g',
            self.grid_dictionary['*']: 'r',
            self.grid_dictionary['R']: 'gold',
            self.grid_dictionary['S']: 'w',
            self.grid_dictionary[' ']: 'w',
            self.grid_dictionary['#']: 'w'
        }
        reward_cmap = colors.ListedColormap(
            [reward_cmap_mapper[reward]
                for reward in np.unique(self.rewards_grid)]
        )
        state_cmap = colors.ListedColormap(['w', 'k'])

        axs[0].matshow(rewards_grid, cmap=reward_cmap)
        axs[0].set_title('Reward')

        axs[1].matshow(values_grid, cmap='coolwarm')
        axs[1].set_title(f'Value')

        axs[2].matshow(current_state_grid, cmap=state_cmap)
        axs[2].set_title(f'State')

        axs[3].plot([i for i in range(len(errors))], errors, c='r')
        yabs_max = abs(max(axs[3].get_ylim(), key=abs))
        axs[3].set_ylim(ymin=-yabs_max, ymax=yabs_max)
        axs[3].set_title(f'Error')
        axs[3].axvline(len(errors) - 1, linestyle='--', c='k')

        for index, ax in enumerate(axs):

            if index != 3:
                ax.set_yticks([])
            ax.set_xticks([])

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return None
