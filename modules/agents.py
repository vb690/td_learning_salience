import pandas as pd

import numpy as np

from modules.utilities.general_utils import create_dir


class TDAgent:
    """
    """
    def __init__(self, world=None, alpha=0.1, gamma=0.9, min_eps=0.05, eps=0.1,
                 salience_factor=1, agent_tag='', error_buffer=20,
                 movement_cost=0.05, actions=['up', 'down', 'left', 'right']):
        """
        """
        self.agent_tag = agent_tag

        self.actions = actions
        self.world = world

        self.error_buffer = [0] * error_buffer
        self.errors_history = 0
        self.rewards_history = 0
        self.attributed_salience = {}

        self.alpha = alpha
        self.gamma = gamma
        self.min_eps = min_eps
        self.eps = eps
        self.salience_factor = salience_factor
        self.movement_cost = movement_cost

    def increment_reward_saliency(self, next_reward, error):
        """
        """
        new_reward = next_reward * self.salience_factor
        return min(new_reward, 1000)

    def update_reward_saliency(self, next_state, next_reward, error):
        """
        """
        if next_state in self.attributed_salience:
            new_reward = self.increment_reward_saliency(
                next_reward=self.attributed_salience[next_state],
                error=error
            )
        else:
            new_reward = self.increment_reward_saliency(
                next_reward=next_reward,
                error=error
            )
        self.attributed_salience[next_state] = new_reward
        return None

    def get_reward_saliency(self, next_state, next_reward):
        """
        """
        if next_state in self.attributed_salience:
            return self.attributed_salience[next_state]
        else:
            return next_reward

    def td_update(self, current_value, next_value, next_reward):
        """
        """
        error = next_reward + ((self.gamma * next_value) - current_value)
        updated_value = current_value + (self.alpha * error)
        return error, updated_value

    def compute_updated_value(self, action, current_value):
        """
        """
        legal, next_state = self.world.get_state(action)
        # check legality
        if not legal:
            next_reward = -self.movement_cost
            next_value = current_value
        else:
            next_reward = self.world.get_reward(next_state)
            next_value = self.world.get_value(next_state)

        if all(next_state == self.world.salient_state):
            next_reward = self.get_reward_saliency(
                next_state=next_state,
                next_reward=next_reward
            )

        error, updated_value = self.td_update(
            current_value=current_value,
            next_value=next_value,
            next_reward=next_reward - self.movement_cost
        )
        if all(next_state == self.world.salient_state):
            self.update_reward_saliency(
                next_state=next_state,
                next_reward=next_reward,
                error=error
            )
        return error, updated_value, next_state

    def pick_action(self):
        """
        """
        current_value = self.world.get_value()

        if np.random.uniform(0, 1) <= self.eps:
            chosen_action = np.random.choice(self.actions)
            max_error, max_value, chosen_state = self.compute_updated_value(
                chosen_action,
                current_value
            )
        else:
            # we perfrom sequential action selection, it is very inefficient
            # but will allow us to plug in the dopamine dreprivation mechanisms
            # by Mc Clure et al. once we understand how to
            for index, action in enumerate(self.actions):

                error, updated_value, next_state = self.compute_updated_value(
                    action,
                    current_value
                )
                if index == 0:
                    chosen_state = next_state
                    chosen_action = action
                    max_value = updated_value
                    max_error = error
                else:
                    if updated_value >= max_value:
                        chosen_state = chosen_state
                        chosen_action = action
                        max_value = updated_value
                        max_error = error

        self.world.update_value(max_value)

        # we retrieve reward here without the incentive salience alteration
        self.rewards_history += self.world.get_reward(chosen_state)
        self.errors_history += max_error

        self.error_buffer = self.error_buffer[1:]
        self.error_buffer.append(max_error)

        return chosen_action

    def take_action(self, action):
        """
        """
        if action is not None:
            legal, current_state = self.world.get_state(action)
            self.world.update_state(current_state)
        return None

    def simulate(self, max_iter=1000, max_steps=300, verbose=50,
                 decay_ratio=3):
        """
        """
        iteration = 0
        step = 0
        eps_decay = (self.eps - self.min_eps) / (max_iter // decay_ratio)

        create_dir(f'results//figures//{self.agent_tag}')
        create_dir(
            f'results//figures//{self.agent_tag}//{iteration}'
        )

        sim_summary = pd.DataFrame(
            columns=['iteration', 'steps', 'reward', 'error', 'value']
        )
        while iteration <= max_iter:

            if self.world.is_terminal() or step > max_steps:

                sim_summary.loc[iteration] = [
                    iteration,
                    step,
                    self.rewards_history,
                    self.errors_history,
                    self.world.get_grid('value').flatten()
                ]
                self.world.reset_state()
                self.world.reset_reward()
                self.errors_history = 0
                self.rewards_history = 0
                self.error_buffer = [0] * len(self.error_buffer)

                iteration += 1
                step = 0
                if iteration % verbose == 0:
                    create_dir(
                        f'results//figures//{self.agent_tag}//{iteration}'
                    )
            else:
                self.take_action(self.pick_action())
                if iteration % verbose == 0:
                    self.world.show_grid(
                        episode=iteration,
                        error_buffer=self.error_buffer,
                        tag=self.agent_tag,
                        iteration=iteration,
                        step=step
                    )
                step += 1

            # linear decay in exploration
            if self.eps > self.min_eps:
                self.eps = self.eps - eps_decay

        return sim_summary
