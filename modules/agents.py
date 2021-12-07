import pandas as pd

import numpy as np

from modules.utilities import create_dir, sigmoid


class TDAgent:
    """
    """
    def __init__(self, world, alpha=0.1, gamma=0.9, eps=0.05,
                 salience_factor=1, dopamine_alteration=1,
                 agent_tag='', error_buffer=20, movement_cost=0.01,
                 actions=['up', 'down', 'left', 'right']):
        """
        """
        self.agent_tag = agent_tag

        self.actions = actions
        self.world = world

        self.error_buffer = error_buffer
        self.errors_history = [] * error_buffer
        self.rewards_history = []
        self.attributed_salience = {}

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.dopamine_alteration = dopamine_alteration
        self.salience_factor = salience_factor
        self.movement_cost = movement_cost

    def td_update(self, current_value, next_value, next_reward):
        """
        """
        error = next_reward + ((self.gamma * next_value) - current_value)
        error = error * self.dopamine_alteration
        updated_value = current_value + (self.alpha * error)
        return error, updated_value

    def update_salience(self, next_state, next_reward, error):
        """
        """
        def increment_reward_saliency(next_reward, error):
            """
            """
            new_reward = next_reward * self.salience_factor
            return min(new_reward, 1000)

        if next_state in self.attributed_salience:
            new_reward = increment_reward_saliency(
                next_reward=self.attributed_salience[next_state],
                error=error
            )
        else:
            new_reward = increment_reward_saliency(
                next_reward=next_reward,
                error=error
            )
        self.attributed_salience[next_state] = new_reward
        return None

    def get_salience(self, next_state, next_reward):
        """
        """
        if next_state in self.attributed_salience:
            return self.attributed_salience[next_state]
        else:
            return next_reward

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

        if all(next_state == self.world.terminal_state):
            next_reward = self.get_salience(
                next_state=next_state,
                next_reward=next_reward
            )

        error, updated_value = self.td_update(
            current_value=current_value,
            next_value=next_value,
            next_reward=next_reward - self.movement_cost
        )
        if all(next_state == self.world.terminal_state):
            self.update_salience(
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
            max_error, max_value, next_state = self.compute_updated_value(
                chosen_action,
                current_value
            )
        else:
            for index, action in enumerate(self.actions):

                error, updated_value, next_state = self.compute_updated_value(
                    action,
                    current_value
                )
                if index == 0:
                    chosen_action = action
                    max_value = updated_value
                    max_error = error
                else:
                    if updated_value >= max_value:
                        chosen_action = action
                        max_value = updated_value
                        max_error = error

        # we retrieve reward here without the incentive salience alteration
        self.rewards_history.append(
            self.world.get_reward(next_state)
        )

        self.world.update_value(max_value)
        max_error *= self.dopamine_alteration
        self.errors_history.append(max_error)

        return chosen_action

    def take_action(self, action):
        """
        """
        if action is not None:
            legal, current_state = self.world.get_state(action)
            self.world.update_state(current_state)
        return None

    def simulate(self, max_iter=1000, max_steps=300, verbose=50):
        """
        """
        iteration = 0
        step = 0
        create_dir(f'results//figures//{self.agent_tag}')
        create_dir(
            f'results//figures//{self.agent_tag}//{iteration}'
        )
        sim_summary = pd.DataFrame(
            columns=['iteration', 'steps', 'reward', 'error']
        )
        while iteration <= max_iter:

            if self.world.is_terminal() or step > max_steps:
                self.world.reset_state()
                self.world.reset_reward()
                sim_summary.loc[iteration] = [
                    iteration,
                    step,
                    np.sum(self.rewards_history),
                    np.sum(self.errors_history)
                ]
                self.errors_history = [0] * self.error_buffer
                self.rewards_history = []

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
                        errors=self.errors_history,
                        error_buffer=self.error_buffer,
                        step=step,
                        save_path=f'results//figures//{self.agent_tag}//{iteration}//{step}.png'
                    )
                step += 1

        return sim_summary
