import pandas as pd

import numpy as np

from modules.utilities import create_dir, sigmoid


class TDAgent:
    """
    """
    def __init__(self, world, alpha=0.1, gamma=0.9, eps=0.05,
                 salience_factor=1, dopamine_alteration=1,
                 agent_tag='', error_buffer=20, movement_cost=0.1,
                 actions=['up', 'down', 'left', 'right']):
        """
        """
        self.agent_tag = agent_tag

        self.actions = actions
        self.world = world
        self.errors = [0] * error_buffer
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
            return min(new_reward, 100)

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
        return error, updated_value

    def pick_action(self):
        """
        """
        current_value = self.world.get_value()

        if np.random.uniform(0, 1) <= self.eps:
            chosen_action = np.random.choice(self.actions)
            max_error, max_value = self.compute_updated_value(
                chosen_action,
                current_value
            )
        else:
            for index, action in enumerate(self.actions):

                error, updated_value = self.compute_updated_value(
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

        log_error = np.log(abs(max_error))
        p_action = sigmoid(log_error)

        self.errors = self.errors[1:]
        self.errors.append(max_error)
        max_error *= self.dopamine_alteration
        self.world.update_value(max_value)
        return chosen_action

    def take_action(self, action):
        """
        """
        if action is not None:
            legal, current_state = self.world.get_state(action)
            self.world.update_state(current_state)
        return None

    def simulate(self, max_iter=1000, verbose=50):
        """
        """
        iteration = 0
        step = 0
        create_dir(f'results//figures//{self.agent_tag}')
        create_dir(
            f'results//figures//{self.agent_tag}//{iteration}'
        )
        sim_summary = pd.DataFrame(
            columns=['iteration', 'steps']
        )
        while iteration <= max_iter:

            if self.world.is_terminal():
                self.world.reset_state()
                self.world.reset_reward()
                self.errors = [0] * 20
                sim_summary.loc[iteration] = [
                    iteration,
                    step
                ]
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
                        errors=self.errors,
                        step=step,
                        save_path=f'results//figures//{self.agent_tag}//{iteration}//{step}.png'
                    )
                step += 1

        return sim_summary
