import os
import shutil

from tqdm import tqdm

import numpy as np

import pandas as pd

from .environments import GridWorld
from .agents import TDAgent


def sigmoid(x, alpha=1, beta=0):
    """Compute sigmoid of x given midpoint and
    steepness
    """
    p = 1 / (1 + np.exp(-alpha*(x - beta)))
    return p


def create_dir(dir_name):
    """Create a directory given a directory location. If the directory already
    exists it will be removed.
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)
    return None


def run_simulations(agents, worlds, save_name, max_iter=10, max_steps=100,
                    verbose=1):
    """
    """
    sim_summaries_tot = []
    for world in worlds:

        print(f'Testing {world}')
        grid = GridWorld(file_path=f'grid_worlds//the_{world}.txt')

        for agent_name, agent_kwargs in tqdm(agents.items()):

            agent = TDAgent(
                world=grid,
                agent_tag=f'{agent_name}_the_{world}',
                **agent_kwargs
            )
            print(agent)

            sim_summary = agent.simulate(
                max_iter=max_iter,
                verbose=verbose,
                max_steps=max_steps
            )
            for metric in ['steps', 'reward', 'error']:

                sim_summary[metric] = sim_summary[metric].astype('float')
                sim_summary[f'smooth_{metric}'] = sim_summary[metric].rolling(
                    window=min(30, max_iter // 10),
                    min_periods=1
                ).mean()

            sim_summary['world'] = world
            sim_summary['agent'] = agent.agent_tag.split('_')[0]
            sim_summaries_tot.append(sim_summary)

    sim_summaries_tot = pd.concat(sim_summaries_tot, ignore_index=True)
    sim_summaries_tot.to_csv(
        f'results//tables//{save_name}.csv'
    )
    return sim_summaries_tot
