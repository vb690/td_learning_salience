import pandas as pd

from multiprocessing import Pool

from modules.utilities.simulation_utils import run_simulation

MAX_ITER = 1000
MAX_STEPS = 500
WORLDS = ['treasure_island', 'grid', 'wall', 'double_wall', 'tbone', 'maze']

agents = [
    {'normal': {}},
    {'addicted': {'salience_factor': 1.5}}
]

args = []

for world in WORLDS:

    for agent in agents:

        args.append(
            (
                agent,
                world,
                MAX_ITER,
                MAX_STEPS,
                20
            )
        )

if __name__ == '__main__':

    pool = Pool()
    sim_summaries = pool.starmap(
        run_simulation,
        args
    )
    sim_summaries = pd.concat(
        sim_summaries,
        ignore_index=True
    )

    sim_summaries.to_csv(
        'results\\tables\\agents_comparison.csv'
    )
