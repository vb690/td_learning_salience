import pandas as pd

from multiprocessing import Pool

from modules.utilities.simulation_utils import run_simulation

MAX_ITER = 10000
MAX_STEPS = 500
WORLDS = ["treasure_island", "grid", "wall", "double_wall", "tbone", "maze"]

agents = [
    {"normal": {}},
    {"addicted_001": {"salience_factor": 0.01}},
    {"addicted_005": {"salience_factor": 0.05}},
    {"addicted_01": {"salience_factor": 0.1}},
]

args = []

for world in WORLDS:

    for agent in agents:

        args.append((agent, world, MAX_ITER, MAX_STEPS, 2500))

if __name__ == "__main__":
    pool = Pool()
    sim_summaries = pool.starmap(run_simulation, args)
    sim_summaries = pd.concat(sim_summaries, ignore_index=True)

    sim_summaries.to_csv("results\\tables\\agents_comparison.csv")
