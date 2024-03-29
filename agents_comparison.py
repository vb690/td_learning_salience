import pandas as pd

from multiprocessing import Pool

from modules.utilities.simulation_utils import run_simulation

MAX_ITER = 10000
MAX_STEPS = 1000
WORLDS = [
    "arcipelagus"
]

agents = [
    {"normal": {"eps": 0.5, "min_eps": 0.3}},
    {
        "addicted_001": {"salience_factor": 1.01, "eps": 0.5, "min_eps": 0.3}
    },  # double every 100 iterations
    {
        "addicted_01": {"salience_factor": 1.1, "eps": 0.5, "min_eps": 0.3}
    },  # double every 10 iterations
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
