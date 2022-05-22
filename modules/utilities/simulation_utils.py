from ..environments import GridWorld
from ..agents import TDAgent


def run_simulation(
    agent, world, max_iter=10, max_steps=100, verbose=1, smooth_window=30
):
    """ """
    for agent_name, agent_kwargs in agent.items():
        grid_world = GridWorld(file_path=f"grid_worlds//{world}.txt")
        agent_obj = TDAgent(
            world=grid_world, agent_tag=f"{agent_name}_{world}", **agent_kwargs
        )
        sim_summary = agent_obj.simulate(
            max_iter=max_iter, verbose=verbose, max_steps=max_steps
        )
        for metric in ["steps", "reward", "error"]:

            sim_summary[metric] = sim_summary[metric].astype("float")
            sim_summary[f"smooth_{metric}"] = (
                sim_summary[metric]
                .rolling(window=min(smooth_window, max_iter // 10))
                .mean()
            )
            sim_summary[f"smooth_{metric}"] = (
                sim_summary[f"smooth_{metric}"].interpolate().ffill().bfill()
            )

        sim_summary["world"] = world
        sim_summary["agent"] = agent_obj.agent_tag.split("_")[0]

    return sim_summary
