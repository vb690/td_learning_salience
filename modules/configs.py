def generate_ffmpeg_config():
    """Generate config used for turning png outputs from simulation into video

    Returns:

    """
    ffmpeg_configs = {
        "simulations": [
            {"agent_name": "addicted_01_wall", "steps": [0, 10000]},
            {"agent_name": "normal_wall", "steps": [0, 10000]},
            {"agent_name": "addicted_01_less_cost_wall", "steps": [0, 10000]},
            {"agent_name": "addicted_01_more_exploration_wall", "steps": [0, 10000]},

            {"agent_name": "addicted_01_triple_wall", "steps": [0, 10000]},
            {"agent_name": "normal_triple_wall", "steps": [0, 10000]},
            {"agent_name": "addicted_01_less_cost_triple_wall", "steps": [0, 10000]},
            {"agent_name": "addicted_01_more_exploration_triple_wall", "steps": [0, 10000]},

            {"agent_name": "addicted_01_treasure_island", "steps": [0, 10000]},
            {"agent_name": "normal_treasure_island", "steps": [0, 10000]},
            {"agent_name": "addicted_01_less_cost_treasure_island", "steps": [0, 10000]},
            {"agent_name": "addicted_01_more_exploration_treasure_island", "steps": [0, 10000]},

            {"agent_name": "addicted_01_grid", "steps": [0, 10000]},
            {"agent_name": "normal_grid", "steps": [0, 10000]},
            {"agent_name": "addicted_01_less_cost_grid", "steps": [0, 10000]},
            {"agent_name": "addicted_01_more_exploration_grid", "steps": [0, 10000]},

            {"agent_name": "addicted_01_tbone", "steps": [0, 10000]},
            {"agent_name": "normal_tbone", "steps": [0, 10000]},
            {"agent_name": "addicted_01_less_cost_tbone", "steps": [0, 10000]},
            {"agent_name": "addicted_01_more_exploration_tbone", "steps": [0, 10000]},

            {"agent_name": "addicted_01_maze", "steps": [0, 10000]},
            {"agent_name": "normal_maze", "steps": [0, 10000]},
            {"agent_name": "addicted_01_less_cost_maze", "steps": [0, 10000]},
            {"agent_name": "addicted_01_more_exploration_maze", "steps": [0, 10000]}
        ],
        "ffmpeg": {"framerate": 5, "resolution": "1280x720"},
    }
    return ffmpeg_configs


def generate_simulation_config():
    """Generate config used for running simulations

    Returns:

    """
    pass