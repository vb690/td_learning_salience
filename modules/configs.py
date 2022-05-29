def generate_ffmpeg_config():
    """Generate config used for turning png outputs from simulation into video

    Returns:

    """
    ffmpeg_configs = {
        "simulations": [
            {"agent_name": "addicted_01_wall", "steps": [0, 10000]},
            {"agent_name": "normal_wall", "steps": [0, 10000]},
        ],
        "ffmpeg": {"framerate": 5, "resolution": "1280x720"},
    }
    return ffmpeg_configs


def generate_simulation_config():
    """Generate config used for running simulations

    Returns:

    """
    pass