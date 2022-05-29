import os

from modules.configs import generate_ffmpeg_config

if __name__ == "__main__":

    config = generate_ffmpeg_config()

    framerate = config["ffmpeg"]["framerate"]
    resolution = config["ffmpeg"]["resolution"]

    for simulation in config["simulations"]:

        agent_name = simulation["agent_name"]

        for step in simulation["steps"]:

            input_dir = f"results\\figures\\{agent_name}\\{step}"
            output_dir = f"results\\videos"

            os.system(
                f"ffmpeg -y -r {framerate} -f image2 -s {resolution} -i {input_dir}\\%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {output_dir}\\{agent_name}_{step}.mp4"
            )
