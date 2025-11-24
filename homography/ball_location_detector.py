# This is a modification of the work done at https://github.com/nreHieW/Eagle/tree/main

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # only for macs
from coordinate_model import CoordinateModel
from processor import Processor
import json
from argparse import ArgumentParser
import pandas as pd
import cv2
import numpy as np
import cv2
import os


def read_video(path: str, fps: int = 24) -> list:
    """
    Read a video file and return a list of frames.
    :param path: Path to the video file.
    :param fps: Frames per second to sample.

    :return: List of frames and the fps of the video.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    cap = cv2.VideoCapture(path)
    frames = []
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    skip = native_fps // fps
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames, fps


def write_video(frames: list, path: str, fps: int = 24, is_rgb: bool = False) -> str:
    """
    Write a list of frames to a video file.
    :param frames: List of images.
    :param path: Path to save the video file.
    :param fps: Frames per second.

    :return: Path to the saved video file.
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        if is_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # cv2 expects BGR
        out.write(frame)
    out.release()
    return path


def main():

    video_path = 'shot_new.mp4'
    fps=24
    os.makedirs("output", exist_ok=True)
    video_name = 'shot_new.mp4'
    os.makedirs(f"output/{video_name}", exist_ok=True)
    root = f"output/{video_name}"

    frames, fps = read_video(video_path, fps)
    model = CoordinateModel()
    coordinates = model.get_coordinates(frames, fps//2, num_homography=0.5, num_keypoint_detection=0.5)

    print("Processing Data")

    # `coordinates` may contain a reserved profiling key (`_Profiling`) added by the
    # coordinate model. Remove it before creating the Processor so lengths match.
    if isinstance(coordinates, dict) and "_Profiling" in coordinates:
        coordinates = {k: v for k, v in coordinates.items() if k != "_Profiling"}

    processor = Processor(coordinates, frames, fps, filter_ball_detections=False)
    df, _ = processor.process_data(smooth=False)
    processed_df = processor.format_data(df)
    # processed_df.to_json(f"{root}/processed_data.json", orient="records")


    for i, _ in df.iterrows():
        print(i)
        try:
            ball = coordinates[i]["Coordinates"]["Ball"][0]["Transformed_Coordinates"]
            print("ball", ball)
            print()
        except KeyError:
            continue
        print()

    # write_video(out, f"{root}/annotated.mp4", fps)


if __name__ == "__main__":
    main()
