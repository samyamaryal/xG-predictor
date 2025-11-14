from moviepy import *
import torch
import torchvision
import numpy as np
import json
import cv2
import os

INPUT_NO_OF_FRAMES = 32
N_NON_SHOT_EVENTS = 15

ROOT_DIR = "data/SoccerNet"

save_directory = "processed_data"
BATCH_PREFIX = "non_shot_batch"
BATCH_SIZE = 32  # number of clips per .npz file

# All non-shot events from the game, labels extracted from JSON 

# If a shot was taken, I extract the time (in seconds) from that shot, and then extract a
# frame sequence of 'n' frames leading upto that shot. 

def time_converter(time_in):
    half, time = time_in.split("-") # Half and time are split on this
    mins, seconds = time.split(":")
    timestamp_seconds = int(mins)*60+int(seconds) 
    return half.strip(), timestamp_seconds

# For all shots: 
X = []
y = []


tol = 5  # seconds of tolerance (±tol)

def within_tol(ts, labeled_seconds, tol):
    if tol <= 0:
        return ts in labeled_seconds
    for d in range(-tol, tol + 1):
        if (ts + d) in labeled_seconds:
            return True
    return False


def get_times_from_json(data):
    times_from_label = []

    for item in data['annotations']: # Extract all times from the label
        _, time_from_label = time_converter(item['gameTime'])
        times_from_label.append(time_from_label)
    return times_from_label

def get_random_time():
    # Randomly select a time, maybe in the same format as the label, i.e. 'min':'sec'
    random_min = np.random.randint(0, 45) # minutes
    random_sec = np.random.randint(0, 60) # seconds

    half_number = np.random.randint(1, 3)

    time = f"{random_min}:{random_sec}"
    mins, seconds = time.split(":")
    timestamp_seconds = int(mins)*60+int(seconds) 
    return half_number, timestamp_seconds

class BatchWriter:
    """
    Accumulates (clip, label) pairs and writes them to disk in .npz batches.
    """

    def __init__(self, save_dir: str, batch_size: int, prefix: str = "data_batch"):
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.prefix = prefix

        os.makedirs(self.save_dir, exist_ok=True)

        self.X_batch = []
        self.y_batch = []
        self.batch_idx = 0
        self.total_samples = 0

    def add(self, clip_array: np.ndarray, label: int):
        """
        Add one clip and its label to the current batch.
        If batch is full, flush it to disk.
        """
        self.X_batch.append(clip_array)
        self.y_batch.append(label)

        if len(self.X_batch) >= self.batch_size:
            self.flush()

    def flush(self):
        """
        Save current batch to disk (if non-empty) and reset buffers.
        """
        if not self.X_batch:
            return

        data = np.stack(self.X_batch)              # shape: (B, T, H, W, C)
        labels = np.array(self.y_batch)            # shape: (B,)

        filename = f"{self.prefix}_{self.batch_idx:04d}.npz"
        filepath = os.path.join(self.save_dir, filename)

        np.savez_compressed(filepath, data=data, labels=labels)

        self.total_samples += len(self.y_batch)
        print(
            f"[BatchWriter] Saved {filepath} with {len(self.y_batch)} samples "
            f"(total={self.total_samples})"
        )

        self.batch_idx += 1
        self.X_batch.clear()
        self.y_batch.clear()

def extract_non_shots(game_path: str, writer: BatchWriter):
    non_shots = []
    non_shot_events=0
    # Crawl label and extract
    with open(f'{game_path}/Labels-v2.json', "r") as file:
        data = json.load(file) # label file for that specific data

    times_from_label = get_times_from_json(data) # Get times present in label
    while non_shot_events < N_NON_SHOT_EVENTS: # Extract N_NON_SHOT_EVENTS non shot events per game
        clip = [] 

        half_number, timestamp_seconds = get_random_time() # Generate a random time
        
        # If the extracted timestamp is within 'tol' seconds of the label, don't include it.
        if within_tol(timestamp_seconds, times_from_label, tol):
            continue
        
        # non-shot event found
        non_shots.append(timestamp_seconds)
        # Generate source file name from the above video
        filename = f'{half_number}_720p.mkv'

        video_path = f'{game_path}/{filename}'

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"{video_path} does not exist")


        # Read video to get shot clip 
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Since we're reading frame-by-frame, we need the starting frame number. 
        # The starting frame number will be a certain number of frames before the shot was taken
        start_frame = int(timestamp_seconds * fps) - INPUT_NO_OF_FRAMES

        # If we want to extract 10 frames, this function will load up 10 frames from the video and stack them up into a numpy array
        # Each array will be of shape (10, (l, w, c))
        for i in range(int(INPUT_NO_OF_FRAMES)): # Add a buffer to get a few frames after the shot, just for sanity
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame+i)
            ret, frame = cap.read()
            clip.append(frame)

        # NOTE: We only know the time a shot was taken and not the exact frame, so the shot could be taken in any of the (INPUT_NO_OF_FRAMES) frames.
        # I have added a small buffer to try and collect the exact moment of a shot for as many clips as possible.

        clip_array = np.stack(clip) # Shape 32, (l, w, c)
        
        # Label = 0 for non shot
        writer.add(clip_array, 0)

        non_shot_events += 1


def crawl_dataset(root_dir: str, writer: BatchWriter):

    for league_name in os.listdir(root_dir):
        league_path = os.path.join(root_dir, league_name)
        print("\nLeague Path: ", league_path)
        if not os.path.isdir(league_path):
            continue
        
        for season in os.listdir(league_path):
            season_path = os.path.join(league_path, season)
            print("- Season Path: ", season_path)
            if not os.path.isdir(season_path):
                continue
            
            for game_name in os.listdir(season_path):
                game_path = os.path.join(season_path, game_name)
                print("-- Game Path: ", game_path)
                if not os.path.isdir(game_path):
                    continue

                if os.path.exists(game_path):
                    print(f"--- Processing: {game_path} ---")
                    # Sample, on average, 4 non-shot events per game
                    try:
                        extract_non_shots(game_path, writer)
                        print(f"--- Processed {game_path} ---")
                    except Exception as e:
                        print(f"--- Error processing {game_path}: {e} ---")
                else:
                    print(f"-- Missing files in {game_path} --") 

if __name__=='__main__':
    writer = BatchWriter(
        save_dir=save_directory,
        batch_size=BATCH_SIZE,
        prefix=BATCH_PREFIX,
    )
    crawl_dataset(ROOT_DIR, writer)

    # Flush any remaining samples that didn't fill a full batch
    writer.flush()

    print(
        f"Done. Total samples saved: {writer.total_samples}, "
        f"batches: {writer.batch_idx}"
    )

