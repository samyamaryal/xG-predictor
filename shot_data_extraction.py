import json
import os
import cv2
import numpy as np

INPUT_NO_OF_FRAMES = 32
ROOT_DIR = "data/SoccerNet"

save_directory = "processed_data"
BATCH_PREFIX = "shot_batch"
BATCH_SIZE = 32  # number of clips per .npz file


def time_converter(time_in):
    half, time = time_in.split("-") # Half and time are split on this
    mins, seconds = time.split(":")
    timestamp_seconds = int(mins)*60+int(seconds) 
    return half.strip(), timestamp_seconds


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


def extract_shots(game_path: str, writer: BatchWriter):
    """
    For a single game folder, read Labels-v2.json,
    extract shot clips, and push them to the BatchWriter.
    """
    labels_path = os.path.join(game_path, "Labels-v2.json")
    if not os.path.exists(labels_path):
        print(f"--- Missing Labels-v2.json in {game_path}, skipping ---")
        return

    with open(labels_path, "r") as file:
        data = json.load(file)

    for item in data.get("annotations", []):
        if "Shot" not in item.get("label", ""):
            continue

        # Get half and time in seconds
        half_number, time_in_seconds = time_converter(item["gameTime"])

        # Generate source file name from the above video
        filename = f"{half_number}_720p.mkv"
        video_path = os.path.join(game_path, filename)

        if not os.path.exists(video_path):
            print(f"--- Video file missing: {video_path}, skipping annotation ---")
            continue

        # Read video to get shot clip
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            print(f"--- Could not read FPS for {video_path}, skipping ---")
            cap.release()
            continue

        # Starting frame number: a window of INPUT_NO_OF_FRAMES before shot time
        start_frame = int(time_in_seconds * fps) - INPUT_NO_OF_FRAMES
        if start_frame < 0:
            start_frame = 0

        clip = []
        for i in range(int(INPUT_NO_OF_FRAMES)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
            ret, frame = cap.read()
            if not ret:
                # Ran out of frames; break early
                break
            clip.append(frame)

        cap.release()

        # If we didn't get enough frames, skip this example
        if len(clip) < INPUT_NO_OF_FRAMES:
            print(
                f"--- Not enough frames for shot at {item['gameTime']} in {video_path}, "
                f"got {len(clip)} ---"
            )
            continue

        clip_array = np.stack(clip)  # shape: (T, H, W, C)

        # Label = 1 for shot
        writer.add(clip_array, 1)


def crawl_dataset(root_dir: str, writer: BatchWriter):
    for league_name in os.listdir(root_dir):
        league_path = os.path.join(root_dir, league_name)
        print("\nLeague Path:", league_path)
        if not os.path.isdir(league_path):
            continue

        for season in os.listdir(league_path):
            season_path = os.path.join(league_path, season)
            print("- Season Path:", season_path)
            if not os.path.isdir(season_path):
                continue

            for game_name in os.listdir(season_path):
                game_path = os.path.join(season_path, game_name)
                print("-- Game Path:", game_path)
                if not os.path.isdir(game_path):
                    continue

                print(f"--- Processing: {game_path} ---")
                try:
                    extract_shots(game_path, writer)
                    print(f"--- Finished {game_path} ---")
                except Exception as e:
                    print(f"--- Error processing {game_path}: {e} ---")


if __name__ == "__main__":
    print("Current working directory:", os.getcwd())

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
