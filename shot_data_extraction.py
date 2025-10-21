from moviepy import *
import torch
import torchvision
import numpy as np
import json
import cv2
import os

INPUT_NO_OF_FRAMES = 30

# All shots from the game, labels extracted from JSON 

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


def extract_shots(game_path):
    with open(f'{game_path}/Labels-v2.json', "r") as file:
        data = json.load(file)

    for item in data['annotations']:
        clip = []
        if "Shot" in item['label']:

            # Get half and time in seconds
            half_number, time_in_seconds = time_converter(item['gameTime']) # time_converter returns the half that the game was played in, and the time of an event in seconds
            # print("Half number and time extracted")

            # Generate source file name from the above video
            filename = f'{half_number}_720p.mkv'

            video_path = f'{game_path}/{filename}'
            # print("Video path", video_path)

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"{video_path} does not exist")

            # Read video to get shot clip 
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Since we're reading frame-by-frame, we need the starting frame number. 
            # The starting frame number will be a certain number of frames before the shot was taken
            start_frame = int(time_in_seconds * fps) - INPUT_NO_OF_FRAMES

            # If we want to extract 10 frames, this function will load up 10 frames from the video and stack them up into a numpy array
            # Each array will be of shape (10, (l, w, c))
            for i in range(int(INPUT_NO_OF_FRAMES*1.5)): # Add a buffer to get a few frames after the shot, just for sanity
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame+i)
                ret, frame = cap.read()
                clip.append(frame)

            # NOTE: We only know the time a shot was taken and not the exact frame, so the shot could be taken in any of the (INPUT_NO_OF_FRAMES) frames.
            # I have added a small buffer to try and collect the exact moment of a shot for as many clips as possible.

            clip_array = np.stack(clip) # Shape 10, (l, w, c)
            

            # frames_list.append(frames)
            y.append(1) # Append label to the data
            X.append(clip_array)


def crawl_dataset(root_dir):

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
                    try:
                        extract_shots(game_path)
                        print(f"--- Processed {game_path} ---")
                    except Exception as e:
                        print(f"--- Error processing {game_path}: {e} ---")
                else:
                    print(f"-- Missing files in {game_path} --") 

def get_data():
    print("Current working directory:", os.getcwd())

    crawl_dataset('data/SoccerNet') # the data is placed inside data/SoccerNet directory

    # We need to stack the training data into a numpy array. However, this can be done once all the frames have been extracted.
    train_data = np.stack(X) 
    test_data = np.stack(y)
    return train_data, test_data

def save_data(data, labels):
    np.savez_compressed("processed_data/data.npz", data=data, labels=labels)
    print("Data saved to processed_data/")


if __name__=='__main__':
    train_data, test_data = get_data()
    save_data(train_data, test_data)

