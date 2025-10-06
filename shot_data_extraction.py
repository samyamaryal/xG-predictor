from moviepy import *
import torch
import torchvision
import numpy as np
import json
import cv2

INPUT_NO_OF_FRAMES = 10

with open("data/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels-v2.json", "r") as file:
    data = json.load(file)


# All shots from the game, labels extracted from JSON 

# If a shot was taken, I want to extract the time (in seconds) from that shot, and then extract a
# frame sequence of 'n' frames leading upto that shot. 

def time_converter(time_in):
    half, time = time_in.split("-") # Half and time are split on this
    mins, seconds = time.split(":")
    timestamp_seconds = mins*60+seconds 
    return half, timestamp_seconds
# For all shots: 
ls = []
y = []


def extract_clips():
    for item in data['annotations']:
        clip = []
        if "Shot" in item['label']:

            # Get half and time in seconds
            half_number, time_in_seconds = time_converter(item['gameTime']) # time_converter returns the half that the game was played in, and the time of an event in seconds
            
            # Generate source file name from the above video
            filename = f'{half_number}_720p.mkv'


            # Read video to get shot clip 
            cap = cv2.VideoCapture(f"data/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/{filename}")
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Since we're reading frame-by-frame, we need the starting frame number. 
            start_frame = int(time_in_seconds * fps)

            # If we want to extract 10 frames, this function will load up 10 frames from the video and stack them up into a numpy array
            # Each array will be of shape (10, (l, w, c))
            for i in range(INPUT_NO_OF_FRAMES):
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame+i)
                ret, frame = cap.read()
                clip.append(frame)


            clip_array = np.stack(clip) # Shape 10, (l, w, c)
            print(clip_array.shape)
            

            # frames_list.append(frames)
            y.append(1) # Append label to the data
            ls.append(clip_array)

    # We need to stack the training data into a numpy array. However, this can be done once all the frames have been extracted.
    # train_data = np.stack(ls) 




