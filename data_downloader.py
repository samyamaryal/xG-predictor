from dotenv import load_dotenv
import os

import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

load_dotenv()
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="data/SoccerNet")
os.getenv("SOCCERNET_PASSWORD")

mySoccerNetDownloader.password = os.getenv("SOCCERNET_PASSWORD")

# Download video files and labels
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv", "Labels-v2.json"], split=["train","valid","test"])