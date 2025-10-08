from dotenv import load_dotenv
import os

import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

load_dotenv()
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="data/SoccerNet")
print(os.getenv("SOCCERNET_PASSWORD"))

mySoccerNetDownloader.password = os.getenv("SOCCERNET_PASSWORD")

# Download video files
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])

# Download labels
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])