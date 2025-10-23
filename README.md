# xG-from-Broadcast

This project predicts **expected goals (xG)** from soccer broadcast videos.

## How to Run

1. **Activate the virtual environment** (or create one if it doesn’t exist):
   ```bash
   python -m venv venv        # Create virtual environment
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
    ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset**

In order to download the dataset, you must sign an NDA with the data provider. The NDA can be found [here](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform).

Upon signing the NDA, you will receive a password to access the data in your email. 

Store that password as an environment variable using:
```bash
export SOCCERNET_PASSWORD=<secret_password>   # macOS/Linux
setx SOCCERNET_PASSWORD <secret_password>     # Windows
```


After storing the password using environment variables, you can download the data using: 
```bash
python data_downloader.py
```

This script will automatically download and organize the SoccerNet data inside a data/ folder. The progress is displayed as a bar in the terminal.


**NOTE** the data contains 550 broadcast videos and their corresponding labels, which will take a long time to download. You may stop the process early (Ctrl+C) to download only a subset of the data. 


For details on ethical use and dataset handling, see [**ETHICS.md**](ETHICS.md).
