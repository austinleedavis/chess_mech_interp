import os
import io
import pandas as pd
import requests
from tqdm import tqdm
import zipfile
import chess
import chess.pgn


def save_data_in_chunks(data_source, chunk_size: int, base_file_path: str):
    # Check if the data source is a DataFrame or a file reader
    if isinstance(data_source, pd.DataFrame):
        # If DataFrame, calculate number of chunks
        num_chunks = len(data_source) // chunk_size + (len(data_source) % chunk_size > 0)
        data_chunks = (data_source[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks))
    elif hasattr(data_source, '__iter__') and not isinstance(data_source, str):
        # If it's an iterable (like a file reader), use it directly
        data_chunks = data_source
        num_chunks = None  # Unknown number of chunks
    else:
        raise ValueError("data_source must be a pandas DataFrame or an iterable file reader")

    # Process and save each chunk
    for i, chunk in enumerate(tqdm(data_chunks, total=num_chunks, desc=f"Saving chunks to {base_file_path}")):
        chunk_path = f"{base_file_path}chunk_{i:03}.feather"
        chunk.to_feather(chunk_path)
        
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_file(url:str, file_path:str):
    print(f"Downloading {url} to {file_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading:"
    )
    with open(file_path, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def check_files_exist(file_list):
    return all(os.path.exists(file) for file in file_list)

if __name__ == "__main__":
    CHUNK_SIZE = 100000
    
    DIR = "./chess_data/lichess/"
    CSV = "lichess_6gb.csv"
    ZIP = "lichess_6gb.zip"
    URL = (
        "https://huggingface.co/datasets/adamkarvonen/chess_games/resolve/main/lichess_6gb.zip?download=true"
    )

    # Prepare source files
    feather_files = [DIR+f'chunk_{i:03}.feather' for i in range(164)]
    if check_files_exist(feather_files):
        print('All feather files already exists')
        exit()
    else:
        if not os.path.exists(DIR+CSV):
            if not os.path.exists(DIR+ZIP):
                download_file(URL, DIR + ZIP)
            else:
                unzip_file(DIR + ZIP, DIR)
        
        big_df = pd.read_csv(DIR + CSV, chunksize=CHUNK_SIZE)
        save_data_in_chunks(big_df, CHUNK_SIZE, DIR)
        
        del big_df
        
    # Convert pgn to uci (~2 minutes/chunked file * 164 files)
    chunk_files = [os.path.join(DIR,f) for f in os.listdir(DIR) if f.startswith('chunk_')]
    
    for file in chunk_files:
        chunk_df = pd.read_feather(file)

        pgn2uci = lambda transcript : [move.uci() for move in chess.pgn.read_game(io.StringIO(transcript)).mainline_moves()]
        
        
        
        game = chess.pgn.read_game(io.StringIO(df['transcript'][0]))
        uci_moves = [move.uci() for move in game.mainline_moves()]