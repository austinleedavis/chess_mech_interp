

# TODO This file uses old tokenization scheme. Needs to be updated

import os
import io
from typing import Tuple
import pandas as pd
import requests
from tqdm import tqdm
import zipfile
import chess
import chess.pgn
import argparse
from transformer_lens import HookedTransformer
from utils import make_official


LICHESS_NUM_GAMES = 16_392_151
"""
The Lichess6gb dataset has 16_392_151 games. 
"""

CHUNK_SIZE = 100_000

DIR = "chess_data/lichess/"
CSV = "lichess_6gb.csv"
ZIP = "lichess_6gb.zip"
URL = "https://huggingface.co/datasets/adamkarvonen/chess_games/resolve/main/lichess_6gb.zip?download=true"


def save_data_in_chunks(data_source, chunk_size: int, base_file_path: str):
    # Check if the data source is a DataFrame or a file reader
    if isinstance(data_source, pd.DataFrame):
        # If DataFrame, calculate number of chunks
        num_chunks = len(data_source) // chunk_size + (
            len(data_source) % chunk_size > 0
        )
        data_chunks = (
            data_source[i * chunk_size : (i + 1) * chunk_size]
            for i in range(num_chunks)
        )
    elif hasattr(data_source, "__iter__") and not isinstance(data_source, str):
        # If it's an iterable (like a file reader), use it directly. Assume default NUM_GAMES
        data_chunks = data_source
        num_chunks = LICHESS_NUM_GAMES // chunk_size + (
            LICHESS_NUM_GAMES % chunk_size > 0
        )
    else:
        raise ValueError(
            "data_source must be a pandas DataFrame or an iterable file reader"
        )

    # Process and save each chunk
    for i, chunk in enumerate(
        tqdm(data_chunks, total=num_chunks, desc=f"Saving chunks to {base_file_path}")
    ):
        chunk_path = f"{base_file_path}chunk_{i:03}.feather"
        chunk.to_feather(chunk_path)


def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def download_file(url: str, file_path: str):
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


def pgn2uci(pgn_transcript: str) -> str:
    return " ".join(
        [
            move.uci()
            for move in chess.pgn.read_game(
                io.StringIO(pgn_transcript)
            ).mainline_moves()
        ]
    )

def had_checkmate(pgn_transcript: str) -> bool:
    return [
        pgn.strip()[-1] == '#' for pgn in pgn_transcript
    ]

def batch_offset_mapping_to_move_index(batch_offset_mapping):
    batch_move_indices = []

    for game_offsets in batch_offset_mapping:
        move_indices = []
        current_group = []
        group_start = 0
        for i, (start, end) in enumerate(game_offsets):
            # Add the current token to the group
            current_group.append((start, end))

            # Check if the current token is the last one or if the next token starts where the current one ends
            if i == len(game_offsets) - 1 or game_offsets[i + 1][0] != end:
                # Group is complete, add to grouped_offsets
                grouped_start = current_group[0][0]
                grouped_end = current_group[-1][1]
                move_indices.extend([group_start] * len(current_group))
                current_group = []
                group_start += 1

        batch_move_indices.append(move_indices)

    return batch_move_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Processes lichess-6gb dataset. Splits it into chunks and saves dataframes with additional columns."
    )
    parser.add_argument(
        "--chunk_min", default=None, help="Minimum chunk index to process"
    )
    parser.add_argument(
        "--chunk_max", default=None, help="Maximum chunk index to process"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(
        f"\nPreparing dataset with min chunk: {args.chunk_min}, max chunk: {args.chunk_max}\n"
    )

    ###
    # Prepare dataset chunks
    ###
    if not os.path.exists(DIR):
        os.makedirs(DIR)
        
    if not os.path.exists(DIR+'uci/'):
        os.makedirs(DIR+'uci/')
        

    feather_files = [DIR + f"chunk_{i:03}.feather" for i in range(164)]
    if check_files_exist(feather_files):
        print("All feather files already exists")
    else:

        if not os.path.exists(DIR + CSV):
            if not os.path.exists(DIR + ZIP):
                download_file(URL, DIR + ZIP)
        
            unzip_file(DIR + ZIP, DIR)

        big_df = pd.read_csv(DIR + CSV, chunksize=CHUNK_SIZE)
        save_data_in_chunks(big_df, CHUNK_SIZE, DIR+'feathers/')

        del big_df

    ###
    # pgn->uci + pre-computed columns
    # Compute time: ~2 minutes/chunked file * 164 files = ~ 5 hrs
    ###
    model = HookedTransformer.from_pretrained(
        make_official(), device="cpu"
    )  # using it for encoding only
    chunk_files = [
        os.path.join(DIR, f) for f in os.listdir(DIR+'feathers/') if f.startswith("chunk_")
    ]

    chunk_files = sorted(chunk_files)[int(args.chunk_min) : int(args.chunk_max)]

    for file in tqdm(chunk_files):
        print(f'Updating {file}')
        chunk_df = pd.read_feather(file)

        uci_moves = chunk_df["transcript"].apply(pgn2uci)

        checkmate = chunk_df['transcript'].apply(had_checkmate)
        
        batch_encoding = model.tokenizer.batch_encode_plus(
            uci_moves.tolist(), return_offsets_mapping=True
        )
        token_to_move_mapping = batch_offset_mapping_to_move_index(
            batch_encoding["offset_mapping"]
        )
        
        # add EOS_token_ID whenever checkmate happens
        eos_tok_id = model.tokenizer.encode('</s>') #id=2
        input_ids = batch_encoding["input_ids"]
        input_ids = [ids + [eos_tok_id] if check else ids for ids, check in zip(input_ids, checkmate)]
        
        num_tokens = [len(row) for row in input_ids]

        chunk_df["num_tokens"] = num_tokens
        chunk_df["uci_moves"] = uci_moves
        chunk_df["tokens_int"] = input_ids
        chunk_df["token2move"] = token_to_move_mapping
        chunk_df['checkmate'] = checkmate
        chunk_df.to_feather(file)
        print(f"\nUpdated and Saved: {file}\n")
        
        # save uci_moves for model training
        chunk_df['uci_moves'].to_csv(DIR+f'uci/{file}.txt', header=False) #read with header=None
