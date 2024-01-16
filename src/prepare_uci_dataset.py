import os
import io
import pandas as pd
from tqdm import tqdm
import chess
import chess.pgn
import argparse



LICHESS_NUM_GAMES = 16_392_151
"""
The Lichess6gb dataset has 16_392_151 games. 
"""

CHUNK_SIZE = 100_000

IN_DIR = "src/chess_data/lichess/feathers"
OUT_DIR = "src/chess_data/lichess/uci/"
CSV = "lichess_6gb.csv"
ZIP = "lichess_6gb.zip"
URL = "https://huggingface.co/datasets/adamkarvonen/chess_games/resolve/main/lichess_6gb.zip?download=true"

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
    # return [pgn.strip()[-1] == "#" for pgn in pgn_transcript]
    return pgn_transcript.strip().endswith('#')

def build_uci_txt_files():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    ###
    # pgn->uci + bos, eos tokens
    ###
    
    chunk_files = [
        os.path.join(IN_DIR, f) for f in os.listdir(IN_DIR) if f.startswith("chunk_")
    ]
    
    assert len(chunk_files) > 0
    

    for file in tqdm(chunk_files, desc = "Processing Chunk Files"):
        
        chunk_df = pd.read_feather(file)


        uci_moves = chunk_df['uci_moves']
        
        checkmate = chunk_df["transcript"].apply(had_checkmate)

        # add BOS (;) token everywhere and EOS (#) tokens when checkmate occurs
        uci_moves = [
            ";" + uci + "#" if mate else ";" + uci
            for uci, mate in zip(uci_moves, checkmate)
        ]

        chunk_df["uci_moves"] = uci_moves

        out_filename = OUT_DIR + f"{os.path.splitext(os.path.basename(file))[0]}.txt"
        
        # save uci_moves for model training
        chunk_df["uci_moves"].to_csv(out_filename, header=False, index=False)

def concat_uci_txt_files(output_file):
    if not os.path.exists(OUT_DIR):
        print('Output directory does not exist!',OUT_DIR,' ',sep='\n')
        exit(-1)

    text_files = [
        os.path.join(OUT_DIR, f) for f in os.listdir(OUT_DIR) if f.startswith("chunk_")
    ]
    
    assert len(text_files) > 0
    
    # Concatenate the contents of each file into the output file
    with open(OUT_DIR+output_file, 'w') as outfile:
        for filename in tqdm(text_files,desc="Concatenating Chunk Files"):
            with open(filename, 'r') as infile:
                # Read the content of the file and write it to the output file
                outfile.write(infile.read())
                # Optionally, you can write a newline between files
                outfile.write('\n')

if __name__ == "__main__":

    if BUILD_FILES:= False:
        build_uci_txt_files()
        
    if CONCAT_FILES:= True:
        concat_uci_txt_files('lichess_uci.txt')