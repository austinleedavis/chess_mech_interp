"""
Simple script to generate the datasets I'm using. Specifically, the 
dataset includes NUM_SEQS games which encode to exactly 127 tokens. 

Why focus on games encoded to a particular length? Because it allows
me to use tensor operations more effectively.

Why 127 tokens? Because the dataset was huge, and the most common ("mode")
number of tokens across the various dataset types was 127. 127 also 
is sufficiently long (31.5) for me to ensure I have tokens from the 
early-, mid-, and late-game phases.


"""

# %%
import sys
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerFast
from mech_interp.visualizations import (
    map_token_to_move_index,
    preprocess_offset_mapping,
    get_legal_tokens,
)

from mech_interp.utils import uci_to_board
from mech_interp.chess_dataset import ChessDataImporter
from mech_interp.fixTL import make_official

# %% [markdown]
# # Parameters
DATA_SET_SOURCE = "dev"
OUTPUT_DIR = "./mi_data/"
NUM_SEQS = 100
NUM_TOKENS = 127



# %% [markdown]
# # Enable CLI if necessary
# %%

if __name__ == "__main__" and not any("jupyter" in arg for arg in sys.argv):
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some arguments.")

    # Add arguments
    parser.add_argument(
        "--data_source",
        default=DATA_SET_SOURCE,
        help="Data set source. Choose from: ['train', 'test', 'dev', 'other_eval']",
    )
    parser.add_argument("--out_dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--max_games",
        type=int,
        default=NUM_SEQS,
        help="Max number of games in the dataset",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=NUM_TOKENS,
        help="Filter games to exactly this number of tokens",
    )
    args = parser.parse_args()
    DATA_SET_SOURCE = args.data_source
    OUTPUT_DIR = args.out_dir
    NUM_SEQS = args.max_games
    NUM_TOKENS = args.num_tokens

    print(
        f"Using: data_source='{DATA_SET_SOURCE}', out_dir='{OUTPUT_DIR}', max_games='{NUM_SEQS}', num_tokens='{NUM_TOKENS}'"
    )


# %% [markdown]
# # Preparing Data

# %%
torch.set_grad_enabled(False)
FILE_SUFFIX = f"_{DATA_SET_SOURCE}_n{NUM_SEQS}_m{NUM_TOKENS}"
# %%

MODEL_NAME = make_official()
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
model = HookedTransformer.from_pretrained(MODEL_NAME, tokenizer=tokenizer)

# Load the raw game data
cdi = ChessDataImporter(DATA_SET_SOURCE)


# %% Filter dataset to games that encode to NUM_TOKENS

encoded_seqs = tokenizer(cdi.games, add_special_tokens=True)
print("encoded_seqs[0]: ", encoded_seqs[0])

filtered_game_indices = [
    i for i in range(len(cdi.games)) if len(encoded_seqs[i]) == NUM_TOKENS
][:NUM_SEQS]

df = pd.DataFrame(
    {
        "game_id": filtered_game_indices,
        "uci_moves": [cdi.games[i] for i in filtered_game_indices],
        "tokens_int": [encoded_seqs[i].ids for i in filtered_game_indices],
        "tokens_str": [encoded_seqs[i].tokens for i in filtered_game_indices],
        "offsets": [encoded_seqs[i].offsets for i in filtered_game_indices],
    }
)

df.head(2)

# %%

tokens2move = []
for c, row in tqdm(df.iterrows()):
    ppom = preprocess_offset_mapping(row["offsets"])
    token2move_temp = []
    for token in range(len(row["tokens_int"])):
        token2move_temp.append(map_token_to_move_index(row["uci_moves"], token, ppom))

    tokens2move.append(token2move_temp)

df["token2move"] = tokens2move

# %% Process valid moves
valid_tokens_list = []

for i in tqdm(range(len(df)), desc="Computing valid moves"):
    row = df.iloc[i]
    board_stack = uci_to_board(
        uci_moves=row.uci_moves,
        force=True,
        fail_silent=True,
        verbose=False,
        as_board_stack=True,
    )
    temp_valid_moves = [
        get_legal_tokens(pos, row.uci_moves, row.tokens_str, row.offsets, board_stack)
        for pos in range(len(row.tokens_str))
    ]
    

    for m in temp_valid_moves:
        m.extend(["<s>", "</s>", "<pad>"])

    valid_tokens_list.append(temp_valid_moves)

df["valid_tokens"] = valid_tokens_list

# %%

df.to_pickle(f"{OUTPUT_DIR}game_data{FILE_SUFFIX}.pkl")

# %% [markdown]
# # Preparing Tensorized data
# ## Valid Moves
# %%

is_valid_move = torch.zeros(NUM_SEQS, NUM_TOKENS, 77, dtype=torch.bool, device="cpu")

for i in tqdm(range(NUM_SEQS), desc="Tensorizing valid moves"):
    for j in range(NUM_TOKENS):
        valid_moves = valid_tokens_list[i][j]
        valid_moves = tokenizer.convert_tokens_to_ids(valid_moves)
        is_valid_move[i, j, valid_moves] = True

temp = is_valid_move[:].tolist()
is_valid_move = is_valid_move.cuda()

torch.save(
    is_valid_move,
    f"{OUTPUT_DIR}is_valid_move{FILE_SUFFIX}.pt",
)


# %% [markdown]
# ## Big Logits and Cache
# %%
print("Running model with cache. This may take a while.")
BIG_LOGITS, BIG_CACHE = model.run_with_cache(torch.tensor(df["tokens_int"]))

print("saving cache and logits tensors to disk")
torch.save(
    BIG_CACHE,
    f"{OUTPUT_DIR}cache{FILE_SUFFIX}.pt",
)
torch.save(
    BIG_LOGITS,
    f"{OUTPUT_DIR}logits{FILE_SUFFIX}.pt",
)
print("finished!")

# %%
