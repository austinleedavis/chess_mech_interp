#%%
from src.train_argparse import get_args
config = get_args()

import os
import chess
import einops
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from fancy_einsum import einsum
import wandb
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerFast
from src.mech_interp.fixTL import make_official

def tqdm(x):
    return x

print("Loading Model")
model_name = make_official()

tokenizer:PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(model_name)
model = HookedTransformer.from_pretrained(model_name)
cfg = model.cfg


#%%
# Load data

print("Loading Dataset")
#loading dataset with 126 chars each (129,798 games)
dataset = pd.read_pickle('chess_data/lichess_train.pkl') 
board_seqs_int = torch.tensor(dataset['input_ids'].tolist()).long()
board_seqs_string = np.stack(dataset['fen_stack'].tolist())

#%%

def _bit_mask_to_bit_list(value: int):
    """
    Converts an integer value into a list of its binary representation.

    Args:
        value (int): The integer value to convert.

    Returns:
        list: A list of 64 elements representing the binary representation of the value.
    """
    bits = [(value >> i) & 1 for i in range(63, -1, -1)]
    return bits

def _board_to_color_state(board: chess.Board):
    state = np.zeros(64)
    color_parity = -1
    for color_mask in board.occupied_co:
        color_parity *= -1
        state[
            np.array(_bit_mask_to_bit_list(color_mask)) == 1
        ] = color_parity
    return state.reshape(8, 8)

def _board_to_piece_and_color_state(board: chess.Board):
    state = np.zeros(64)
    color_parity = -1
    for color_mask in board.occupied_co:
        color_parity *= -1
        for val, piece_type_mask in enumerate(
            [
                board.pawns,
                board.knights,
                board.bishops,
                board.rooks,
                board.kings,
                board.queens,
            ]
        ):
            state[
                np.array(_bit_mask_to_bit_list(piece_type_mask & color_mask)) == 1
            ] = color_parity * (val + 1)
    return state.reshape(8, 8)

def seq_to_state_stack(game_fen_stack):
    state_stack = np.stack([_board_to_color_state(chess.Board(move_state)) for move_state in game_fen_stack])
    # need repeat_interleave because moves require two tokens.
    # and the state of the board does not change in between these tokens
    state_stack_repeated = np.repeat(state_stack,2,axis=0)
    return state_stack_repeated[:-1] # copied one extra token's worth of state. drop it
    

# state_stack = torch.tensor(
#     np.stack([seq_to_state_stack(seq) for seq in board_seqs_string[:50]]) # NOTE: original used strings here
# )
# print(state_stack.shape) # [num_games, pos, row, col]

# %%
# -------------
# Parameter Placeholders
# -------------
# Note: All parameters below are overridden by the configurator.
#       They remain here to allow linting to work properly.
layer = 6
batch_size = 30
lr = 1e-4
wd = 0.01
pos_start = 5
pos_end = model.cfg.n_ctx - 5
length = pos_end - pos_start
rows = 8
cols = 8
num_epochs = 2
num_games = 100000
probe_name = "main_linear_probe"
output_dir = "linear_probes/"
log_frequency = 1
checkpoint_frequency = 1
# The first mode is blank or not, the second mode is next or prev GIVEN that it is not blank
modes = 3 # blank vs color (mode)
options = 3 # the three options

# %%

# Add all argparse arguments to the global context
for arg in vars(config):
    globals()[arg] = getattr(config, arg)

# %%
run = wandb.init(
                project="chess_world", 
                config=config,
                notes = f'SLURM Job ID:{os.getenv("SLURM_JOB_ID")}',
                save_code=True,
            )

if probe_name == '': probe_name = run.name #overwrite default name if it's missing

print("Training initialized with the following configuration:\n",run.config)


if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok = True)

probe_filename = os.path.join(output_dir, probe_name) + ".pth"


# %%

def state_stack_to_one_hot(state_stack):
    one_hot = torch.zeros(
        modes, # blank vs color (mode)
        state_stack.shape[0], # num games
        state_stack.shape[1], # num moves
        rows, # rows
        cols, # cols
        options, # the two options
        device=state_stack.device,
        dtype=torch.int,
    )
    one_hot[:, ..., 0] = state_stack == 0
    one_hot[:, ..., 1] = state_stack == -1
    one_hot[:, ..., 2] = state_stack == 1
    
    return one_hot

# state_stack_one_hot = state_stack_to_one_hot(state_stack)
# print(state_stack_one_hot.shape)
# print((state_stack_one_hot[:, 0, 17, 4:9, 2:5]))
# print((state_stack[0, 17, 4:9, 2:5]))

#%%


linear_probe = torch.randn(
    modes, model.cfg.d_model, rows, cols, options, requires_grad=False, device="cuda"
)/np.sqrt(model.cfg.d_model)
linear_probe.requires_grad = True
optimiser = torch.optim.AdamW([linear_probe], lr=lr, betas=(0.9, 0.99), weight_decay=wd)

for epoch in range(num_epochs):
    full_train_indices = torch.randperm(num_games)
    batch = 0
    for i in tqdm(range(0, num_games, batch_size)):
        batch += 1
        indices = full_train_indices[i:i+batch_size]
        games_int = board_seqs_int[indices]
        games_str = board_seqs_string[indices]
        state_stack = torch.stack(
            [torch.tensor(seq_to_state_stack(games_str[j])) for j in range(min(batch_size,len(games_str)))] #use min in case batches don't evenly divide num_games
        )
        state_stack = state_stack[:, pos_start:pos_end, :, :]

        state_stack_one_hot = state_stack_to_one_hot(state_stack).cuda()
        with torch.inference_mode():
            _, cache = model.run_with_cache(games_int.cuda()[:, :-1], return_type=None)
            resid_post = cache["resid_post", layer][:, pos_start:pos_end]
        probe_out = einsum(
            "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
            resid_post,
            linear_probe,
        )
        # print(probe_out.shape)

        acc_blank = (probe_out[0].argmax(-1) == state_stack_one_hot[0].argmax(-1)).float().mean()
        acc_color = ((probe_out[1].argmax(-1) == state_stack_one_hot[1].argmax(-1)) * state_stack_one_hot[1].sum(-1)).float().sum()/(state_stack_one_hot[1]).float().sum()

        probe_log_probs = probe_out.log_softmax(-1)
        probe_correct_log_probs = einops.reduce(
            probe_log_probs * state_stack_one_hot,
            "modes batch pos rows cols options -> modes pos rows cols",
            "mean"
        ) * options # Multiply to correct for the mean over options
        loss_even = -probe_correct_log_probs[0, 0::2].mean(0).sum() # note that "even" means odd in the game framing, since we offset by 5 moves lol
        loss_odd = -probe_correct_log_probs[1, 1::2].mean(0).sum()
        loss_all = -probe_correct_log_probs[2, :].mean(0).sum()
        
        loss = loss_even + loss_odd + loss_all
        
        
        loss.backward() # it's important to do a single backward pass for mysterious PyTorch reasons, so we add up the losses - it's per mode and per square.
        optimiser.step()
        
        #logging
        if batch % log_frequency == 0:
            logging_dict = {
            'epoch':epoch,
            
            'sample':i,
            'acc_blank':acc_blank,
            'acc_color':acc_color,
            'loss':loss
            }
            
            wandb.log(logging_dict)
            print(logging_dict)
        
        # checkpoint
        if batch % checkpoint_frequency == 0:
            print(f'Saving checkpoint to "{probe_filename}"')
            torch.save(linear_probe, probe_filename)
        
        optimiser.zero_grad()
        
torch.save(linear_probe, probe_filename)

logging_dict = {
            'epoch':epoch,
            
            'sample':i,
            'acc_blank':acc_blank,
            'acc_color':acc_color,
            'loss':loss
            }
            
wandb.log(logging_dict)
wandb.log_model(probe_filename, name = probe_name)
run.finish()
# %%
# %%