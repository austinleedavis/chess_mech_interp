#%%
from src.train_argparse import get_args
argparse_args = get_args()

#%%
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

import src.chess_utils as cu

def tqdm(x):
    return x

print("Loading Model")
model_name = make_official()

tokenizer:PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(model_name)
model = HookedTransformer.from_pretrained(model_name)

cfg = model.cfg

model = HookedTransformer(cfg, tokenizer)


#%%
# Load data

print("Loading Dataset")
#loading dataset with 126 chars each (129,798 games)
if os.uname().nodename.startswith('ev'):
    dataset = pd.read_pickle('chess_data/lichess_train.pkl') 
else:
    dataset = pd.read_pickle('chess_data/lichess_test.pkl') 
board_seqs_int = torch.tensor(dataset['input_ids'].tolist()).long()
board_seqs_string = np.stack(dataset['fen_stack'].tolist())
print(f'Dataset loaded with {len(dataset)} samples.')

#%%

state_stack = cu.create_state_stacks(board_seqs_string[:27], cu.board_to_piece_state)

print(state_stack.shape) # [num_games, pos, row, col]
print(state_stack[0,0])

# %%
# -------------
# Parameter Placeholders
# -------------
# Note: All parameters below are overridden by the configurator.
#       They remain here to allow linting to work properly.
layer = 6
batch_size = 10
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
log_frequency = 10
checkpoint_frequency = 10
custom_board_state_fn = cu.board_to_piece_state

# %%

# Add all argparse arguments to the global context
config = vars(argparse_args)

for arg in config:
    globals()[arg] = getattr(config, arg)


config.update({"JobID":os.getenv("SLURM_JOB_ID")}) # do this after pushing global vars
run = wandb.init(
                project="chess_world", 
                config=config,
                save_code=True,
            )

if probe_name == '': probe_name = run.name #overwrite default name if it's missing

config.update({'probe_name': probe_name})

print("Training initialized with the following configuration:\n",run.config)

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok = True)

probe_filename = os.path.join(output_dir, probe_name) + ".pth"


# %%

### ------------------
# Computed values
### ------------------

min_val = int(state_stack[0,0,0].min())
max_val = int(state_stack[0,0,0].max())
options = max_val - min_val + 1
modes = state_stack.shape[0]
if num_games > len(dataset):
    print(f'INVALID NUMBER OF GAMES. Given: {num_games}. Available: {len(dataset)}. Setting num_games to {len(dataset)}')
    num_games = min(num_games,len(dataset))

state_stack_one_hot = cu.state_stack_to_one_hot(modes,rows,cols,min_val,max_val,model.cfg.device,state_stack)

print(state_stack_one_hot.shape) #torch.Size([13, 50, 125, 8, 8, 13])
for i in range(13): print(state_stack_one_hot[0,0,0,...,i])

#%%

linear_probe = torch.randn(
    modes, model.cfg.d_model, rows, cols, options, requires_grad=False, device=model.cfg.device
)/np.sqrt(model.cfg.d_model)
linear_probe.requires_grad = True
optimiser = torch.optim.AdamW([linear_probe], lr=lr, betas=(0.9, 0.99), weight_decay=wd)
batch = 0
sample = 0
for epoch in range(num_epochs):
    full_train_indices = torch.randperm(num_games)
    for i in tqdm(range(0, num_games, batch_size)):
        batch += 1
        sample += batch_size
        indices = full_train_indices[i:i+batch_size]
        games_int = board_seqs_int[indices]
        games_str = board_seqs_string[indices]
        state_stack = cu.create_state_stacks(games_str, custom_board_state_fn)
        state_stack = state_stack[:, :, pos_start:pos_end, :]
        #torch.Size([1, 5, 121, 8, 8])

        state_stack_one_hot = cu.state_stack_to_one_hot(
            modes, rows, cols, min_val, max_val, model.cfg.device, state_stack
        )[:, :, :-1, ...] # lose last index here
        # torch.Size([1, 5, 120, 8, 8, 13])
        with torch.inference_mode():
            _, cache = model.run_with_cache(games_int.cuda()[:, :-1], return_type=None)
            resid_post = cache["resid_post", layer][:, pos_start:pos_end]
        probe_out = einsum(
            "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
            resid_post,
            linear_probe,
        )
        probe_log_probs = probe_out.log_softmax(-1)
        probe_correct_log_probs = einops.reduce( 
            probe_log_probs * state_stack_one_hot,
            "modes batch pos rows cols options -> modes pos rows cols",
            "mean"
        ) * options # Multiply to correct for the mean over options
        
        loss = -probe_correct_log_probs[0, :].mean(0).sum()
        
        loss.backward() # it's important to do a single backward pass for mysterious PyTorch reasons, so we add up the losses - it's per mode and per square.
        optimiser.step()
        
        #logging
        if batch % log_frequency == 0:
            
            accuracy = (
                (probe_out[0].argmax(-1) == state_stack_one_hot[0].argmax(-1))
                .float()
                .mean()
            )
            
            logging_dict = {
                'epoch': epoch,
                'batch': batch,
                'sample': sample,
                'acc': accuracy,
                'loss': loss
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
    'epoch': epoch,
    'batch': batch,
    'sample': sample,
    'acc': accuracy,
    'loss': loss
    }
            
wandb.log(logging_dict)
wandb.log_model(probe_filename, name = probe_name)
run.finish()
# %%
# %%