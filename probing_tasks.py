"""
The vision for this study now is to apply DLA layer-by-layer, and perform similar probing tasks as in the LCB paper. 

The probings tasks are.  
"""

# %%
import chess
import torch
from IPython.display import display
import pandas as pd
from transformer_lens import HookedTransformer, ActivationCache
from transformers import PreTrainedTokenizerFast
from src.mech_interp.fixTL import make_official
from src.mech_interp.mappings import logitId2square, logitId2squareName
from plotly_express import imshow, histogram
from tqdm import tqdm


torch.set_grad_enabled(False)

# %%
MODEL_NAME = make_official()

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
model = HookedTransformer.from_pretrained(MODEL_NAME)
# %%


# Load Dataset
dataset = pd.read_pickle('chess_data/lichess_test.pkl') 
print(f'dataset.keys():\n{dataset.keys()}')
print(f'num_games: {len(dataset)}')
# board_seqs_int = torch.tensor(dataset['input_ids'].tolist()).long()
# board_seqs_string = np.stack(dataset['fen_stack'].tolist())


# %%

game_index, target_layer, move_index = (0,-1, 0)

start_counts = pd.Series()
end_counts = pd.Series()

for game_index in tqdm(range(len(dataset))):
    game_data = dataset.iloc[game_index]

    # get logits for all steps in the game
    _, cache = model.run_with_cache(game_data['transcript'])
    unembedded = cache['resid_post', target_layer] @ model.W_U
    logits = unembedded[0,:,logitId2square].softmax(-1).cpu()
    
    for move_index in range(len(game_data['fen_stack'])-2):
        
        game_state = game_data['fen_stack'][move_index]
        board = chess.Board(game_state)
        legal_moves = list(board.generate_legal_moves())
        
        legal_start_squares = [mv.from_square for mv in legal_moves]
        legal_end_squares = [mv.to_square for mv in legal_moves]
        print(len(legal_end_squares))

        # rank order the logits
        tok_pred_start = logits[2*move_index].topk(64).indices
        tok_pred_end = logits[2*move_index+1].topk(64).indices
        
        legal_start_tokens_count = -1
        for start_tok in tok_pred_start:
            legal_start_tokens_count += 1
            if start_tok not in legal_start_squares:
                break
        
        legal_end_tokens_count = -1
        for end_tok in tok_pred_end:
            legal_end_tokens_count += 1
            if end_tok not in legal_end_squares:
                break
        
        start_counts[game_index] = legal_start_tokens_count
        end_counts[game_index] = legal_end_tokens_count
    
    del cache
    del logits
    del unembedded
    
    if game_index == 5:
        break

    # display(board)




# %%
display(histogram(start_counts))
display(histogram(end_counts))

# %%
imshow(
    logits[0, :, logitId2square].cpu().view(-1, 8, 8).permute(1, 2, 0).flip(0),
    x="a b c d e f g h".split(),
    y=[f"{i}" for i in range(8, 0, -1)],
    aspect="equal",
    animation_frame=2,
    
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
    
)

# %%


# %%

# %%
