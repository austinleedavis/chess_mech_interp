#%%
from transformer_lens import  HookedTransformer
import torch
from transformers import PreTrainedTokenizerFast
from mech_interp.fixTL import make_official
import pandas as pd
#%%

MODEL_NAME = make_official()

# tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
model = HookedTransformer.from_pretrained(MODEL_NAME)

#%%
test = pd.read_pickle('./chess_data/lichess_test.pkl')

#%%

set_size = 300
subset = torch.tensor(test['input_ids'][:set_size].tolist())

_, _ = model.run_with_cache(subset)

#%%
logits, cache = model.run_with_cache(torch.tensor([[1,5],[1,2]]))
logits.shape
# %%
df = pd.DataFrame({'mydat':[[1,2],[3,4],[5,6]]})
indices = [2,1,0]
df['mydat'][indices]
# %%
cache['resid_post',9].shape
# %%
