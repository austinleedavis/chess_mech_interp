"""Valid Moves Exploration
"""
# pylint disable=unused-import

# %% [markdown]
# # Valid Moves Exploration
# During my meeting with Gita on 4 Jan 2024, we discussed finding ways to build
# a cohesive paper out of my data analysis efforts. Although I desire to dive
# more into the DLA and probing techniques from MI, I think it's important to
# dig very quickly into concrete questions about the model's performance. To
# this end, I'm starting this set of explorations.

# ## Setup
# %% Imports
import chess
from austin.imports import *
from mech_interp.mappings import logitId2token, logitId2square
from mech_interp.utils import uci_to_board
from mech_interp.fixTL import make_official

pysvelte.Hello(name="austin")


# %%
# # Load Dataset
def data_src(file, ext):
    return f"./mi_data/{file}_dev_n100_m127{ext}"


df = pd.read_pickle(data_src("game_data", ".pkl"))
is_valid_move = torch.load(data_src("is_valid_move", ".pt"))
big_logits: torch.Tensor = torch.load(data_src("logits", ".pt"))
big_cache: ActivationCache = torch.load(data_src("cache", ".pt"))

display(df.head(2))
print("Number of games: ", len(df))
print("Tokens per game: ", len(df["tokens_int"][0]))
print("is_valid_move.shape", is_valid_move.shape)
print("big_cache['attn',0].shape: ", big_cache["attn", 0].shape)
print("big_cache['mlp_post',0].shape: ", big_cache["mlp_post", 0].shape)

model = big_cache.model
tokenizer = big_cache.model.tokenizer
print("Model and tokenizer loaded!")

# %%

# %% [markdown]
# ## Logit allocation toward valid destination squares
# In this section I want to answer the following question:
# > Of all the logits allocated by the model, how much is allocated to valid moves.
#
# # Which tokens are "valid"
# There are three types of tokens we must consider.
# - Tile tokens (6): representing the 64 squares on a board
# - Promotion tokens (8): the promotion options for black/white (treated differently)
# - Special tokens (3): These are "<s>", "</s>", and "<pad>"
# Whether we include some or all of these categories will heavily influence the analysis process. I'll do each in turn.
#
# Before I begin analysis, here's a plot showing valid moves for game 0 for tokens 0 and 1:
# %%

GAME = 0
for pos in range(2):
    print("input sequence: ", df["tokens_str"][0][:pos])
    print("valid next tokens: ", df["valid_tokens"][0][pos])
    fit = imshow(
        is_valid_move[GAME, pos, logitId2square].reshape(8, 8).flip(0),
        title=f"Valid board tiles for Game={GAME} pos={pos}",
        x="a b c d e f g h".split(),
        y=[f"{i}" for i in range(8, 0, -1)],
        xaxis="",
        yaxis="",
        color_continuous_scale="Bugn",
        range_color=[0, 1],
        aspect="equal",
    )


# %% Helper functions
true_count = is_valid_move.sum(dim=-1).clamp(min=1)
false_count = (~is_valid_move).sum(dim=-1).clamp(min=1)


def plot_lines(corrects, incorrects, title, opacity=0.1):
    combined_fig = go.Figure()
    combined_fig.update_layout(
        title=title,
        xaxis_title="Token Position",
        yaxis_title="Diffed Logit Value",
    )
    combined_fig.update_layout(template="plotly_dark")  # or 'plotly_white'

    fig = line(
        torch.sort(corrects - incorrects, dim=-1).values.cpu(),
        return_fig=True,
    )

    mean_fig = line(
        torch.sort((corrects - incorrects), dim=-1).values.mean(0).cpu(),
        return_fig=True,
        color_discrete_sequence=["white"],
    )

    for trace in fig.data:
        combined_fig.add_trace(
            go.Line(
                x=trace["x"],
                y=trace["y"],
                mode=trace["mode"],
                name=trace["name"],
                opacity=opacity,  # Set the opacity to 0.2
            )
        )

    for trace in mean_fig.data:
        combined_fig.add_trace(trace)

    combined_fig.show()


# %% [markdown]
# ### All Tokens
# Here I'm looking at logit allocation across all tokens based on position. What's
# interesting to observe is that the model allocates so much weight to tokens later
# in the input stream than it does in the earlier positions. It's as if the model makes
# more confident predictions on later tokens which is counter-intuitive. This also shows
# that initially, the model goes from approximately even allocation between valid and
# invalid logits to a ~3x allocation toward valid.

# %%

valid_logits = torch.where(is_valid_move, big_logits, 0.0)
invalid_logits = torch.where(~is_valid_move, big_logits, 0.0)

plot_lines(
    valid_logits.sum(dim=-1) / 77.0,
    invalid_logits.sum(dim=-1) / 77.0,
    "Correct-Incorrect diff by token position (mean in bold)",
)


# %% [markdown]
# I need to look at this trend, breaking up the predictions by even/odd
# position. So, that's what I'm doing below. These plots show that the
# model's allocation of logits is drastically different between selecting
# a starting tile and selecting a destination tile.

# %%
# line((valid_logits[:,1:-1:2]-invalid_logits[:,1:-1:2]).mean(0))
plot_lines(
    valid_logits[:, 0:-1:2].sum(dim=-1) / 77.0,
    invalid_logits[:, 0:-1:2].sum(dim=-1) / 77.0,
    title="Valid-Invalid diff by pos (bold mean) start tiles",
)
plot_lines(
    valid_logits[:, 1:-1:2].sum(dim=-1) / 77.0,
    invalid_logits[:, 1:-1:2].sum(dim=-1) / 77.0,
    title="Valid-Invalid diff by pos (bold mean) end tiles",
)


# %% [markdown]
# These upward trends are a bit confusing; why would logits intensify over the course of a game?
# I'm going to plot a few games showing just the correct/incorrect logits so try to understand what
# I see.

# %%

imshow(valid_logits[0], title="Valid logits, game 0")
imshow(invalid_logits[0], title="Inalid logits, game 0")


# %%

from plotly.subplots import make_subplots


special_range = torch.tensor(
    sorted(
        list(
            set(
                [tokenizer.vocab.get(i) for i in tokenizer.vocab.keys() if len(i) == 1]
                + list(range(9))
            )
        )
    )
)
special_range_str = tokenizer.convert_ids_to_tokens(special_range)

for game in [25, 37]:
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        imshow(
            valid_logits[game, :, special_range[:3]],
            return_fig=True,
            x=special_range_str[:3],
        ).data[0],
        row=1,
        col=1,
    )
    fig.add_trace(
        imshow(
            invalid_logits[game, :, special_range[3:]],
            return_fig=True,
            x=special_range_str[3:],
        ).data[0],
        row=1,
        col=2,
    )
    fig.update_layout(
        height=600,
        width=800,
        title_text=f"Special Tokens Valid vs Invalid, Game {game}",
    )
    fig.update_layout(coloraxis=dict(colorscale="RdBu", cmid=0))
    fig.show()


# %% [markdown]
# Now, I'd like to investigate allocation of logits to board tiles.

# %%

valid_tile_logits = torch.where(is_valid_move, big_logits, 0.0)[:, :, logitId2square]
invalid_tile_logits = torch.where(~is_valid_move, big_logits, 0.0)[:, :, logitId2square]

for val in [valid_tile_logits,invalid_tile_logits]:
    for start in [0,1,2,3]:
        imshow(
            val.mean(0)[start::4], 
            x=[logitId2token[i] for i in logitId2square],
            title="Mean Logit Value",
            yaxis="pos",
            xaxis="board tile")
# %%


# stacked_big_resid, labels = big_cache.decompose_resid(-1, return_labels=True)
# stacked_big_resid = big_cache.apply_ln_to_stack(stacked_big_resid, -1)
# big_decomp_logits = stacked_big_resid @ model.W_U
# print("big_decomp_logits.shape", big_decomp_logits.shape)

# %%
