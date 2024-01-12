from typing import Callable, List, Literal, Tuple, Optional, Union
import chess
from torch import Tensor
from transformers import PreTrainedTokenizerFast
from transformer_lens import ActivationCache
from .utils import uci_to_board
from austin_plotly import to_numpy
import plotly.graph_objects as go


def get_legal_tokens(pos:int, 
                     uci_moves:str, 
                     tokens_with_specials, 
                     offset_mapping, 
                     board_stack: List[chess.Board],
                     return_board=False):
    """
    This function maps a token position to the corresponding legal chess board states based on UCI (Universal Chess Interface) move notation. It considers the phase of the move (start tile, end tile, or promotion) and returns all possible tokens that can be added to the token sequence to form a legal move.

    Args:
        pos (int): The position of the token in the token sequence.
        uci_moves (str): A string containing the UCI moves.
        tokens_with_specials: A sequence of tokens which MUST INCLUDE special tokens.
        offset_mapping: A mapping that associates tokens with their positions in the UCI string.
        board_stack (List[chess.Board]): A stack of chess board states.
        return_board (bool, optional): If True, returns the current board state along with legal tokens. Defaults to False.

    Returns:
        list: A sorted list of legal tokens based on the current board state and the phase of the move.
        chess.Board (optional): The current board state, returned if return_board is True.

    Example:
        legal_moves, board = get_legal_tokens(pos, uci_moves, tokens, encoded['offset_mapping'], return_board=True)

    Note:
        - The function uses the 'map_token_to_move_index', 'map_token_to_move_offsets', and 'determine_move_phase' helper functions to map tokens to their respective move phases and legal moves.
        - Special handling is done for the special ("<s>") token.
        - `tokens_with_specials` must include special tokens
    """
    
    assert(tokens_with_specials[0] == '<s>')
    
    mv_index = map_token_to_move_index(uci_moves,pos+1,offset_mapping=offset_mapping,zero_index=True)
    board: chess.Board = board_stack[mv_index]
    token_str = tokens_with_specials[pos]
    start, end = map_token_to_move_offsets(pos+1,offset_mapping=offset_mapping)
    phase = determine_move_phase(uci_moves[start:end],token_str)
    
    all_legal_moves = board.generate_legal_moves()
    
    # default case when token_str is special (<s>) token
    legal_tokens = []
    
    if phase == 'from':
        legal_tokens = [move.uci()[2:4] for move in all_legal_moves if move.uci()[:2]==token_str]
    elif phase == 'to':
        legal_tokens = [move.uci()[:2] for move in all_legal_moves if move.uci()[2:4]==token_str]
    elif phase == 'promote':
        legal_tokens = [move.uci()[:2] for move in all_legal_moves if move.uci()[-1]==token_str]
    else:
        legal_tokens = [move.uci()[:2] for move in all_legal_moves]
    
    legal_tokens=sorted(list(set(legal_tokens)))
    
    if return_board:
        return legal_tokens, board
    else:
        return legal_tokens

    
def get_game_prefix_up_to_token_idx(
    token_idx: int, 
    offset_mapping: list, 
    uci_moves: str) -> str:
    """
    Retrieves a substring of the original UCI moves string up to the position corresponding to a specified token index.

    This function calculates the starting position of the token at the given index in the original UCI moves string using the offset mapping. It then returns the substring of the UCI moves string up to (but not including) this position.

    Parameters:
    - token_idx (int): The index of the token in the encoded representation for which the prefix is to be retrieved.
    - offset_mapping (list): A list of tuples where each tuple contains the start and end character positions of each token in the original string.
    - uci_moves (str): The original string of UCI moves.

    Returns:
    - str: A substring of the original UCI moves string up to the start position of the token at the given index.

    Raises:
    - ValueError: If the token index is out of the range of encoded tokens.

    Example:
    >>> enc_moves = tokenizer("e2e4 c7c5", return_offsets_mapping=True)
    >>> enc_moves
    [(0, 4), (5, 9), ...]
    >>> get_game_prefix_up_to_token_idx(5, enc_moves, "e2e4 c7c5")
    "e2e4 "

    Note:
    This function assumes that `offset_mapping` correctly maps the tokens in the encoded representation to their positions in the original UCI moves string.
    """
    
    if token_idx > len(offset_mapping):
        raise ValueError("Index is out of range of encoded tokens")
    
    start_pos = offset_mapping[token_idx][0] if token_idx < len(offset_mapping) else len(uci_moves)
    
    return uci_moves[:start_pos]

def preprocess_offset_mapping(offset_mapping):
    """
    Processes the offset mapping to group tokens into contiguous parts of the original string.

    This function takes the offset mapping of tokens and groups them based on their contiguity in the original string.
    Each group represents a contiguous part of the string, such as a complete chess move. The function extends the 
    start and end positions of each group for the number of tokens in that group, thus aligning the new grouped 
    offset mapping with the original token sequence.

    Parameters:
    - offset_mapping (list): A list of tuples where each tuple contains the start and end character positions 
      of each token in the original string.

    Returns:
    - list: A list of tuples containing the start and end positions of grouped tokens. Each tuple corresponds 
      to a contiguous part of the original string.

    Example usage:
    ```
    >>> original_offsets = tokenizer(uci_moves, return_offsets_mapping=True)
    >>> print(original_offsets)
    [(0, 2), (2, 4), (5, 7), (7, 9), ...]
    >>> grouped_offsets = preprocess_offset_mapping(original_offsets)
    >>> print(grouped_offsets)
    ```
    
    This will output a list of tuples with grouped start and end positions for each contiguous part of the string.
    """
    grouped_offsets = []
    current_group = []

    for i, (start, end) in enumerate(offset_mapping):
        # Add the current token to the group
        current_group.append((start, end))

        # Check if the current token is the last one or if the next token starts where the current one ends
        if i == len(offset_mapping) - 1 or offset_mapping[i+1][0] != end:
            # Group is complete, add to grouped_offsets
            grouped_start = current_group[0][0]
            grouped_end = current_group[-1][1]
            grouped_offsets.extend([(grouped_start, grouped_end)] * len(current_group))
            current_group = []

    return grouped_offsets
    
def map_token_to_move_offsets(
    token_idx: int,
    preprocessed_offsets: List[Tuple] = None,
    offset_mapping: List[Tuple] = None) -> Tuple[int,int]:
    """
    Given a token index, returns the start and end indices of the current move in the original string.

    This function can work with either a raw offset mapping or a preprocessed offset mapping. If a raw offset 
    mapping is provided, it preprocesses it to group tokens into contiguous parts of the original string, 
    each corresponding to a part of a chess move.

    Parameters:
    - token_idx (int): Index of the token whose move offsets are to be found.
    - offset_mapping (list, optional): A list of tuples where each tuple contains the start and end character 
      positions of each token in the original string. This is used if a preprocessed offset mapping is not provided.
    - preprocessed_offsets (list, optional): A list of tuples containing preprocessed start and end positions for 
      each token. If provided, the function uses this instead of offset_mapping.

    Returns:
    - tuple: A tuple containing the start and end indices of the current move associated with the given token index.

    Raises:
    - ValueError: If the token index is out of range of encoded tokens.

    Example usage:
    ```
        >>> uci_moves = "d2d4 d7d5 c2c4 d5c4"
        >>> enc_moves = tokenizer(uci_moves,return_offset_mapping=True)
        >>> for i in range(8):
        ...     start, end = map_token_to_move_offsets(i, offset_mapping=enc_moves['offset_mapping'])
        ...     print(f'|{uci_moves[:start]} >>{uci_moves[start:end]}<< {uci_moves[end:19]}|')
        
            | >>d2d4<<  d7d5 c2c4 d5c4|
            | >>d2d4<<  d7d5 c2c4 d5c4|
            | >>d2d4<<  d7d5 c2c4 d5c4|
            |d2d4  >>d7d5<<  c2c4 d5c4|
            |d2d4  >>d7d5<<  c2c4 d5c4|
            |d2d4 d7d5  >>c2c4<<  d5c4|
            |d2d4 d7d5  >>c2c4<<  d5c4|
            |d2d4 d7d5 c2c4  >>d5c4<< |
    ```
    
    This will output the segmented parts of the UCI moves string, highlighting the current move for each token index.
    """
    
    if offset_mapping is not None:
        preprocessed_offsets = preprocess_offset_mapping(offset_mapping)
        
    if token_idx > len(preprocessed_offsets):
        raise ValueError("Index is out of range of encoded tokens")
    
    if token_idx == len(preprocessed_offsets):
        token_idx-=1
    
    return  preprocessed_offsets[token_idx]

def map_token_to_move_index(
    uci_moves: str,
    token_idx: int,
    preprocessed_offsets: List[Tuple] = None,
    offset_mapping: List[Tuple] = None,
    zero_index = False) -> int:
    """
    Maps a token index to its corresponding move index in a sequence of UCI moves.

    This function takes a token index from a tokenized representation of a chess game and maps it to the
    corresponding move index in the game's sequence of Universal Chess Interface (UCI) moves. It utilizes the
    `map_token_to_move_offsets` function to determine the starting position of the move in the UCI move string 
    associated with the given token index. The move index is then calculated based on this position.

    Parameters:
    - uci_moves (str): A string containing the sequence of UCI moves of a chess game.
    - token_idx (int): The index of the token in the tokenized representation of the chess game.
    - preprocessed_offsets (List[Tuple], optional): A list of tuples containing preprocessed offsets, used to
      aid in mapping tokens to moves. Each tuple typically represents a range of characters in the original text.
      Default is None.
    - offset_mapping (List[Tuple], optional): A list of tuples representing the mapping of tokens to character
      offsets in the original text. Default is None.

    Returns:
    - int: The index of the move in the UCI moves string that corresponds to the given token index. 

    Note:
    - The function assumes that the UCI moves are space-separated.
    - The token index and move index are both 0-based, i.e., the first token or move is indexed as 0.

    Example:
    ```
    uci_moves = "e2e4 e7e5 g1f3"
    token_idx = 5
    enc_moves = tokenizer(uci_moves,return_offset_mapping=True)
    move_index = map_token_to_move_index(uci_moves, token_idx, offset_mapping = enc_moves['offset_mapping'])
    # move_index will be the index of the move corresponding to the token at index 5
    ```
    """
    start_pos, _ = map_token_to_move_offsets(
        token_idx,
        preprocessed_offsets=preprocessed_offsets,
        offset_mapping=offset_mapping)
    
    return len(uci_moves[:start_pos].split(' ')) + (-1 if zero_index else 0)

from typing import Any
def get_board_states(board: chess.Board, map_function: Optional[Callable]=lambda x: x) -> List[Any]:
    """
    Generates a list of board states for a given chess game.

    This function iterates through the move stack of a provided chess board, 
    applying each move to a temporary board to capture its state after each move. 
    A mapping function can be applied to each board state before adding it to the list. 
    By default, the mapping function is the identity function, returning the board state as is.

    Parameters:
    - board (chess.Board): A chess.Board object representing the chess game to analyze.
    - map_function (Optional[Callable], default=lambda x: x): An optional function to apply to each board state.
      This function should take a chess.Board object as input and return a modified representation of the board.
      If not specified, the identity function is used, and the board states are returned unmodified.

    Returns:
    - List[Any]
    - List[str]: A list of board states, each transformed by the map_function. The list includes the initial state of the board
      (before any moves are made) as well as the state after each move in the game.

    Example:
    ```
    import chess

    # Create a chess board and apply some moves
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")

    # Get board states with a custom mapping function
    board_states = get_board_states(board, map_function=lambda b: str(b))

    # Each element in board_states is now a string representation of the board at each move
    ```
    """
    temp_board = chess.Board()
    board_states = [map_function(chess.Board())]
    
    for move in board.move_stack:
        temp_board.push(move)
        board_states.append(map_function(temp_board.copy()))
    
    return board_states

def get_hover_text(i: int, j: int, 
        z_data: Tensor,
        token_idx: List[int], 
        game_board_states: List[str], 
        uci_moves: str, 
        preprocessed_offsets: List[Tuple], 
        tokenizer: PreTrainedTokenizerFast
        ) -> str:
    
    imove_idx = map_token_to_move_index(uci_moves, i, preprocessed_offsets)
    jmove_idx = map_token_to_move_index(uci_moves, j, preprocessed_offsets)
    
    uci_moves_split = uci_moves.split(' ')
    #subtract 1 to see move which is being considered, not the move that results from this move 
    imove = uci_moves_split[imove_idx-1]
    jmove = uci_moves_split[jmove_idx-1]
    
    iboard = game_board_states[imove_idx] 
    jboard = game_board_states[jmove_idx]
    itokstr = tokenizer.decode(token_idx[i])
    jtokstr = tokenizer.decode(token_idx[j])
    sep = '<br> ↓ ATTENDS TO ↓<br>'
    return f'logit: {z_data[i][j]:.6f}<br>i:{i} mv#:{imove_idx} s:{itokstr}<br>{"W" if imove_idx%2==1 else "B"} to move, uci:{imove}<br>{iboard}{sep}j:{j} mv:{jmove_idx} s:{jtokstr}<br>{"W" if jmove_idx%2==1 else "B"} to move,uci:{jmove}<br>{jboard}'

def show_activations_from_cache(
    tokenizer: PreTrainedTokenizerFast,
    uci_moves: str,
    cache: ActivationCache,
    batch: int,
    layer: int,
    head: int,
    component: str = 'attn',
    return_fig: bool = False,
    pre_indexed_cache = False
    ):
    """
    
    Usage: 
    #visualizing layer 0, batch 0, head 0
    >>> data = cache['attn', 0][0, 0, :, :] 
    """
    
    batch_encoding = tokenizer(uci_moves,return_offsets_mapping=True)
    token_ids = batch_encoding['input_ids']
    offset_mapping=batch_encoding['offset_mapping']
    preprocessed_offsets = preprocess_offset_mapping(offset_mapping)
    game_board = uci_to_board(uci_moves)
    mapper = lambda x: x.unicode(empty_square = ' ').replace('\n','<br>')
    game_board_states = get_board_states(game_board, map_function=mapper)
    
    if not pre_indexed_cache:
        cache = cache[component,layer]
    
    z_data = to_numpy(cache[batch,head,:,:])
    # Create the figure
    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=[f'{idx}' for idx, tok in enumerate(token_ids)],
        y=[f'{idx}' for idx, tok in enumerate(token_ids)],
        hoverongaps=False,
        colorscale="Blues",
    ))
    
    hovertext = [[
        get_hover_text(i,j,z_data,token_ids,game_board_states,uci_moves,preprocessed_offsets,tokenizer) 
        for j in range(len(token_ids))] 
                 for i in range(len(token_ids))]

    fig.data[0].hovertext = hovertext
    fig.data[0].hoverinfo = 'text'
    
    tick_text = [f'{tokenizer.decode(tok)}({tok})' for i,tok in enumerate(token_ids)]
    
    fig.update_xaxes(
        tickvals=[i for i in range(len(token_ids))],
        ticktext=tick_text,
    )
    fig.update_yaxes(
        tickvals=[i for i in range(len(token_ids))],
        ticktext=tick_text,
    )

    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Monospace"
        ),
        xaxis_title="Source Token",
        yaxis_title="Destination Token",
        title={
            'text': f'{component} Batch:{batch} Layer:{layer} Head:{head}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
    )
    fig['layout']['yaxis']['autorange'] = "reversed"

    if return_fig:
        return fig
    else:
        fig.show()
        
from plotly.subplots import make_subplots

def show_activations_from_cache_animated(
    tokenizer: PreTrainedTokenizerFast,
    uci_moves: str,
    cache: Union[ActivationCache, Tensor],
    batch: int,
    layer: int,
    n_heads: int,  # Adjusted to accept the total number of heads
    component: str = 'attn',
    return_fig: bool = False,
    pre_indexed_cache = False,
):
    batch_encoding = tokenizer(uci_moves, return_offsets_mapping=True)
    token_ids = batch_encoding['input_ids']
    offset_mapping = batch_encoding['offset_mapping']
    preprocessed_offsets = preprocess_offset_mapping(offset_mapping)
    game_board = uci_to_board(uci_moves)
    mapper = lambda x: x.unicode(empty_square=' ').replace('\n', '<br>')
    game_board_states = get_board_states(game_board, map_function=mapper)

    # Initialize the figure with subplots (required for animations)
    fig = make_subplots()

    # Create a list to hold the frames
    frames = []

    # Add a frame for each head
    for head in range(n_heads):
        if pre_indexed_cache:
            indexed_cache = cache
        else:
            indexed_cache = cache[component,layer]
        z_data = to_numpy(indexed_cache[batch,head,:,:])
        hovertext = [
            [
                get_hover_text(i, j, z_data, token_ids, game_board_states, uci_moves, preprocessed_offsets, tokenizer)
                for j in range(len(token_ids))
            ]
            for i in range(len(token_ids))
        ]

        frame = go.Frame(
            data=[
                go.Heatmap(
                    z=z_data,
                    x=[f'{idx}' for idx, tok in enumerate(token_ids)],
                    y=[f'{idx}' for idx, tok in enumerate(token_ids)],
                    hoverongaps=False,
                    colorscale="Blues",
                    hovertext=hovertext,
                    hoverinfo='text'
                )
            ],
            name=str(head)
        )
        
        
        frames.append(frame)

    # Assign the list of frames to fig.frames
    fig.frames = frames

    # Make the first frame
    fig.add_trace(frames[0].data[0])

    # Construct the steps for the slider
    steps = []
    for head in range(n_heads):
        step = dict(
            method='animate',
            label=str(head),
            args=[
                [str(head)],
                {
                    'frame': {'duration': 300, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 300}
                }
            ]
        )
        steps.append(step)

    # Define the slider
    sliders = [{
        'steps': steps,
        'transition': {'duration': 300},
        'x': 0.1,
        'len': 0.9,
        'currentvalue': {'visible': True, 'prefix': 'Head: '}
    }]

    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Monospace"
        ),
        xaxis_title="Source Token",
        yaxis_title="Destination Token",
        title={
            'text': f'{component} Batch:{batch} Layer:{layer}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        sliders=sliders
    )
    
    # tick_text = [f'{tokenizer.decode(tok)}({tok})' for i,tok in enumerate(token_ids)]
    
    # fig.update_xaxes(
    #     # tickvals=[i for i in range(len(token_ids))],
    #     tickmode='array',
    #     ticktext=tick_text,
    # )
    # fig.update_yaxes(
    #     # tickvals=[i for i in range(len(token_ids))],
    #     tickmode='array',
    #     ticktext=tick_text,
    # )
    fig['layout']['yaxis']['autorange'] = "reversed"

    if return_fig:
        return fig
    else:
        fig.show()
        
        
        
########################## BOARD LOG PROBS

import torch
from austin_plotly import imshow
from .utils import logitId2square

idx2sq = [56, 57, 58, 59, 60, 61, 62, 63, 48, 49, 50, 51, 52, 53, 54, 55, 40, 41,
        42, 43, 44, 45, 46, 47, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27,
        28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23,  8,  9, 10, 11, 12, 13,
        14, 15,  0,  1,  2,  3,  4,  5,  6,  7]

def determine_move_phase(move:str, token:str) -> Literal['from','to','promote','special']:
    """
    For a given move string, returns the phase of the token string. Excellent when used in conjunction 
    with :func:`map_token_to_move_offsets()` to get the move start and end indices.
    
    Usage:
    >>> determine_move_phase('d2d4', 'd2')
    'from'
    >>> determine_move_phase('d2d4', 'd4')
    'to'
    >>> determine_move_phase('d7d8Q', 'Q')
    'promote'
    >>> determine_move_phase('<s>d2d4','<s>')
    'special'
    """
    if move[:2] == token:
        return 'from'
    if move[2:4] == token:
        return 'to'
    if len(token) == 1:
        return 'promote'
    return 'special'


from transformers import BatchEncoding
def tok2move(token_idx:int, uci_moves_split:List[str], enc_tokens:BatchEncoding):
    prefix = tok2gameprefix(token_idx,uci_moves_split,enc_tokens)
    move = prefix[-1]
    return move

def tok2gameprefix(token_idx:int, uci_moves_split:List[str], enc_tokens:BatchEncoding):
    orig_tok_charspan = enc_tokens.token_to_chars(token_idx)
    game_prefix = uci_moves_split[:orig_tok_charspan.end]
    return game_prefix

import numpy as np

def plot_board_log_probs(uci_moves: str, 
    tokenizer: PreTrainedTokenizerFast,
    logits: torch.Tensor, 
    return_fig=False, 
    slice_tokens:slice=slice(None),
    color_continuous_scale="Blues",
    ):
    
    logits = logits.squeeze(batch_dim:=0)
    encoding = tokenizer.encode_plus(uci_moves,return_offsets_mapping=True)
    token_ids = encoding['input_ids'][slice_tokens]
    token_strs = [tokenizer.decode(id) for id in token_ids]
    preprocessed_offsets = preprocess_offset_mapping(encoding['offset_mapping'][slice_tokens])
    board_stack: List[chess.Board] = uci_to_board(
            uci_moves,
            force=True,
            as_board_stack=True)
    
    assert len(token_ids) == len(logits), f"len(token_ids): {len(token_ids)} =/= len(logits): {len(logits)}"
    
    log_probs = logits.log_softmax(dim=-1)
    log_probs_template = torch.zeros((len(token_ids), 64)).cuda() - 100
    log_probs_template = log_probs[:,logitId2square]
    log_probs_template = log_probs_template.reshape(-1, 8, 8).flip(-2)
    
    move_id = lambda idx: map_token_to_move_index(
                        uci_moves,
                        idx,
                        preprocessed_offsets=preprocessed_offsets,
                        zero_index=True
                    )
    
    fig = imshow(
        log_probs_template,
        zmin=-6.0,
        zmax=0.0,
        aspect="equal",
        x=["a", "b", "c", "d", "e", "f", "g", "h"],
        y=list(reversed(["1", "2", "3", "4", "5", "6", "7", "8"])),
        color_continuous_scale=color_continuous_scale,
        animation_frame=0,
        animation_index=[
            f'Index: {i} Token: "{token_strs[i]}" ({token_ids[i]}) ' + 
            f'Move: {uci_moves.split(" ")[move_id(i)]} ({move_id(i)})'
                for i in range(len(token_ids))
            ],
        animation_name="Token",
        return_fig=True,
    )
    
    for idx, frame in enumerate(fig.frames):
        text = ['']
        
        # nasty hack to ensure board matches intuition since <s> is included
        if idx +1 < len(fig.frames):
            idx+=1 
            
        move_idx = move_id(idx)
        
        board = board_stack[move_idx]
        
        strip_all = lambda s: s.replace(' ', '').replace('\n', '')
        
        text[-1] = [f"<span style='font-size: 16em;'>{c}</span>" 
                for c in strip_all(board.unicode(empty_square='·'))]
        
        frame.data[0]["text"] = np.array(text).reshape(8,8)
        frame.data[0]["texttemplate"] = "%{text}"
        frame.data[0]["hovertemplate"] = """
        <b>%{x}%{y}</b><br>
        log prob: %{z}<br>
        prob=%{customdata}<extra></extra>
        """
        frame.data[0]["customdata"] = to_numpy(log_probs_template[idx].exp())
    
    fig.data[0]["text"] = fig.frames[0].data[0]["text"]
    fig.data[0]["texttemplate"] = fig.frames[0].data[0]["texttemplate"]
    fig.data[0]["customdata"] = fig.frames[0].data[0]["customdata"]
    fig.data[0]["hovertemplate"] = fig.frames[0].data[0]["hovertemplate"]
    if return_fig:
        return fig
    else:
        fig.show()
    
    ####################
    # batch_enc_toks = tokenizer(uci_moves)
    # token_ids = tokenizer.tokenize(uci_moves)
    
    # uci_moves_split = uci_moves.split(' ')
    # token_ids.insert(0,'') # add initial tile selection step
    
    
    # for idx, frame in enumerate(tqdm(fig.frames)):
    #     text = []
    #     shapes = []
        
    #     move_phase = 'from'
        
    #     if idx > 0: #Ignore initial board orientation
    #         token_str = token_ids[idx] #only the current token
    #         token_range = batch_enc_toks.token_to_chars(idx)
    #         game_prefix = uci_moves[:token_range.end-len(token_str)] #excludes token
    #         prev_move_count = len(game_prefix.split(' '))-1
    #         move_str = uci_moves_split[prev_move_count]
    #         move_idx = len(game_prefix.split(' '))-1

    #         move_phase = determine_move_phase(move_str, token_str)


    #     if move_phase == 'from':
    #         if idx > 0: #i.e., not the initial move
    #             #update the board
    #             try:
    #                 prev_move = uci_moves_split[move_idx]
    #                 board.push_uci(prev_move)
    #             except (chess.IllegalMoveError,chess.InvalidMoveError) as ex:
    #                 print("ERR!!!",prev_move)
    #                 break

        
    #     elif move_phase == 'promote':
    #         pass
        
    #     elif move_phase == 'special':
    #         pass
        
    #     for position in range(64):
    #             text.append("")
    #             piece = board.piece_at(idx2sq[position])
    #             symbol = ''
                
    #             if piece is not None:
    #                 symbol = chess.UNICODE_PIECE_SYMBOLS.get(piece.symbol(),'')
                
    #             text[-1] = f"<span style='font-size: 16em;'>{symbol}</span>"
    #             # text[-1] = f"<span style='font-size: 16em;'>{position}</span>"
                
    #     # print(shapes)
    #     frame.layout.shapes = tuple(shapes)
    #     frame.data[0]["text"] = np.array(text).reshape(8,8)
    #     frame.data[0]["texttemplate"] = "%{text}"
    #     frame.data[0]["hovertemplate"] = "<b>%{y}%{x}</b><br>log prob: %{z}<br>prob=%{customdata}<extra></extra>"
    #     frame.data[0]["customdata"] = to_numpy(log_probs_template[idx].exp())
        
    # # fig.layout.shapes = fig.frames[0].layout.shapes
    # fig.data[0]["text"] = fig.frames[0].data[0]["text"]
    # fig.data[0]["texttemplate"] = fig.frames[0].data[0]["texttemplate"]
    # fig.data[0]["customdata"] = fig.frames[0].data[0]["customdata"]
    # fig.data[0]["hovertemplate"] = fig.frames[0].data[0]["hovertemplate"]
    
    # if return_fig:
    #     return fig
    # else:
    #     fig.show()
            


if __name__ == "__main__":
    from mech_interp.visualizations import show_activations_from_cache
    from mech_interp.fixTL import make_official
    from mech_interp.mappings import self_play_move_seq
    from transformers import PreTrainedTokenizerFast
    from transformer_lens import HookedTransformer
    from mech_interp.utils import uci_to_board
    from tqdm import tqdm

    model_name = make_official()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = HookedTransformer.from_pretrained(model_name)

    uci_moves = ' '.join(self_play_move_seq.split(' ')[:60])

    logits, cache = model.run_with_cache(uci_moves)
    for layer in tqdm(range(0,12)):
        fig = show_activations_from_cache_animated(tokenizer, uci_moves, cache, 0,layer, 12, 'attn',return_fig = True)
        fig.write_html(f'./output/animated_activations/layer{layer:02}_heads.html')
        del fig
        
def plot_valid_moves(
    uci_moves,
    is_valid_move: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    return_fig=False,
    slice_tokens=slice(None),
):
    """
    Plots a chessboard highlighting the valid moves based on the provided tensor.

    This function uses a tensor indicating valid moves to plot a chessboard. 
    Each cell of the board is marked to reflect whether a move is valid. 
    The function is specifically tailored for chess move analysis.

    Parameters:
    uci_moves (list): A list of moves in UCI format.
    is_valid_move (torch.Tensor): A 2D tensor indicating valid moves. 
        Each entry should be a boolean indicating whether the corresponding move is valid.
    tokenizer (PreTrainedTokenizerFast): Tokenizer used for encoding UCI moves.
    return_fig (bool, optional): If True, the function returns the figure object. Defaults to False.
    slice_tokens (slice, optional): Slice object to select specific tokens. Defaults to slice(None).

    Returns:
    matplotlib.figure.Figure or None: The chessboard plot figure if return_fig is True, otherwise None.

    Raises:
    AssertionError: If `is_valid_move` does not have the correct shape or the last dimension is not 77.

    Example Usage:
    >>> plot_valid_moves(uci_moves, is_valid_move_tensor, tokenizer, return_fig=True)
    """
    assert len(is_valid_move.shape) == 2, "Use is_valid_move tensor for the correct game only"
    assert is_valid_move.shape[-1] == 77, "is_invalid_move tensor must have final dim == 77"
    
    cell_values = torch.where(is_valid_move, _t(100), _t(-100))
    return plot_board_log_probs(
        uci_moves, tokenizer, cell_values, return_fig, slice_tokens, "Greens"
    )
    

def _t(n):
    return torch.tensor(n, device="cuda", dtype=torch.float32)


def tensor_to_board(tensor):
    board = torch.zeros(
        size=tensor.shape[:-1] + (64,), device=tensor.device, dtype=tensor.dtype
    )
    board[...] = tensor[logitId2square]
    return board.reshape(board.shape[:-1] + (8, 8)).flip(0)


def plot_single_board(
    tensor,
    return_fig = False,
    *args, **kwargs
):
    fig = imshow(
        tensor_to_board(tensor),
        x=[chr(i) for i in range(ord("a"), ord("h") + 1)],
        y=[f'{i}' for i in range(8,0,-1)],
        return_fig = True,
        aspect='equal',
        color_continuous_scale="Blues",
        *args, **kwargs
    )
    if return_fig:
        return fig
    else:
        fig.show()
