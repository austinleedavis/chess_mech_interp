import chess
import chess.svg
from IPython.display import display
from enum import Enum
import textwrap
from dataclasses import dataclass
from os import path
import torch
from transformer_lens import HookedTransformer
import warnings



self_play_move_seq = "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6 f2f3 f8e7 d1d2 e8g8 e1c1 a6a5 f1b5 b8a6 c1b1 a6c7 e3b6 d8b8 g2g4 f8c8 g4g5 f6h5 b6e3 a5a4 b3c1 a4a3 b2b3 c7b5 c3b5 c8c6 c1d3 d6d5 e4d5 e6d5 d3b4 c6g6 b4d5 h5f4 e3f4 e5f4 d2f4 e7g5 f4f5 g5e3 d5e3 g6g2 d1d2 g2d2 e3g4 b8f4 f5f4 d2c2 b1c2 a8a2 c2d3 a2b2 h1d1 h7h5 g4e3 h5h4 d3e4 b2h2 d1d8 g8h7 d8d7 h2b2 d7f7 b2b3 e4f5 b3b5 f5g6 h7g8 f7b7 b5g5 g6f5 g5f5 e3f5 g7g5 f5h4 g5h4 b7b8 g8f7 b8b7 f7e6 b7h7 e6d5 h7h4 d5c4 h4h6 c4d3 h6h5 d3e2"
""" The model hosed in the colab notebook predicted 61 valid moves before making one invalid prediction;
this string contains those 61 moves in uci notation
"""

actual_seq = self_play_move_seq
""" The model hosed in the colab notebook predicted 61 valid moves before making one invalid prediction;
this string contains those 61 moves in uci notation
"""


def get_subgame_prompt_answer(uci_moves:str, k:int, pred_start:bool) -> tuple[str,str]:
    """From a uci move string, we get the subgame up to (inclusive) the kth move. If 
    pred_start then we're predicting the start token, so return that as the 
    `answer`. Otherwise return the destination as the `answer`"""
    
    moves = uci_moves.split(' ')
    
    if k >= len(moves)-1:
        raise ValueError("unable to get answer if prompt uses all tokens. Use a smaller k")
    
    next_move = moves[k]
    prompt = ' '.join(moves[:k])
    
    if pred_start:
        answer = next_move[0:2]
    else:
        answer = next_move[2:]
        prompt += ' '+ next_move[0:2]
        
    return prompt, answer

from typing import Iterable, Union, Callable

def uci_to_board(uci_moves:Union[str,Iterable],
                 force=False, 
                 fail_silent = False, 
                 verbose=True, 
                 as_board_stack=False,
                 map_function:Callable=lambda x: x) -> chess.Board:
    """Returns a chess.Board object from a string of UCI moves
    Paraams:
        force: If true, illegal moves are forcefully made. O/w, the rror is thrown
        verbose: Alert user via prints that illegal moves were attempted."""
    board = chess.Board()
    forced_moves = []
    did_force = False
    board_stack = [map_function(board.copy())]
    
    if isinstance(uci_moves,str):
        uci_moves = uci_moves.split(' ')
    
    for i, move in enumerate(uci_moves):
        try:
            move_obj = board.parse_uci(move)
            board.push(move_obj)
        except (chess.IllegalMoveError,chess.InvalidMoveError) as ex:
            if force:
                did_force = True
                forced_moves.append((i,move))
                piece = board.piece_at(chess.parse_square(move[:2]))
                board.set_piece_at(chess.parse_square(move[:2]),None)
                board.set_piece_at(chess.parse_square(move[2:4]),piece)
            elif fail_silent:
                if as_board_stack:
                    return board_stack
                else:
                    return map_function(board)
            else:
                raise ex
        board_stack.append(map_function(board.copy()))
    if verbose and did_force:
        print(f'Forced (move_id, uci): {forced_moves}')
        
    if as_board_stack:
        return board_stack
    else:
        return map_function(board)


import textwrap

def pretty_moves(uci_moves:str):
    wrapped_moves = '\n '.join(textwrap.wrap(uci_moves,4*13))
    
    colored_moves =  ''.join(['\u001b[30;47m '+word if i % 2 == 0 else '\u001b[40;37m '+word 
                            for i, word in enumerate(wrapped_moves.split(' '))])
    output = 'Moves for \u001b[30;47m White \u001b[0m vs \u001b[40;37m Black \u001b[0m\n'+colored_moves+'\u001b[0m'
            
    return output

def prepare_uci_for_model(model:HookedTransformer, uci_moves:str) -> torch.Tensor:
    game_prefix = [model.tokenizer.bos_token_id]+model.tokenizer.encode(uci_moves,add_special_tokens=False)
    greedy_game_prefix = list(game_prefix)
    prefix_tens = torch.tensor([greedy_game_prefix])
    return prefix_tens

def get_next_tile_pred(model:HookedTransformer, uci_moves:str='') -> tuple[str, torch.Tensor]:
    """moves in uci format, but can be partial moves"""
    logits = model(uci_moves)
    from mech_interp.mappings import logitId2squareName
    return logitId2squareName[logits[0,-1,:].argmax(-1)], logits

def get_next_move(model:HookedTransformer,uci_moves:str='')-> str|None:
    """Assumes the uci string is valid. """
    
    next_move = ''
    game_prefix = uci_moves + ' '
    
    for part in range(3):
        tile, _ = get_next_tile_pred(model,game_prefix)
        game_prefix += tile
        next_move += tile
        if tile is None:
            return None
        
    if len(next_move) >5: #fix move if predicted non-move token
        next_move = next_move[:4]
    return next_move

def get_next_move_arrow(uci_moves:str='') -> chess.svg.Arrow:
    next_move_uci = get_next_move(uci_moves)
    next_move = chess.Move.from_uci(next_move_uci)
    return chess.svg.Arrow(next_move.from_square,next_move.to_square)

def display_game(board: chess.Board, uci_moves:str, **kwargs) -> None:
    color_move_str = pretty_moves(uci_moves)
    color_move_str.replace('\n','\n    ')
    print(
            f'   Game length: {len(board.move_stack)} moves\n',
            f'  Game result: {board.result()}\n',
            f'  Move sequence: \n   ',
            
            color_move_str
          )
    svg = chess.svg.board(board, **kwargs)
    svg = svg.replace('stroke="none"','stroke="black"')
    display(svg)
    
def _normalize_tensor(data: torch.Tensor) -> torch.Tensor:
    return (data - data.min()) / (data.max()-data.min())


from mech_interp.mappings import logitId2square
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _tensor_to_colors(tensor:torch.Tensor, cm_theme = 'Blues') -> dict[int,str]:
    """converts a tensor to a dictionary of hex colors according to a matplotlib colormap."""
    sorted_values = tensor[logitId2square]
    normalized = (sorted_values-sorted_values.min()) / (sorted_values.max()-sorted_values.min())
    colormap = plt.colormaps[cm_theme]
    colors_list = [mcolors.to_hex(colormap(value)) for value in normalized.cpu()]
    colors_dict = {index: color for index, color in enumerate(colors_list)}
    return colors_dict

def plot_board_probs(board:chess.Board, 
                         moves:str, 
                         logits:torch.Tensor,
                         return_fig=False, 
                         cm_theme = 'viridis', **kwargs) -> chess.Board | None:
    """
    Displays (and optionally returns) an svg where each board tile color indicates magnitude of the probability.
        return_fig: boolean to either return OR display the svg
        cm_theme: matplotlib color theme name cf [Matplotlib Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html).
    """
    
    probs = logits.softmax(-1)
    # sorted_lob_probs = probs[logitId2square]
    
    # normalized = _normalize_tensor(sorted_lob_probs)
    
    # colormap = plt.colormaps[cm_theme]
    # colors_list = [mcolors.to_hex(colormap(value)) for value in normalized.cpu()]
    # colors_dict = {index: color for index, color in enumerate(colors_list)}
    colors_dict = _tensor_to_colors(probs,cm_theme=cm_theme)
    
    if return_fig:
        svg = chess.svg.board(board, fill=colors_dict, **kwargs)
        svg = svg.replace('stroke="none"','stroke="black"')
        return svg
    else:
        display_game(board, moves, fill=colors_dict, **kwargs)

######
# DEPRICATION SECTION
######
# Since this is a developmental effort, some things are no longer userful, or have been
# supplanted by better techniques. This section maintains old code for backward compatibility
# of my code without having to update the old code.

@dataclass 
class ChessDataSetEntry():
    """Depricated"""
    file_path:str
    num_games:int

class ChessDataset(Enum):
        """Depricated"""
        DEV = ChessDataSetEntry('chess_data/uci/dev.txt',15_000)
        OTHER_EVAL = ChessDataSetEntry('chess_data/uci/other_eval.txt',50_000)
        TEST = ChessDataSetEntry('chess_data/uci/test.txt',15_000)
        TRAIN = ChessDataSetEntry('chess_data/uci/train.txt',580_000)


class ChessDataProvider():
    """Depricated"""
    def __init__(self, data_dir=None):
        warnings.warn("Use of ChessDataProvider is no longer best. Use mech_interp.chess_dataset.ChessDataImporter instead")
        self.train_file = path.join(data_dir,"train.txt")
        self.test_file = path.join(data_dir,"test.txt")
        self.dev_file = path.join(data_dir,"dev.txt")
        self.other_eval_file = path.join(data_dir,"other_eval.txt")


def __read_line_from_file__(file_path, line_number):
    """Depricated"""
    if line_number < 1:
        raise ValueError("Line number must be greater than 0")

    with open(file_path, 'r') as file:
        for current_line_number, line in enumerate(file, start=1):
            if current_line_number == line_number:
                return line.strip()

    raise ValueError("Line number out of range")


def load_game_from_dataset(data_set: ChessDataset, game_number:int, ) -> tuple[chess.Board, str]:
    """
    Depricated
    
    Reads a single game from the dataset. 
        data_set: The data set to read from
        game_number: The game number to read
    
    Returns a tuple of the board and the move sequence as a string
    """
    moves = __read_line_from_file__(data_set.value.file_path, game_number)
    return uci_to_board(moves), moves

def display_game_from_dataset(data_set: ChessDataset, game_number:int, **kwargs) -> None:
    
    """
    Depricated
    
    Displays a single game from the dataset. 
        data_set: The data set to read from
        game_number: The game number to read
    """
    board, moves= load_game_from_dataset(data_set, game_number)
    print(f'Game {game_number} from {data_set.name}')
    display_game(board,moves, **kwargs)


if '__main__' == __name__:
    pass
    # import sys
    # sys.path.append('/LCB')
    # sys.path = list(set(sys.path))
    
    # game_str = 'd2d4 g8f6 c2c4'
    # print(f'get_next_tile_pred(game_str):\n{get_next_tile_pred("d2d4 g8f6 c2")}')
    # print(f'get_next_move(game_str):\n{get_next_move(game_str)}')
    # from IPython.display import display 
    # display(chess.svg.board(uci_to_board(game_str),arrows=[get_next_move_arrow(game_str)],size=300))
    
    
    # actual_seq = "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6 f2f3 f8e7 d1d2 e8g8 e1c1 a6a5 f1b5 b8a6 c1b1 a6c7 e3b6 d8b8 g2g4 f8c8 g4g5 f6h5 b6e3 a5a4 b3c1 a4a3 b2b3 c7b5 c3b5 c8c6 c1d3 d6d5 e4d5 e6d5 d3b4 c6g6 b4d5 h5f4 e3f4 e5f4 d2f4 e7g5 f4f5 g5e3 d5e3 g6g2 d1d2 g2d2 e3g4 b8f4 f5f4 d2c2 b1c2"
    # """ The model hosed in the colab notebook predicted 61 valid moves before making one invalid prediction;
    # this string contains those 61 moves in uci notation
    # """

    # from tqdm import tqdm

    # moves = ''
    # for i in tqdm(range(61)):
    #     next_move = get_next_move(moves)
    #     moves += ' '+ next_move

    # from austin.utils import print_moves

    # print(f'Local model matches colab? {moves.strip() == actual_seq}')
    
    # #Verifying model.run_with_cache() matches model() output
    # uci_moves = 'e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6'
    # prefix_tens = prepare_uci_for_model(actual_seq)
    # logits1 = model(prefix_tens)

    # logits2, _ = model.run_with_cache(prefix_tens)

    # print(f'model.run_with_cache equals simple inference?: {logits1.equal(logits2)}')
