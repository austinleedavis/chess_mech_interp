import chess
import torch
import pandas as pd
from typing import List, Union, Iterable
from transformer_lens import HookedTransformer
from typing import Any, Callable, Tuple

def get_batches(
    full_df: pd.DataFrame,
    batch_size: int,
    custom_board_state_function: Callable,
    target_layer: int,
    model: HookedTransformer,
) -> Tuple[torch.Tensor, Any]:
    """
    Generate batches of data for training or evaluation.

    Args:
        full_df (pd.DataFrame): The full dataset.
        batch_size (int): The size of each batch.
        custom_board_state_function (Callable): A function that takes a batch of data and returns the corresponding labels.
        target_layer (int): The target layer for obtaining the activation cache.
        model (HookedTransformer): The model used for inference.

    Yields:
        Tuple: A tuple containing the batch's residual stream and labels.

    """
    for start_idx in range(0, len(full_df), batch_size):
        batch_df = full_df[start_idx : start_idx + batch_size]
        batch_df = batch_df.sample(frac=1).reset_index(drop=True)

        # Obtain ActivationCache for the batch
        with torch.inference_mode():
            _, batch_cache = model.run_with_cache(
                torch.tensor(batch_df["input_ids"].tolist())
            )

        batch_residual_stream = batch_cache[
            "resid_post", target_layer
        ]  # torch.Size([d_batch, n_pos, d_model])

        residuals_clone = batch_residual_stream.clone().detach()
        residuals_clone.requires_grad_(True)
        
        # Get labels for each game in the batch
        batch_labels = custom_board_state_function(batch_df)
        yield residuals_clone, batch_labels
        
        
def make_official(model_name:str = 'AustinD/gpt2-chess-uci-hooked', **kwargs) -> HookedTransformer:
    """
    Hacky fix for Transformer Lens.
    Transformer Lens only supports a few models out-of-the-box. This method adds the
    `model_name` to the official model list, as a workaround, allowing us to directly
    load nearly any model that uses compatible configurations.
    """
    
    import transformer_lens.loading_from_pretrained
    
    if model_name not in transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES:
        transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(model_name)
    return model_name


from typing import Union, Iterable, Callable
import chess

def uci_to_board(uci_moves: Union[str, Iterable],
                 force=False, 
                 fail_silent=False, 
                 print_forced_moves=True, 
                 as_board_stack=False,
                 map_function: Callable = lambda x: x) -> chess.Board | List[chess.Board]:
    """
    Converts a sequence of UCI (Universal Chess Interface) moves to a chess.Board object.

    Args:
        uci_moves (Union[str, Iterable]): The UCI moves to convert. It can be either a string of space-separated moves or an iterable of moves.
        force (bool, optional): If True, allows illegal moves to be forced. Illegal moves must still identify both a valid starting and a valid ending tile. Defaults to False.
        fail_silent (bool, optional): If True, returns the board list up to (but not including) the first illegal move. Defaults to False.
        verbose (bool, optional): If True, prints the forced moves. Defaults to True.
        as_board_stack (bool, optional): If True, returns the history of board states defined by the uci_moves. Defaults to False.
        map_function (Callable, optional): A function to apply to each board state. Defaults to identity function.

    Returns:
        chess.Board or List[chess.Board]: The resulting chess.Board object or a list of board states if `as_board_stack` is True.
    """
    
    board = chess.Board()
    forced_moves = []
    did_force = False
    board_stack = [map_function(board.copy())]
    
    if isinstance(uci_moves, str):
        uci_moves = uci_moves.split(' ')
    
    for i, move in enumerate(uci_moves):
        try:
            move_obj = board.parse_uci(move)
            board.push(move_obj)
        except (chess.IllegalMoveError, chess.InvalidMoveError) as ex:
            if force:
                did_force = True
                forced_moves.append((i, move))
                piece = board.piece_at(chess.parse_square(move[:2]))
                board.set_piece_at(chess.parse_square(move[:2]), None)
                board.set_piece_at(chess.parse_square(move[2:4]), piece)
            elif fail_silent:
                if as_board_stack:
                    return board_stack
                else:
                    return map_function(board)
            else:
                raise ex
        board_stack.append(map_function(board.copy()))
    if print_forced_moves and did_force:
        print(f'Forced (move_id, uci): {forced_moves}')
        
    if as_board_stack:
        return board_stack
    else:
        return map_function(board)