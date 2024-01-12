# %%

import pandas as pd
import chess
import torch


def df_to_color_state(df: pd.DataFrame):
    """
    Converts a DataFrame of chess games into a tensor representation of the board states,
    with separate channels for white pieces, black pieces, and empty squares.

    Parameters:
    df (pd.DataFrame): DataFrame containing chess games. Each game is represented as a list of FEN strings in the 'fen_stack' column.

    Returns:
    torch.Tensor: A 5D tensor representing the board states. The dimensions are [batch, position, row, column, class],
    where class is one of {white pieces, black pieces, empty squares}.
    """

    state_list = []
    for index, row in df[['fen_stack','input_ids']].iterrows():
        board_state_list = []
        for board_state in row['fen_stack']:
            board = chess.Board(board_state)

            mask_w = board.occupied_co[chess.WHITE]
            mask_b = board.occupied_co[chess.BLACK]

            # # This swaps masks based on mine-vs-theirs idea
            # if board.turn:
            #     temp = mask_w
            #     mask_w = mask_b
            #     mask_b = mask_w

            blanks = ~(mask_w | mask_b) & 0xFFFFFFFFFFFFFFFF
            board_state_list.append(
                [_int_to_bits(mask_w), _int_to_bits(mask_b), _int_to_bits(blanks)]
            )
            # append twice because moves require 2 input tokens
            board_state_list.append(
                [_int_to_bits(mask_w), _int_to_bits(mask_b), _int_to_bits(blanks)]
            )
        state_list.append(board_state_list)
    
    state_tensor = torch.tensor(state_list, dtype=torch.float32, device="cuda:0")
    
    state_tensor = state_tensor.reshape(len(df), len(row['input_ids']), 3, 8, 8 )

    return state_tensor.permute(0, 1, 3, 4, 2)  # [batch, pos, row, col, class]

def df_to_color_state_flip_player(df: pd.DataFrame):
    """
    Converts a DataFrame of chess games into a tensor representation of the board states,
    with separate channels for white pieces, black pieces, and empty squares.

    Parameters:
    df (pd.DataFrame): DataFrame containing chess games. Each game is represented as a list of FEN strings in the 'fen_stack' column.

    Returns:
    torch.Tensor: A 5D tensor representing the board states. The dimensions are [batch, position, row, column, class],
    where class is one of {white pieces, black pieces, empty squares}.
    """

    state_list = []
    for index, row in df[['fen_stack','input_ids']].iterrows():
        board_state_list = []
        for board_state in row['fen_stack']:
            board = chess.Board(board_state)

            mask_w = board.occupied_co[chess.WHITE]
            mask_b = board.occupied_co[chess.BLACK]

            # # This swaps masks based on mine-vs-theirs idea
            if board.turn:
                temp = mask_w
                mask_w = mask_b
                mask_b = mask_w

            blanks = ~(mask_w | mask_b) & 0xFFFFFFFFFFFFFFFF
            board_state_list.append(
                [_int_to_bits(mask_w), _int_to_bits(mask_b), _int_to_bits(blanks)]
            )
            # append twice because moves require 2 input tokens
            board_state_list.append(
                [_int_to_bits(mask_w), _int_to_bits(mask_b), _int_to_bits(blanks)]
            )
        state_list.append(board_state_list)
    
    state_tensor = torch.tensor(state_list, dtype=torch.float32, device="cuda:0")
    
    state_tensor = state_tensor.reshape(len(df), len(row['input_ids']), 3, 8, 8 )

    return state_tensor.permute(0, 1, 3, 4, 2)  # [batch, pos, row, col, class]


def df_to_piece_state(df: pd.DataFrame):
    state_list = []

    for fen_stack in df["fen_stack"]:
        board = chess.Board(fen_stack[-1])

        mask_w = board.occupied_co[chess.WHITE]
        mask_b = board.occupied_co[chess.BLACK]

        # # This swaps masks based on mine-vs-theirs idea
        # if board.turn:
        #     temp = mask_w
        #     mask_w = mask_b
        #     mask_b = mask_w

        blanks = ~(mask_w | mask_b) & 0xFFFFFFFFFFFFFFFF

        state_list.append(
            [
                # white pieces
                _int_to_bits(board.pawns & mask_w),
                _int_to_bits(board.knights & mask_w),
                _int_to_bits(board.bishops & mask_w),
                _int_to_bits(board.rooks & mask_w),
                _int_to_bits(board.queens & mask_w),
                _int_to_bits(board.kings & mask_w),
                # black pieces
                _int_to_bits(board.pawns & mask_b),
                _int_to_bits(board.knights & mask_b),
                _int_to_bits(board.bishops & mask_b),
                _int_to_bits(board.rooks & mask_b),
                _int_to_bits(board.queens & mask_b),
                _int_to_bits(board.kings & mask_b),
                # blanks
                _int_to_bits(blanks),
            ]
        )

    return state_list


def _int_to_bits(value: int):
    bits = [(value >> i) & 1 for i in range(63, -1, -1)]
    return bits


# %%
if __name__ == "__main__":
    
    # do Run Above!
    
    df = pd.read_pickle("chess_data/lichess_test.pkl")
    print(len(df), df.keys(), sep="\n")
    small_df = df.iloc[:5]
    color_state_stack = df_to_color_state(small_df)
    print(color_state_stack.shape)


# %%
