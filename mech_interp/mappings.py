# This file contains mappings between vocab indices, board squares, and vocab tokens

self_play_move_seq = "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6 f2f3 f8e7 d1d2 e8g8 e1c1 a6a5 f1b5 b8a6 c1b1 a6c7 e3b6 d8b8 g2g4 f8c8 g4g5 f6h5 b6e3 a5a4 b3c1 a4a3 b2b3 c7b5 c3b5 c8c6 c1d3 d6d5 e4d5 e6d5 d3b4 c6g6 b4d5 h5f4 e3f4 e5f4 d2f4 e7g5 f4f5 g5e3 d5e3 g6g2 d1d2 g2d2 e3g4 b8f4 f5f4 d2c2 b1c2 a8a2 c2d3 a2b2 h1d1 h7h5 g4e3 h5h4 d3e4 b2h2 d1d8 g8h7 d8d7 h2b2 d7f7 b2b3 e4f5 b3b5 f5g6 h7g8 f7b7 b5g5 g6f5 g5f5 e3f5 g7g5 f5h4 g5h4 b7b8 g8f7 b8b7 f7e6 b7h7 e6d5 h7h4 d5c4 h4h6 c4d3 h6h5 d3e2"

logitId2square = [33, 20, 52, 50, 48, 17, 44, 67, 29, 35, 9, 38, 26, 68, 13, 54, 30, 36, 21, 39, 27, 65, 14, 55, 53, 23, 10, 45, 28, 57, 46, 64, 43, 71, 47, 24, 12, 41, 59, 66, 72, 60, 16, 31, 34, 19, 70, 69, 42, 61, 37, 49, 11, 40, 58, 63, 56, 15, 62, 32, 25, 22, 18, 51]
"""List indices from chess.Squares map to model vocab and/or logits. 

Usage: To align logits to their board positions, run:
```
    logits[logitId2square]
```
Not only will this align the logits to the board squares, but it will truncate
non-square logits (e.g., those associated with eos/bos/pad tokens)


Created using:
```
    template = [None]*64
    for sq in chess.SQUARES:
        sq_name = chess.SQUARE_NAMES[sq]
        logit_id = squareName2logitId_dict.get(sq_name)
        template[sq]=logit_id
```

"""

square2logitId_with_nones = [None, None, None, None, None, None, None, None, None, 10, 26, 52, 36, 14, 22, 57, 42, 5, 62, 45, 1, 18, 61, 25, 35, 60, 12, 20, 28, 8, 16, 43, 59, 0, 44, 9, 17, 50, 11, 19, 53, 37, 48, 32, 6, 27, 30, 34, 4, 51, 3, 63, 2, 24, 15, 23, 56, 29, 54, 38, 41, 49, 58, 55, 31, 21, 39, 7, 13, 47, 46, 33, 40, None, None, None, None]
""" Samge as square2logitId, except Nones are preserved."""


square2logitId = [10, 26, 52, 36, 14, 22, 57, 42, 5, 62, 45, 1, 18, 61, 25, 35, 60, 12, 20, 28, 8, 16, 43, 59, 0, 44, 9, 17, 50, 11, 19, 53, 37, 48, 32, 6, 27, 30, 34, 4, 51, 3, 63, 2, 24, 15, 23, 56, 29, 54, 38, 41, 49, 58, 55, 31, 21, 39, 7, 13, 47, 46, 33, 40]
"""Maps vocab indices or logit index to the chess.Square

Usage: To align board squares to logits, run:
```
    squares[square2logitId]
```

```
    template = [None]*77
    for sq in chess.SQUARES:
        sq_name = chess.SQUARE_NAMES[sq]
        logit_id = squareName2logitId_dict.get(sq_name)
        template[logit_id]=sq
```
"""


squareName2logitId_dict = {"a1":33, "b1":20, "c1":52, "d1":50, "e1":48, "f1":17, "g1":44, "h1":67, "a2":29, "b2":35, "c2":9, "d2":38, "e2":26, "f2":68, "g2":13, "h2":54, "a3":30, "b3":36, "c3":21, "d3":39, "e3":27, "f3":65, "g3":14, "h3":55, "a4":53, "b4":23, "c4":10, "d4":45, "e4":28, "f4":57, "g4":46, "h4":64, "a5":43, "b5":71, "c5":47, "d5":24, "e5":12, "f5":41, "g5":59, "h5":66, "a6":72, "b6":60, "c6":16, "d6":31, "e6":34, "f6":19, "g6":70, "h6":69, "a7":42, "b7":61, "c7":37, "d7":49, "e7":11, "f7":40, "g7":58, "h7":63, "a8":56, "b8":15, "c8":62, "d8":32, "e8":25, "f8":22, "g8":18, "h8":51}
"""Maps board squares to vocab/logit indices. Created with:
```
    for sq in chess.SQUARE_NAMES:
        print(f'"{sq}":{token2logit_dict.get(sq)}', end=', ')
```
"""

logitId2squareName = [None, None, None, None, None, None, None, None, None, 'c2', 'c4', 'e7', 'e5', 'g2', 'g3', 'b8', 'c6', 'f1', 'g8', 'f6', 'b1', 'c3', 'f8', 'b4', 'd5', 'e8', 'e2', 'e3', 'e4', 'a2', 'a3', 'd6', 'd8', 'a1', 'e6', 'b2', 'b3', 'c7', 'd2', 'd3', 'f7', 'f5', 'a7', 'a5', 'g1', 'd4', 'g4', 'c5', 'e1', 'd7', 'd1', 'h8', 'c1', 'a4', 'h2', 'h3', 'a8', 'f4', 'g7', 'g5', 'b6', 'b7', 'c8', 'h7', 'h4', 'f3', 'h5', 'h1', 'f2', 'h6', 'g6', 'b5', 'a6', None, None, None, None]
"""Maps logit indices to board squares. Since d_vocab=77 and a board has only 64 tiles, 
logits that are not associated to a square are assigned the `None` value. Created with:

TODO: Double check the order if used for sorting. This has not been reversed the way the square2logitId and logitId2square lists were.
```
    template = [None]*77
    for sq in chess.SQUARE_NAMES:
        template[square2logit_dict.get(sq)]=sq
```
"""

logitId2token = ['<pad>', '<s>', '</s>', 'P', 'N', 'R', 'B', 'Q', 'K', 'c2', 'c4', 'e7', 'e5', 'g2', 'g3', 'b8', 'c6', 'f1', 'g8', 'f6', 'b1', 'c3', 'f8', 'b4', 'd5', 'e8', 'e2', 'e3', 'e4', 'a2', 'a3', 'd6', 'd8', 'a1', 'e6', 'b2', 'b3', 'c7', 'd2', 'd3', 'f7', 'f5', 'a7', 'a5', 'g1', 'd4', 'g4', 'c5', 'e1', 'd7', 'd1', 'h8', 'c1', 'a4', 'h2', 'h3', 'a8', 'f4', 'g7', 'g5', 'b6', 'b7', 'c8', 'h7', 'h4', 'f3', 'h5', 'h1', 'f2', 'h6', 'g6', 'b5', 'a6', 'q', 'r', 'b', 'n']

"""Maps vocab/logit indices to tokens. Created with

TODO: Double check the order if used for sorting. This has not been reversed the way the square2logitId and logitId2square lists were.

```
    template = [None]*77
    for key in model.tokenizer.vocab.keys():
        template[model.tokenizer.vocab[key]] = key
    print(template)# print(template)
```"""

token2logitId_dict = {'<pad>': 0, '<s>': 1, '</s>': 2, 'P': 3, 'N': 4, 'R': 5, 'B': 6, 'Q': 7, 'K': 8, 'c2': 9, 'c4': 10, 'e7': 11, 'e5': 12, 'g2': 13, 'g3': 14, 'b8': 15, 'c6': 16, 'f1': 17, 'g8': 18, 'f6': 19, 'b1': 20, 'c3': 21, 'f8': 22, 'b4': 23, 'd5': 24, 'e8': 25, 'e2': 26, 'e3': 27, 'e4': 28, 'a2': 29, 'a3': 30, 'd6': 31, 'd8': 32, 'a1': 33, 'e6': 34, 'b2': 35, 'b3': 36, 'c7': 37, 'd2': 38, 'd3': 39, 'f7': 40, 'f5': 41, 'a7': 42, 'a5': 43, 'g1': 44, 'd4': 45, 'g4': 46, 'c5': 47, 'e1': 48, 'd7': 49, 'd1': 50, 'h8': 51, 'c1': 52, 'a4': 53, 'h2': 54, 'h3': 55, 'a8': 56, 'f4': 57, 'g7': 58, 'g5': 59, 'b6': 60, 'b7': 61, 'c8': 62, 'h7': 63, 'h4': 64, 'f3': 65, 'h5': 66, 'h1': 67, 'f2': 68, 'h6': 69, 'g6': 70, 'b5': 71, 'a6': 72, 'q': 73, 'r': 74, 'b': 75, 'n': 76}
"""The model vocab, but I don't have to load the model to use it.
Can be created with:
```
    {item: index for index, item in enumerate(logit2token)}
```"""

if __name__ == '__main__':
    import torch
    mock_logits = torch.Tensor(list(range(0,77)))
    
    # If everythign is working correctly, the mock logits run from 0-76.
    # When we print the mock_logits[logitId2square], the logits are placed
    # to match the chess.board square indices. So, the logit associated 
    # with square 'a1' is in position 1, since the chess.board object 
    # puts `a1` in position 1. 
    
    print(mock_logits)
    print(mock_logits[logitId2square])
    print(token2logitId_dict)
    print(logitId2token)
    
