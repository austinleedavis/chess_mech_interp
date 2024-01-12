from dataclasses import dataclass, field
from typing import ClassVar, Literal, List, Final
import os

col_names = [f'{i}' for i in range(8,0,-1)]
row_names = 'a b c d e f g h'.split(' ')

id2square = [33, 20, 52, 50, 48, 17, 44, 67, 29, 35, 9, 38, 26, 68, 13, 54, 30, 36, 21, 39, 27, 65, 14, 55, 53, 23, 10, 45, 28, 57, 46, 64, 43, 71, 47, 24, 12, 41, 59, 66, 72, 60, 16, 31, 34, 19, 70, 69, 42, 61, 37, 49, 11, 40, 58, 63, 56, 15, 62, 32, 25, 22, 18, 51]
"""Usage:  `logits[id2square]`
ill this align logits to board squares, but it will discard
non-square logits (e.g., those associated with eos/bos/pad tokens)"""

id2squareName = [None, None, None, None, None, None, None, None, None, 'c2', 'c4', 'e7', 'e5', 'g2', 'g3', 'b8', 'c6', 'f1', 'g8', 'f6', 'b1', 'c3', 'f8', 'b4', 'd5', 'e8', 'e2', 'e3', 'e4', 'a2', 'a3', 'd6', 'd8', 'a1', 'e6', 'b2', 'b3', 'c7', 'd2', 'd3', 'f7', 'f5', 'a7', 'a5', 'g1', 'd4', 'g4', 'c5', 'e1', 'd7', 'd1', 'h8', 'c1', 'a4', 'h2', 'h3', 'a8', 'f4', 'g7', 'g5', 'b6', 'b7', 'c8', 'h7', 'h4', 'f3', 'h5', 'h1', 'f2', 'h6', 'g6', 'b5', 'a6', None, None, None, None]

@dataclass()
class ChessDataImporter():
    
    name: str
    """name of the dataset, either 'train', 'test', 'dev', or 'other_eval'"""
    num_games: int
    """number of games in the dataset"""
    games: List[str] = field(repr=False)
    """The list of the game strings loaded in the dataset"""
    
    VALID_NAMES: ClassVar[List[str]] = ['train','test','dev','other_eval']
    
    def __init__(self, 
                 data_set: Literal['train','test','dev','other_eval']='other_eval',
                 root_folder = './chess_data/uci/',
                 ):

        if data_set not in self.VALID_NAMES:
            raise ValueError("data_set must be one of 'train', 'test', 'dev', or 'other_eval'")
        self.name = data_set
        
        with open(os.path.join(root_folder,data_set+".txt")) as file:
            self.games = [line.strip() for line in file]
        
            
        self.num_games = len(self.games)
            
if __name__ == '__main__':
    val = ChessDataImporter(data_set='dev')
    print(val)
    
