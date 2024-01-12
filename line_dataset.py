import logging
import os
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from os import path
import numpy as np
from chess_dataset import *

logger = logging.getLogger(__name__)

class LineByLineTextDataset(Dataset):

    def __init__(self, tokenizer, file_path, n_ctx, batch_size=None,):
        
        assert os.path.isfile(file_path)
        
        self.tokenizer = tokenizer

        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            if batch_size:
                self.lines = self.lines[:batch_size]

        batch_encoding = tokenizer(self.lines, add_special_tokens=True, truncation=True, max_length=n_ctx)
        self.examples = batch_encoding["input_ids"]
        self.end_positions = batch_encoding["end_positions"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        output_dict = {}

        output_dict['input_ids'] = torch.tensor(self.examples[i])
        output_dict['separator_ind'] = self.end_positions[i]

        return output_dict

