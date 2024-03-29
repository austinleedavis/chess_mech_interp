print("Running train_probes.py")

import os
import argparse
from dataclasses import dataclass
from typing import Literal, Callable, Any, List
import pandas as pd
import torch
import numpy as np
import wandb

# import fancy_einsum as fancy
import einops
from mech_interp.fixTL import make_official
from transformers import PreTrainedTokenizerFast
from transformer_lens import HookedTransformer
from enum import Enum
import board_state_functions
import utils

WANDB_LOG = True
RELOAD_FROM_CHECKPOINT = True


@dataclass
class ProbeConfig:
    linear_probe_name: str
    num_classes: int
    custom_board_state_function: Callable[[pd.DataFrame], List[Any]]
    num_rows: int = 8
    num_cols: int = 8
    target_layer: int = -1
    model: HookedTransformer = None


class DataSetSplits(Enum):
    TRAIN = "train"
    TEST = "test"

class ProbeType(Enum):
    COLOR = ProbeConfig(
        "color_probe",
        3,  # white, black, none
        board_state_functions.df_to_color_state,
    )
    
    COLOR_FLIPPED = ProbeConfig(
        "color_probe_flipped",
        3,  # white, black, none
        board_state_functions.df_to_color_state_flip_player,
    )
    
    PIECE = ProbeConfig(
        "piece_probe",
        13,  # one of: KQBRNPkqbrnp or 'empty'
        board_state_functions.df_to_piece_state,
    )

    def from_str(probe_type: str):
        for enum_val in ProbeType:
            if enum_val.name == probe_type.upper():
                return enum_val
        raise ValueError("No such probe type exists")


class ProbeTrainer:
    # Training parameters
    num_epochs: int
    batch_size: int
    learning_rate = 0.001
    weight_decay = 0.01
    wd = 0.01
    betas = (0.9, 0.99)

    probe_config: ProbeConfig
    split: str
    dataset_prefix: str
    dataset_dir: str
    linear_probe: torch.Tensor
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    run_name: str
    probe_ext = ".pth"
    probe_dir = "linear_probes/saved_probes/"
    checkpoint_filename: str
    pos_slice = slice(2,-1,4)

    def __init__(self):
        print("Initializing Probe")
        parser = initialize_argparser()
        args = parser.parse_args()

        self.num_epochs = args.epochs
        self.batch_size = args.batch_size

        # Setup the probe config
        self.probe_config = ProbeType.from_str(args.probe_type).value
        MODEL_NAME = make_official()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
        self.probe_config.model = HookedTransformer.from_pretrained(
            MODEL_NAME, tokenizer=tokenizer
        )
        self.probe_config.target_layer = args.target_layer

        # Load dataset
        self.dataset_prefix = args.dataset_prefix
        self.dataset_dir = args.dataset_dir
        self.split = args.split
        self.df = pd.read_pickle(
            f"{self.dataset_dir}{self.dataset_prefix}{self.split}.pkl"
        )

        self.run_name = (
            # f"{self.probe_config.linear_probe_name}_"
            # f"{self.probe_config.model.cfg.model_name}_"
            f"color_pos_{self.pos_slice}_"
            f"layer_{self.probe_config.target_layer}_"
            f"indexing_{self.probe_config.custom_board_state_function.__name__}"
        )

        # Initialize the linear probe tensor
        self.checkpoint_filename = self.probe_dir + self.run_name + self.probe_ext
        print("Probe will be saved to ", self.checkpoint_filename)

        self.linear_probe = self.initialize_probe_tensor(RELOAD_FROM_CHECKPOINT)

        # Loss and Optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            [self.linear_probe],
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.wd,
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0 / 10, total_iters=len(self.df)
        )

        project_name = "mech_interp_chess"

        self.logging_dict = {
            "linear_probe_name": self.probe_config.linear_probe_name,
            "model_name": self.probe_config.model.cfg.model_name,
            "layer": self.probe_config.target_layer,
            "indexing_function_name": self.probe_config.custom_board_state_function.__name__,
            "batch_size": self.batch_size,
            "wd": self.wd,
            "split": self.split,
            "num_epochs": self.num_epochs,
            "num_classes": self.probe_config.num_classes,
            "wandb_project": project_name,
            "wandb_run_name": self.run_name,
            "dataset_prefix": self.dataset_prefix,
            "token_slice": self.pos_slice
        }

        if WANDB_LOG:
            wandb.init(
                project=project_name, name=self.run_name, config=self.logging_dict
            )
        
        print('Probe initialized with config:\n', self.logging_dict)

    def train(
        self,
    ):
        print("Begging Training")
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}")
            for batch_idx, (batch_residuals, batch_labels) in enumerate(
                utils.get_batches(
                    full_df=self.df,
                    batch_size=self.batch_size,
                    custom_board_state_function=self.probe_config.custom_board_state_function,
                    target_layer=self.probe_config.target_layer,
                    model=self.probe_config.model,
                )
            ):
                
                # Select every 4th pos from residuals and labels
                batch_residuals = batch_residuals[:,1:-1:4,:]
                batch_labels = batch_labels[:,1:-1:4,...]
                
                # print('batch_labels.shape: ', batch_labels.shape)
                # print('batch_residuals.shape: ', batch_residuals.shape)
                # Forward pass using einsum
                probe_output: torch.Tensor = einops.einsum(
                    batch_residuals,
                    self.linear_probe,
                    "batch pos d_model, d_model rows cols classes -> batch pos rows cols classes",
                )
                
                # print('probe_output.shape: ',probe_output.shape)

                # residuals: torch.Size([batch, pos, 768])
                # probe:    torch.Size([768, 8, 8, 3])
                # batch_labels:  torch.Size([batch, 126, 8, 8, 3]) -> torch.Size([403200, 3])
                # output:   torch.Size([batch, 126, 8, 8, 3]) -> torch.Size([403200, 3])

                # Reshape probe_output and batch_labels for CrossEntropyLoss

                # Assumes final dim is the class
                # probe_output = probe_output.flatten(0, -2)
                # batch_labels = batch_labels.flatten(0, -2)

                # Calculate loss
                loss = self.criterion(probe_output, batch_labels)

                accuracy = (
                    (probe_output.argmax(-1) == batch_labels.argmax(-1)).float().mean()
                )

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if (batch_idx + 1) % 20 == 0:
                    checkpoint = {
                        "acc": accuracy,
                        "loss": loss,
                        "lr": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "batch": batch_idx,
                        "linear_probe": self.linear_probe,
                        "batch": batch_idx,
                        "epoch": epoch,
                    }

                    checkpoint.update(self.logging_dict)
                    print("SKIP SAVE!")
                    torch.save(checkpoint, self.checkpoint_filename)

                    if epoch % 1 == 0:
                        print(
                            f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}"
                        )
                        if WANDB_LOG:
                            wandb.log(
                                {
                                    "acc": accuracy,
                                    "loss": loss,
                                    "lr": self.scheduler.get_last_lr()[0],
                                    "epoch": epoch,
                                    "batch": batch_idx,
                                }
                            )

    def initialize_probe_tensor(self, reload_checkpoint=True) -> torch.Tensor:
        if reload_checkpoint and os.path.exists(self.checkpoint_filename):
            probe: torch.Tensor = torch.load(self.checkpoint_filename)["linear_probe"]
            probe.cuda()
            probe.requires_grad = True

            print(f"Reloaded from checkpoint:\b    {self.checkpoint_filename}")
            return probe

        cfg = self.probe_config

        probe = torch.randn(
            # dimensions
            cfg.model.cfg.d_model,
            cfg.num_rows,
            cfg.num_cols,
            cfg.num_classes,
            # options
            requires_grad=False,
            device=cfg.model.cfg.device,
        ) / np.sqrt(cfg.model.cfg.d_model)

        probe.requires_grad = True
        return probe


def initialize_argparser():
    parser = argparse.ArgumentParser(
        description="CLI for training linear regression models on a Language Model's residual stream. "
        "This tool allows you to specify various parameters for the dataset and model configuration. "
        "It's designed for experiments with different layers and types of probes on the residual stream of a Language Model, "
        "such as a transformer-based model, to analyze and understand its behavior."
    )

    # Dataset prefix
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        choices=["lichess_", "stockfish_"],
        default="lichess_",
        help="Prefix of the dataset (Default: 'lichess_')",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="chess_data/",
        help="Directory where data is stored. (Default 'chess_data/')",
    )

    # Layer
    parser.add_argument(
        "--target_layer",
        type=int,
        default=0,
        help="Optional. Target layer number (Default: -1)",
    )

    # Split
    parser.add_argument(
        "--split",
        type=str,
        choices=[member.name for member in DataSetSplits],
        default="train",
        help=f"Optional. Dataset split. Choose from {[member.name for member in DataSetSplits]}) (Default: TRAIN)",
    )

    # Probe type
    parser.add_argument(
        "--probe_type",
        type=str,
        choices=[member.name for member in ProbeType],
        default="color",
        help=f"Optional. Type of probe. Choose from {[member.name for member in ProbeType]}) (Default: COLOR)",
    )

    # epocs
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Optional. Number of epochs to train (Default: 1)",
    )

    # epocs
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Optional. batch sizes (Default: 100)",
    )

    return parser


if __name__ == "__main__":
    pt = ProbeTrainer()
    pt.train()
