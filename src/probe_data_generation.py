import sys
import warnings


print("Running train_probes.py")
print(" ".join(sys.argv))


import os
import time
import math
import argparse
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import wandb
import einops
from mech_interp.fixTL import make_official
from transformers import PreTrainedTokenizerFast
from transformer_lens import HookedTransformer
from dataclasses import field, dataclass
from typing import Callable, Any, List
from enum import Enum
import board_state_functions
import pandas as pd
import utils
# for probe visuals
import plotly.express as px
import plotly.offline as pyo

torch.set_grad_enabled(False)

WANDB_LOG = False
RELOAD_FROM_CHECKPOINT = False
ALWAYS_SAVE_CHECKPOINT = False
LOG_FREQ_IN_SAMPLES = 8000  # this is independent of batch size/epoch
MAKE_PROBE_VISUALS = False

@dataclass
class ProbeConfig:
    num_classes: int
    custom_board_state_function: Callable[[pd.DataFrame], List[Any]]
    num_rows: int = 8
    num_cols: int = 8
    target_layer: int = -1
    slicer: Any = field(default_factory=lambda: slice(None))

class DataSetSplits(Enum):
    TRAIN = "train"
    TEST = "test"

class ProbeType(Enum):
    COLOR_0, COLOR_1, COLOR_2, COLOR_3 = [
        ProbeConfig(
            3,  # white, black, none
            board_state_functions.to_color,
            slicer=slice(start, -1, 4),
        )
        for start in range(4)
    ]

    COLOR_FLIPPING_0, COLOR_FLIPPING_1 = [
        ProbeConfig(
            3,  # white, black, none
            board_state_functions.to_color_flipping,
            slicer=slice(start, -1, 2),
        )
        for start in range(2)
    ]

    PIECE_ANY_0, PIECE_ANY_1 = [
        ProbeConfig(
            7,  # one of: KQBRNP or 'empty'
            board_state_functions.to_piece,
            slicer=slice(start, -1, 2),
        )
        for start in range(2)
    ]

    PIECE_BY_COLOR_0, PIECE_BY_COLOR_1 = [
        ProbeConfig(
            13,  # one of: KQBRNPkqbrnp or 'empty'
            board_state_functions.to_piece_by_color,
            slicer=slice(start, -1, 2),
        )
        for start in range(2)
    ]

    MY_CONTROLLED_0, MY_CONTROLLED_1 = [
        ProbeConfig(
            64,  # one vector for each tile
            board_state_functions.to_my_controlled_tiles,
            slicer=slice(start, -1, 2),
        )
        for start in range(2)
    ]

    THEIR_CONTROLLED_0, THEIR_CONTROLLED_1 = [
        ProbeConfig(
            64,  # one vector for each tile
            board_state_functions.to_my_controlled_tiles,
            slicer=slice(start, -1, 2),
        )
        for start in range(2)
    ]

    def from_str(probe_type: str):
        for enum_val in ProbeType:
            if enum_val.name == probe_type.upper():
                return enum_val
        raise ValueError("No such probe type exists")

class ProbeTrainer:
    # Training parameters
    num_epochs: int
    batch_size: int

    max_lr = 1e-3
    min_lr = max_lr / 100
    weight_decay = 0.01
    wd = 0.01
    betas = (0.9, 0.99)

    max_iters: int
    probe_config: ProbeConfig
    model: HookedTransformer
    split: str
    dataset_prefix: str
    dataset_dir: str
    linear_probe: torch.Tensor
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    run_name: str
    probe_ext = ".pth"
    probe_dir = "chess_data/" # this gets updated with probe name
    checkpoint_filename: str

    def __init__(self):
        print("Initializing Probe")
        parser = initialize_argparser()
        args = parser.parse_args()

        self.num_epochs = args.epochs
        self.batch_size = args.batch_size

        # Setup the probe config
        self.probe_config = ProbeType.from_str(args.probe_type).value

        if args.slice is not None:
            print("Range argument was given. Overriding default probe slice.")
            start, stop, step = args.slice
            self.probe_config.slicer = slice(start, stop, step)

        MODEL_NAME = make_official()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
        self.model = HookedTransformer.from_pretrained(MODEL_NAME, tokenizer=tokenizer)
        self.probe_config.target_layer = args.target_layer

        # Load dataset
        self.dataset_prefix = args.dataset_prefix
        self.dataset_dir = args.dataset_dir
        self.split = args.split
        self.df = pd.read_pickle(
            f"{self.dataset_dir}{self.dataset_prefix}{self.split}.pkl"
        )
        self.max_iters = math.ceil(len(self.df) / self.batch_size) * self.num_epochs

        self.run_name = (
            f"probe_L{self.probe_config.target_layer}_"
            f"_B{self.batch_size}"
            f"_{self.probe_config.slicer}"
            f"_FN_{self.probe_config.custom_board_state_function.__name__}"
        )
        
        self.probe_dir += self.run_name + '/'

        # Initialize the linear probe tensor
        self.checkpoint_filename = self.probe_dir + self.run_name + self.probe_ext
        print("Probe will be saved to ", self.checkpoint_filename)
        if not os.path.exists(self.probe_dir):
            os.makedirs(self.probe_dir, exist_ok=True)
        
        
        if MAKE_PROBE_VISUALS:
            with open(self.probe_dir+'figure_combined.html', "w") as file:
                file.write(get_combined_html())
        
        if args.reload_filename != "":
            self.reload_filename = args.reload_filename
        else:
            self.reload_filename = self.checkpoint_filename
        
        self.linear_probe = self.initialize_probe_tensor(RELOAD_FROM_CHECKPOINT)

        # Loss and Optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            [self.linear_probe],
            lr=self.max_lr,
            betas=self.betas,
            weight_decay=self.wd,
        )
        # self.optimizer = torch.optim.SGD([self.linear_probe],lr=self.max_lr, weight_decay=self.wd)
        
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0 / 10, total_iters=len(self.df)/self.batch_size
        )

        wandb_project_name = "chess_linear_probes"

        self.logging_dict = {
            # "linear_probe": self.linear_probe,
            "num_classes": self.probe_config.num_classes,
            "custom_board_state_function": self.probe_config.custom_board_state_function,
            "num_rows": self.probe_config.num_rows,
            "num_cols": self.probe_config.num_cols,
            "target_layer": self.probe_config.target_layer,
            "slicer_start": self.probe_config.slicer.start,
            "slicer_stop": self.probe_config.slicer.stop,
            "slicer_step": self.probe_config.slicer.step,
            "probe_type": args.probe_type,
            "batch_size": self.batch_size,
            "wd": self.wd,
            "split": self.split,
            "num_epochs": self.num_epochs,
            "wandb_project": wandb_project_name,
            "wandb_run_name": self.run_name,
            "dataset_prefix": self.dataset_prefix,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "wd": self.wd,
            "betas": self.betas,
            "split": self.split,
            "dataset_dir": self.dataset_dir,
            "criterion": self.criterion,
            "scheduler": self.scheduler,
            "optimizer": self.optimizer,
            "probe_ext": self.probe_ext,
            "probe_dir": self.probe_dir,
            "checkpoint_filename": self.checkpoint_filename,
            "RELOAD_FROM_CHECKPOINT": RELOAD_FROM_CHECKPOINT,
            "reload_filename": self.reload_filename,
            "lr": max(self.scheduler.get_last_lr()),
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "function_call": " ".join(sys.argv),
            "job_id": args.jobid,
        }

        if WANDB_LOG:
            tags = None
            if len(args.tags)>0:
                tags = args.tags.split(',')
        
            wandb.init(
                project=wandb_project_name, name=self.run_name, config=self.logging_dict,tags=tags
            )

            artifact = wandb.Artifact("python_scripts", type="file")
            artifact.add_file(os.path.abspath(__file__))
            artifact.add_file(board_state_functions.__file__)
            wandb.log_artifact(artifact)

        print("Probe initialized with config:\n", self.logging_dict)

    def get_lr(
        self, current_iter: int, max_iters: int, max_lr: float, min_lr: float
    ) -> float:
        """
        Calculate the learning rate using linear decay.

        Args:
        - current_iter (int): The current iteration.
        - max_iters (int): The total number of iterations for decay.
        - lr (float): The initial learning rate.
        - min_lr (float): The minimum learning rate after decay.

        Returns:
        - float: The calculated learning rate.
        """
        # Ensure current_iter does not exceed max_iters
        current_iter = min(current_iter, max_iters)

        # Calculate the linearly decayed learning rate
        decayed_lr = max_lr - (max_lr - min_lr) * (current_iter / max_iters)

        return decayed_lr

    def train(
        self,
    ):
        
        with torch.inference_mode():
            # features = []
            # labels = []
            # df = pd.DataFrame({'features':features, 'labels':labels})
            # Use this context manager to ignore the specific FutureWarning

            for batch_idx, (batch_residuals, batch_labels) in tqdm(enumerate(
                utils.get_batches(
                    full_df=self.df,
                    batch_size=self.batch_size,
                    custom_board_state_function=self.probe_config.custom_board_state_function,
                    target_layer=self.probe_config.target_layer,
                    model=self.model,
                )
            ),total = len(self.df)/self.batch_size):
                batch_features_list = []  # Batch of new features
                batch_labels_list = []  # Batch of new labels
                
                batch_size, n_pos, d_model = batch_residuals.shape
                
                for i in range(batch_size):
                    for j in range(n_pos):
                        batch_features_list.append(batch_residuals[i,j].detach().cpu().numpy())
                        batch_labels_list.append(batch_labels[i,j].detach().flatten().cpu().numpy())
            
                batch_df = pd.DataFrame({'features': batch_features_list, 'labels': batch_labels_list})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    batch_df.to_feather(f'chess_data/feathers/batch{batch_idx:06d}_layer0_probe_data.feather')
                # df = pd.concat([df, batch_df], ignore_index=True)
            # df.to_feather('chess_data/layer0_probe_data.feather')
        

    def initialize_probe_tensor(self, reload_checkpoint=True) -> torch.Tensor:
        if reload_checkpoint and os.path.exists(self.reload_filename):
            probe: torch.Tensor = torch.load(self.reload_filename)["linear_probe"]
            probe.cuda()

            print(f"Reloaded from checkpoint:\b    {self.reload_filename}")
            return probe

        cfg = self.probe_config

        probe = torch.randn(
            # dimensions
            self.model.cfg.d_model,
            cfg.num_rows,
            cfg.num_cols,
            cfg.num_classes,
            # options
            requires_grad=False,
            device=self.model.cfg.device,
        ) / np.sqrt(self.model.cfg.d_model)

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
        choices=[
            "lichess_",
        ],  # , "stockfish_"
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
        default=-1,
        help="Optional. Target layer number (Default: -1)",
    )

    # Split
    parser.add_argument(
        "--split",
        type=str,
        choices=[member.name.lower() for member in DataSetSplits],
        default="train",
        help=f"Optional. Dataset split. Choose from {[member.name.lower() for member in DataSetSplits]}) (Default: TRAIN)",
    )

    # Probe type
    parser.add_argument(
        "--probe_type",
        type=str,
        choices=[member.name for member in ProbeType],
        default=list(ProbeType)[0].name,
        help=f"Optional. Type of probe. Choose from {[member.name.lower() for member in ProbeType]}) (Default: COLOR)",
    )

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
        default=50,
        help="Optional. batch sizes (Default: 50)",
    )

    parser.add_argument(
        "--slice",
        type=lambda s: [int(item) for item in s.split(":")],
        default=None,
        help="Enter a range in the format start:stop:step",
    )

    parser.add_argument(
        "--jobid",
        type=str,
        default="",
        help="Job ID if called from SLURM",
    )
    
    parser.add_argument(
        "--reload_filename",
        type=str,
        default="",
        help="(Optional) Relative path to reload checkpoint"
    )
    
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Comma-separated list of tags to add to wandb"
    )

    return parser


def make_visuals(tensor, title, filename, range_color=None):
    tensor = tensor.permute(0,3,1,2)
    # positions = tensor.shape[0]
    # flat_tens = tensor.reshape(positions,-1,8)
    # figure = px.imshow(flat_tens.detach().cpu(), animation_frame = 0, title = title, range_color=range_color, aspect = 'auto')
    figure = px.imshow(
        tensor[0].detach().cpu(),
        facet_col=0,
        # animation_frame=0,
        title=title,
        range_color=range_color,
        aspect="auto",
    )
    if WANDB_LOG:
        wandb.log({f"chart_{title}":figure})
    return pyo.plot(figure, filename=filename, auto_open=False)


if __name__ == "__main__":
    pt = ProbeTrainer()
    try:
        pt.train()
    except KeyboardInterrupt as e:
        if WANDB_LOG:
            wandb.finish(-1)
        raise e
    except Exception as e:
        if WANDB_LOG:
            wandb.finish(-2)
        raise e
    finally:
        # Training is finished. Log probe weights to WANDB if enabled
        if WANDB_LOG:
            artifact = wandb.Artifact("linear_probe", type="model")
            artifact.add_file(pt.checkpoint_filename)
            wandb.log_artifact(artifact)
