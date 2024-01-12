
import einops
import torch
import utils
import wandb
import pandas as pd
from transformer_lens import HookedTransformer

#training parameters
NUM_EPOCHS = None
BATCH_SIZE = None
CRITERION = None
OPTIMIZER = None
SCHEDULER = None
DATASET_FILE = None

DF = pd.read_pickle(DATASET_FILE)
MODEL = None
CHECKPOINT_FILENAME = None
CUSTOM_BOARD_STATE_FUNCTION = None
TARGET_LAYER = None
LINEAR_PROBE = None
LOGGING_DICT = None
WANDB_LOG = False

for epoch in range(NUM_EPOCHS):
    for batch_idx, (batch_residuals, batch_labels) in enumerate(
        utils.get_batches(
            full_df=DF,
            batch_size=BATCH_SIZE,
            custom_board_state_function=CUSTOM_BOARD_STATE_FUNCTION,
            target_layer=TARGET_LAYER,
            model=MODEL,
        )
    ):
        
        # Forward pass using einsum
        probe_output: torch.Tensor = einops.einsum(
            batch_residuals,
            LINEAR_PROBE,
            "batch pos d_model, d_model rows cols classes -> batch pos rows cols classes",
        )

        # residuals: torch.Size([batch, pos, 768])
        # probe:    torch.Size([768, 8, 8, 3])
        # batch_labels:  torch.Size([batch, 126, 8, 8, 3]) -> torch.Size([403200, 3])
        # output:   torch.Size([batch, 126, 8, 8, 3]) -> torch.Size([403200, 3])

        # Reshape probe_output and batch_labels for CrossEntropyLoss

        # Assumes final dim is the class
        # probe_output = probe_output.flatten(0, -2)
        # batch_labels = batch_labels.flatten(0, -2)

        # Calculate loss
        loss = CRITERION(probe_output, batch_labels)

        accuracy = (
            (probe_output.argmax(-1) == batch_labels.argmax(-1)).float().mean()
        )

        # Backward and optimize
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        SCHEDULER.step()

        if (batch_idx + 1) % 20 == 0:
            checkpoint = {
                "acc": accuracy,
                "loss": loss,
                "lr": SCHEDULER.get_last_lr()[0],
                "epoch": epoch,
                "batch": batch_idx,
                "linear_probe": LINEAR_PROBE,
                "batch": batch_idx,
                "epoch": epoch,
            }

            checkpoint.update(@@logging_dict)
            torch.save(checkpoint, @@checkpoint_filename)

            if epoch % 1 == 0:
                print(
                    f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}"
                )
                if WANDB_LOG:
                    wandb.log(
                        {
                            "acc": accuracy,
                            "loss": loss,
                            "lr": SCHEDULER.get_last_lr()[0],
                            "epoch": epoch,
                            "batch": batch_idx,
                        }
                    )