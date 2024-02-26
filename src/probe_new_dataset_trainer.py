import os
import numpy as np
import pandas as pd
import wandb
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PretrainedConfig
from transformers import Trainer, TrainingArguments
from tqdm.auto import tqdm

# for evaluation steps
import evaluate
EVAL_ACC = evaluate.load('accuracy')
EVAL_PRECISION = evaluate.load('precision')
EVAL_RECALL = evaluate.load('recall')


class FeatherDataset(Dataset):
    def __init__(self, feather_dir, limit_size=None, feature_col='features', label_col='labels'):
        self.feather_dir = feather_dir
        self.feature_col = feature_col
        self.label_col = label_col
        self.feather_files = sorted([f for f in os.listdir(feather_dir) if f.endswith('.feather')])
        self.current_dataframe = None
        self.current_file_index = -1
        self.cumulative_lengths = self._get_cumulative_lengths()
        if limit_size is not None:
            self.cumulative_lengths = [L for L in self.cumulative_lengths if L <= limit_size]
            if not self.cumulative_lengths:
                self.cumulative_lengths = [limit_size]

    def _load_dataframe(self, index):
        file_index = next(i for i, cum_len in enumerate(
            self.cumulative_lengths) if cum_len > index)
        if file_index != self.current_file_index:
            file_path = os.path.join(
                self.feather_dir, self.feather_files[file_index])
            self.current_dataframe = pd.read_feather(file_path)
            self.current_file_index = file_index
        local_index = index - \
            (self.cumulative_lengths[file_index - 1] if file_index > 0 else 0)
        return self.current_dataframe.iloc[local_index]

    def _get_cumulative_lengths(self):
        lengths = []
        cum_length = 0
        for file in tqdm(self.feather_files, desc="Reading feather dataset cumulative lengths."):
            df_length = pd.read_feather(os.path.join(
                self.feather_dir, file), columns=[self.label_col]).shape[0]
            cum_length += df_length
            lengths.append(cum_length)
        return lengths

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):
        row = self._load_dataframe(index)
        feature = torch.tensor(row[self.feature_col], dtype=torch.float32)
        label = torch.tensor(row[self.label_col], dtype=torch.float32)
        return feature, label


class SimpleLinearConfig(PretrainedConfig):
    model_type = "simple_linear"

    def __init__(self, input_dim=768, output_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim


class SimpleLinearModel(PreTrainedModel):
    config_class = SimpleLinearConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config.input_dim, config.output_dim)
        # Initialize weights
        self.init_weights()

    def forward(self, input_ids, labels=None):
        outputs = self.linear(input_ids)
        # If labels are provided, calculate loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs, labels)
        return (loss, outputs) if loss is not None else outputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.where(logits.flatten() >= 0.5, 1, 0)
    references = np.where(labels.flatten() >= 0.5, 1, 0) 
    
    return {
        "accuracy": EVAL_ACC.compute(predictions=predictions, references=references),
        "precision": EVAL_PRECISION.compute(predictions=predictions, references=references, average='micro'),
        "recall": EVAL_RECALL.compute(predictions=predictions, references=references, average='micro'),
    }



def main():
    # Initialize the dataset
    ds_train = FeatherDataset('chess_data/feathers_step4')
    ds_eval = FeatherDataset('chess_data/feathers_step4_eval',limit_size=1_000)
    
    # Initialize the model
    config = SimpleLinearConfig(input_dim=768, output_dim=192)
    model = SimpleLinearModel(config)

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=2,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=1000, 
        warmup_steps=5000,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_total_limit=6,
        logging_dir='./logs',            # directory for storing logs
        report_to='wandb',
        
        overwrite_output_dir=False,      # False to resume training
        load_best_model_at_end=True,
        evaluation_strategy="steps", 
        save_strategy="steps",
        logging_steps=50,
        eval_steps=600,  # Number of update steps between two evaluations.
        save_steps=600,  # must be multiple of eval steps
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        compute_metrics=compute_metrics,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data], dim=0),
                                    'labels': torch.stack([f[1] for f in data], dim=0)}
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        wandb.finish(0)
