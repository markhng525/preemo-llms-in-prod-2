import os
import sys
import torch
import contextlib
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from utils.memory_trace import TorchTracemalloc, b2mb
import typer
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from constants import PROJ_ROOT


class StreamToLogger:
    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass


logger.remove()
logger.add(sys.__stdout__)
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
cli = typer.Typer()
OUTPUT_DIR = f"{PROJ_ROOT}/output/gsm8k-exp/{timestamp}/"

CHECKPOINTS_ROOT_DIR = f"{PROJ_ROOT}/output/gsm8k-exp/{timestamp}/checkpoints"
os.makedirs(CHECKPOINTS_ROOT_DIR, exist_ok=True)
logger.add(sink=os.path.join(OUTPUT_DIR, "log.log"))

DATA_ROOT_DIR = "/mnt/ml-data/data-synthesis/finetuning/"


@cli.command()
def train(
    model_name_or_path: str,
    batch_size: int = 2,
    max_length: int = 256,
    num_epochs: int = 1,
) -> None:
    accelerator = Accelerator()
    batch_size = batch_size
    gradient_accumulation_steps = 1
    lr = 1e-4
    num_epochs = num_epochs
    train_data = "gsm8k_train.jsonl"
    test_data = "gsm8k_test.jsonl"
    pad_token_id = -100
    seed = 42
    set_seed(seed)
    max_length = max_length
    eval_step_num = 500

    global_step = 1
    if int(os.environ["LOCAL_RANK"]) == 0:
        logger.info(f"num_epochs: {num_epochs}")
        logger.info(f"batch size:{batch_size}, max_length: {max_length}")
    # specific to T5

    peft_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
    )

    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
    )
    # model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        DATA_ROOT_DIR,
        data_files={
            "train": train_data,
            "test": test_data,
        },
        cache_dir="./cache",
    )

    dataset_train_val_split = dataset["train"].train_test_split(test_size=0.2)
    dataset["validation"] = dataset_train_val_split["test"]
    dataset["train"] = dataset_train_val_split["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples, padding="max_length"):
        model_inputs = tokenizer(
            examples["question"],
            max_length=max_length,
            padding=padding,
            truncation=True,
        )

        labels = tokenizer(
            examples["answer"],
            max_length=max_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    # Run all of the dataset preprocessing on the CPU main process
    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Tokenizing dataset...",
        )
    accelerator.wait_for_everyone()

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=pad_token_id,
        padding="max_length",
        max_length=max_length,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    (
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )

    def evaluation(model, global_step: int) -> None:
        logger.info(f"now evaluating model at step {step}")
        model.eval()
        eval_loss = 0
        with TorchTracemalloc() as tracemalloc:
            for _step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model)(
                        **batch,
                        # synced_gpus=is_ds_zero_3
                    )
                loss = outputs.loss
                eval_loss += loss.detach().float()
            logger.critical(
                f"eval_loss at global step {global_step}: {eval_loss/len(eval_dataloader)}"
            )
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print(
            "GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin))
        )
        accelerator.print(
            "GPU Memory consumed at the end of the eval (end-begin): {}".format(
                tracemalloc.used
            )
        )
        accelerator.print(
            "GPU Peak Memory consumed during the eval (max-begin): {}".format(
                tracemalloc.peaked
            )
        )
        accelerator.print(
            "GPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print(
            "CPU Memory before entering the eval : {}".format(
                b2mb(tracemalloc.cpu_begin)
            )
        )
        accelerator.print(
            "CPU Memory consumed at the end of the eval (end-begin): {}".format(
                tracemalloc.cpu_used
            )
        )
        accelerator.print(
            "CPU Peak Memory consumed during the eval (max-begin): {}".format(
                tracemalloc.cpu_peaked
            )
        )
        accelerator.print(
            "CPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )

    def dump_model(model, step: int) -> None:
        logger.info(f"dumping model at step {step}")
        model.save_pretrained(
            f"{CHECKPOINTS_ROOT_DIR}/{model_name_or_path.split('/')[-1]}-finetuned-model-step-{step}"
        )

    for epoch in range(num_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()

                global_step += 1
                if global_step % eval_step_num == 0:
                    evaluation(model=model, step=global_step)
                    dump_model(model=model, step=global_step)
                    model.train()
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print(
            "GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin))
        )
        accelerator.print(
            "GPU Memory consumed at the end of the train (end-begin): {}".format(
                tracemalloc.used
            )
        )
        accelerator.print(
            "GPU Peak Memory consumed during the train (max-begin): {}".format(
                tracemalloc.peaked
            )
        )
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print(
            "CPU Memory before entering the train : {}".format(
                b2mb(tracemalloc.cpu_begin)
            )
        )
        accelerator.print(
            "CPU Memory consumed at the end of the train (end-begin): {}".format(
                tracemalloc.cpu_used
            )
        )
        accelerator.print(
            "CPU Peak Memory consumed during the train (max-begin): {}".format(
                tracemalloc.cpu_peaked
            )
        )
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
        evaluation(model=model, step=global_step)
        dump_model(model=model, step=global_step)


if __name__ == "__main__":
    stream = StreamToLogger()
    with contextlib.redirect_stdout(stream):
        cli()
