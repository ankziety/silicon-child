"""LoRA Supervised Fine-Tuning with resume-safe training."""

import argparse
import json
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ai_infant.data import Store


class ResumeSafeTrainer:
    """LoRA trainer with resume-safe functionality and checkpointing."""

    def __init__(
        self,
        store: Store,
        base_model: str = "microsoft/DialoGPT-small",
        output_dir: str = "adapters",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        learning_rate: float = 5e-5,
        max_steps: int = 1000,
        checkpoint_steps: int = 100,
        seed: int = 42,
    ):
        """Initialize trainer with configuration."""
        self.store = store
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.checkpoint_steps = checkpoint_steps
        self.seed = seed

        # Training state
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.checkpoint_path = None

        # Set random seeds
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.interrupted = False

    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully."""
        print(f"\nReceived signal {signum}, saving checkpoint and exiting...")
        self.interrupted = True
        if self.trainer:
            self.trainer.save_checkpoint()

    def load_dataset(self, dataset_path: Path) -> Dataset:
        """Load and prepare dataset from JSONL file."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Load JSONL data
        data = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        if not data:
            raise ValueError("Dataset is empty")

        # Convert to HuggingFace Dataset
        return Dataset.from_list(data)

    def prepare_model_and_tokenizer(self) -> None:
        """Load and prepare model and tokenizer with LoRA configuration."""
        print(f"Loading base model: {self.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["c_attn"],  # DialoGPT uses c_attn for attention
            bias="none",
            inference_mode=False,
            fan_in_fan_out=True,  # Required for Conv1D layers in DialoGPT
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """Tokenize input/output pairs for training."""
        # Combine input and output with separator for instruction tuning
        texts = []
        for input_text, output_text in zip(examples["input"], examples["output"]):
            combined = (
                f"Input: {input_text}\nOutput: {output_text}{self.tokenizer.eos_token}"
            )
            texts.append(combined)

        # Tokenize
        tokenized = self.tokenizer(
            texts, truncation=True, padding=True, max_length=512, return_tensors=None
        )

        # Set labels to input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in output directory."""
        if not self.output_dir.exists():
            return None

        checkpoints = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith(PREFIX_CHECKPOINT_DIR):
                try:
                    # Extract number after "checkpoint-"
                    checkpoint_num = int(item.name[len(PREFIX_CHECKPOINT_DIR) + 1 :])
                    checkpoints.append((checkpoint_num, str(item)))
                except ValueError:
                    continue

        if not checkpoints:
            return None

        # Return the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: x[0])
        return latest_checkpoint[1]

    def train(self, dataset_path: Path) -> str:
        """Train LoRA adapter with resume-safe functionality."""
        print("Starting LoRA training...")

        # Load dataset
        dataset = self.load_dataset(dataset_path)
        print(f"Loaded dataset with {len(dataset)} examples")

        # Prepare model and tokenizer
        self.prepare_model_and_tokenizer()

        # Tokenize dataset with a standalone function to avoid hashing issues
        def standalone_tokenize(examples):
            """Standalone tokenization function to avoid hashing issues."""
            texts = []
            for input_text, output_text in zip(examples["input"], examples["output"]):
                combined = f"Input: {input_text}\nOutput: {output_text}{self.tokenizer.eos_token}"
                texts.append(combined)

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors=None,
            )

            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(
            standalone_tokenize, batched=True, remove_columns=dataset.column_names
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Check for existing checkpoint
        resume_from_checkpoint = self.find_latest_checkpoint()
        if resume_from_checkpoint:
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            max_steps=self.max_steps,
            per_device_train_batch_size=1,
            learning_rate=self.learning_rate,
            warmup_steps=0,
            logging_steps=1,
            save_steps=self.checkpoint_steps,
            save_total_limit=3,
            report_to=None,
            seed=self.seed,
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # Start training
        start_time = time.time()
        try:
            print(f"Starting training with {len(tokenized_dataset)} tokenized examples")
            print(
                f"Sample tokenized example: {tokenized_dataset[0] if len(tokenized_dataset) > 0 else 'None'}"
            )
            train_result = self.trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
            )
            training_time = time.time() - start_time

            # Save final model
            final_model_path = self.output_dir / "cand.pt"
            self.model.save_pretrained(str(final_model_path))
            self.tokenizer.save_pretrained(str(final_model_path))

            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Final model saved to: {final_model_path}")

            return str(final_model_path)

        except Exception as e:
            training_time = time.time() - start_time
            print(f"Training failed after {training_time:.2f} seconds: {e}")
            raise

    def log_job(
        self,
        dataset_path: str,
        final_model_path: str,
        training_time: float,
        steps_completed: int,
        checkpoint_paths: List[str],
    ) -> str:
        """Log training job."""
        job_id = f"train-{int(time.time() * 1000)}"
        now = datetime.utcnow().isoformat() + "Z"

        job_data = {
            "id": job_id,
            "type": "train",
            "status": "completed",
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
            "input": {
                "dataset_path": dataset_path,
                "base_model": self.base_model,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "learning_rate": self.learning_rate,
                "max_steps": self.max_steps,
                "checkpoint_steps": self.checkpoint_steps,
                "seed": self.seed,
            },
            "output": {
                "final_model_path": final_model_path,
                "training_time_seconds": training_time,
                "steps_completed": steps_completed,
                "checkpoint_paths": checkpoint_paths,
                "model_size_mb": self._get_model_size(final_model_path),
            },
            "error": None,
            "metadata": {
                "version": "0.1.0",
                "priority": 5,
                "retries": 0,
                "max_retries": 3,
                "timeout_seconds": 3600,  # 1 hour timeout
            },
        }

        self.store.store_job(job_data)
        return job_id

    def _get_model_size(self, model_path: str) -> float:
        """Get model size in MB."""
        try:
            total_size = 0
            for file_path in Path(model_path).rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0


def main():
    """Main entry point for LoRA training."""
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter with resume-safe functionality"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        required=True,
        help="Path to training dataset JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="adapters",
        help="Output directory for adapters and checkpoints",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/DialoGPT-small",
        help="Base model for LoRA training",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Maximum training steps"
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--store-path",
        type=str,
        default="data/ai_infant.db",
        help="Path to store database",
    )

    args = parser.parse_args()

    # Initialize store and trainer
    store = Store(args.store_path)
    trainer = ResumeSafeTrainer(
        store=store,
        base_model=args.base_model,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        checkpoint_steps=args.checkpoint_steps,
        seed=args.seed,
    )

    # Train model
    start_time = time.time()
    try:
        final_model_path = trainer.train(args.dataset)
        training_time = time.time() - start_time

        # Get checkpoint paths
        checkpoint_paths = []
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_dir() and item.name.startswith(PREFIX_CHECKPOINT_DIR):
                    checkpoint_paths.append(str(item))

        # Log job
        trainer.log_job(
            dataset_path=str(args.dataset),
            final_model_path=final_model_path,
            training_time=training_time,
            steps_completed=args.max_steps,
            checkpoint_paths=checkpoint_paths,
        )

        print("Training completed successfully")
        print(f"Final model: {final_model_path}")

        return 0

    except Exception as e:
        training_time = time.time() - start_time
        print(f"Training failed: {e}")

        # Log failed job
        job_id = f"train-{int(time.time() * 1000)}"
        now = datetime.utcnow().isoformat() + "Z"

        job_data = {
            "id": job_id,
            "type": "train",
            "status": "failed",
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
            "input": {
                "dataset_path": str(args.dataset),
                "base_model": args.base_model,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "max_steps": args.max_steps,
                "checkpoint_steps": args.checkpoint_steps,
                "seed": args.seed,
            },
            "output": None,
            "error": {
                "type": "training_error",
                "message": str(e),
                "stack": None,
            },
            "metadata": {
                "version": "0.1.0",
                "priority": 5,
                "retries": 0,
                "max_retries": 3,
                "timeout_seconds": 3600,
            },
        }

        store.store_job(job_data)
        return 1


if __name__ == "__main__":
    sys.exit(main())
