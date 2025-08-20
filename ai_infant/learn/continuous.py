"""Continuous learning engine that updates AI model during research."""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data import Store

# Optional pipeline components (jury, dataset selection, SFT, promotion)
try:
    from scripts.promote import PromotionManager
    from scripts.select import DatasetSelector

    from ..learn.eval import LLMJury, create_affordable_jury
    from ..learn.sft import ResumeSafeTrainer
except Exception:  # pragma: no cover - keep import-optional for tests
    create_affordable_jury = None  # type: ignore
    LLMJury = None  # type: ignore
    ResumeSafeTrainer = None  # type: ignore
    DatasetSelector = None  # type: ignore
    PromotionManager = None  # type: ignore


class ContinuousLearner:
    """Engine that continuously updates the AI model during research.

    Two modes:
    - Lightweight online updates (default): local gradient steps on buffered examples.
    - Full pipeline (enable_training_pipeline=True): LLM jury filters examples, dataset
      selection to JSONL, LoRA SFT training, and A/B promotion via jury.
    """

    def __init__(
        self,
        store: Store,
        base_model_path: str = "adapters/cand.pt",
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        max_memory_updates: int = 10,
        enable_training_pipeline: bool = True,
        evaluation_threshold: float = 0.7,
        training_dataset_path: str = "data/training_dataset.jsonl",
    ):
        self.store = store
        self.base_model_path = Path(base_model_path)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_memory_updates = max_memory_updates
        self.enable_training_pipeline = enable_training_pipeline
        self.evaluation_threshold = evaluation_threshold
        self.training_dataset_path = Path(training_dataset_path)

        # Learning state
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.memory_buffer: list[dict[str, Any]] = []
        self.update_count = 0
        self.last_update_time = datetime.utcnow()

        # Pipeline components
        self.jury: Optional[Any] = None
        self.dataset_selector: Optional[Any] = None
        self.trainer: Optional[Any] = None
        self.promotion_manager: Optional[Any] = None

        # Initialize model if available
        self._load_model()

        # Initialize optional training pipeline
        if self.enable_training_pipeline:
            self._initialize_pipeline_components()

    def _initialize_pipeline_components(self) -> None:
        """Initialize jury, dataset selector, trainer, and promotion manager."""
        try:
            if create_affordable_jury:
                self.jury = create_affordable_jury()
            if DatasetSelector:
                self.dataset_selector = DatasetSelector(self.store)
            if ResumeSafeTrainer:
                # Trainer outputs to adapters/ by default per implementation
                self.trainer = ResumeSafeTrainer(store=self.store)
            if PromotionManager and self.jury is not None:
                self.promotion_manager = PromotionManager(
                    store=self.store, jury=self.jury
                )
            print("ContinuousLearner pipeline components initialized")
        except Exception as e:  # pragma: no cover
            print(f"Failed initializing training pipeline components: {e}")

    def _load_model(self):
        """Load the base model and tokenizer."""
        try:
            if self.base_model_path.exists():
                print(f"Loading model from {self.base_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.base_model_path)
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.base_model_path),
                    torch_dtype=(
                        torch.float16 if torch.cuda.is_available() else torch.float32
                    ),
                    device_map="auto" if torch.cuda.is_available() else None,
                )

                # Set up optimizer for continuous learning
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
                )

                print("Model loaded successfully for continuous learning")
            else:
                print(f"Model not found at {self.base_model_path}, will use base model")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def add_learning_example(
        self,
        input_text: str,
        output_text: str,
        confidence: float,
        source_url: str,
        thought_id: str,
    ):
        """Add a learning example to the memory buffer."""
        example: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "input_text": input_text,
            "output_text": output_text,
            "confidence": confidence,
            "source_url": source_url,
            "thought_id": thought_id,
            "timestamp": datetime.utcnow(),
            "used_for_training": False,
            "jury_score": None,
        }

        # If enabled, score example with LLM jury to gate training quality
        if self.enable_training_pipeline and self.jury is not None:
            try:
                jury_result = self.jury.evaluate(
                    prompt=input_text,
                    response=output_text,
                    context=source_url,
                )
                example["jury_score"] = float(jury_result.candidate_score)
            except Exception as e:  # pragma: no cover
                print(f"Jury evaluation failed for example: {e}")
                example["jury_score"] = 0.0

        self.memory_buffer.append(example)

        # Log the learning example
        self._log_learning_example(example)

        # Check if we should update the model
        if len(self.memory_buffer) >= self.batch_size:
            self._consider_model_update()

    def _log_learning_example(self, example: dict[str, Any]):
        """Log a learning example."""
        print("\n📚 LEARNING: New example added")
        print(f"   Input: {example['input_text'][:100]}...")
        print(f"   Output: {example['output_text'][:100]}...")
        print(f"   Confidence: {example['confidence']:.2f}")
        print(f"   Source: {example['source_url']}")
        print()

        # Store in database
        self.store.store_trace(
            {
                "id": f"learning-{example['id']}",
                "job_id": f"learning-{int(time.time() * 1000)}",
                "component": "continuous_learning",
                "operation": "add_example",
                "status": "completed",
                "timestamp": example["timestamp"].isoformat() + "Z",
                "duration_ms": 0,
                "input": {
                    "input_text": example["input_text"],
                    "confidence": example["confidence"],
                    "source_url": example["source_url"],
                },
                "output": {
                    "example_id": example["id"],
                    "buffer_size": len(self.memory_buffer),
                },
                "metadata": {"learning_example": True},
            }
        )

    def _consider_model_update(self):
        """Consider whether to update the model based on various factors."""
        if not self.model or not self.optimizer:
            return

        # Check if we've reached max updates
        if self.update_count >= self.max_memory_updates:
            print("Maximum model updates reached")
            return

        # Check time since last update
        time_since_update = (datetime.utcnow() - self.last_update_time).total_seconds()
        if time_since_update < 300:  # 5 minutes minimum between updates
            return

        # Check if we have enough high-confidence (+jury) examples
        high_confidence_examples = []
        for ex in self.memory_buffer:
            if ex["used_for_training"]:
                continue
            meets_conf = ex["confidence"] > 0.7
            if self.enable_training_pipeline and ex.get("jury_score") is not None:
                meets_conf = meets_conf and (
                    ex["jury_score"] >= self.evaluation_threshold
                )
            if meets_conf:
                high_confidence_examples.append(ex)

        if len(high_confidence_examples) >= self.batch_size:
            if self.enable_training_pipeline and self._pipeline_ready():
                self._run_training_pipeline(high_confidence_examples)
            else:
                self._update_model(high_confidence_examples)

    def _update_model(self, examples: list[dict[str, Any]]):
        """Update the model with new examples."""
        if not self.model or not self.optimizer:
            return

        print(f"\n🔄 UPDATING MODEL with {len(examples)} examples")

        try:
            # Prepare training data
            training_data = []
            for example in examples:
                # Format as instruction-following data
                formatted_text = (
                    f"Input: {example['input_text']}\nOutput: {example['output_text']}"
                )
                training_data.append(formatted_text)

            # Tokenize batch
            tokenized = self.tokenizer(
                training_data,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move to device
            if torch.cuda.is_available():
                tokenized = {k: v.cuda() for k, v in tokenized.items()}

            # Forward pass
            self.model.train()
            outputs = self.model(**tokenized, labels=tokenized["input_ids"])
            loss = outputs.loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update state
            self.update_count += 1
            self.last_update_time = datetime.utcnow()

            # Mark examples as used
            for example in examples:
                example["used_for_training"] = True

            # Log the update
            self._log_model_update(loss.item(), len(examples))

            # Save checkpoint periodically
            if self.update_count % 5 == 0:
                self._save_checkpoint()

            print(f"Model updated successfully. Loss: {loss.item():.4f}")

        except Exception as e:
            print(f"Model update failed: {e}")
            self._log_model_update_error(str(e))

    def _pipeline_ready(self) -> bool:
        """Check if training pipeline components are available."""
        return (
            self.jury is not None
            and self.dataset_selector is not None
            and self.trainer is not None
            and self.promotion_manager is not None
        )

    def _run_training_pipeline(self, examples: list[dict[str, Any]]) -> None:
        """Run selection -> SFT -> evaluation/promotion on buffered examples."""
        try:
            # 1) Build/update a JSONL dataset from high-quality buffered examples
            dataset_items = [
                {"input": ex["input_text"], "output": ex["output_text"]}
                for ex in examples
            ]
            self.training_dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.training_dataset_path, "w", encoding="utf-8") as f:
                for item in dataset_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            # 2) Train LoRA adapter (resume-safe)
            final_model_path = self.trainer.train(self.training_dataset_path)

            # 3) Evaluate candidate vs incumbent with jury
            # Use a representative example (highest jury score or confidence)
            rep = max(
                examples,
                key=lambda ex: (ex.get("jury_score") or 0.0, ex["confidence"]),
            )

            jury_decision = self.promotion_manager.promote_candidate(
                model_path=str(final_model_path),
                prompt=rep["input_text"],
                response=rep["output_text"],
                context=rep.get("source_url"),
                seed=42,
            )

            # 4) If promoted, reload as new base for online updates
            if jury_decision.get("promoted"):
                self.base_model_path = Path(final_model_path)
                self._load_model()

            # Mark examples as used
            for ex in examples:
                ex["used_for_training"] = True

            # Log training/promotion as a model update event
            self._log_model_update(loss=0.0, example_count=len(examples))

        except (
            Exception
        ) as e:  # pragma: no cover - robust in absence of APIs during tests
            print(f"Training pipeline failed: {e}")
            self._log_model_update_error(str(e))

    def _log_model_update(self, loss: float, example_count: int):
        """Log a model update."""
        self.store.store_trace(
            {
                "id": f"model-update-{int(time.time() * 1000)}",
                "job_id": f"model-update-{int(time.time() * 1000)}",
                "component": "continuous_learning",
                "operation": "model_update",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "duration_ms": 0,
                "input": {
                    "example_count": example_count,
                    "learning_rate": self.learning_rate,
                },
                "output": {
                    "loss": loss,
                    "update_count": self.update_count,
                    "buffer_size": len(self.memory_buffer),
                },
                "metadata": {"model_update": True},
            }
        )

    def _log_model_update_error(self, error: str):
        """Log a model update error."""
        self.store.store_trace(
            {
                "id": f"model-update-error-{int(time.time() * 1000)}",
                "job_id": f"model-update-error-{int(time.time() * 1000)}",
                "component": "continuous_learning",
                "operation": "model_update",
                "status": "failed",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "duration_ms": 0,
                "input": {},
                "output": {},
                "error": {
                    "type": "model_update_error",
                    "message": error,
                    "stack": None,
                },
                "metadata": {"model_update": True},
            }
        )

    def _save_checkpoint(self):
        """Save a checkpoint of the updated model."""
        try:
            checkpoint_dir = Path("adapters/continuous_learning")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"checkpoint_{self.update_count}.pt"

            # Save model state
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "update_count": self.update_count,
                    "learning_rate": self.learning_rate,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                checkpoint_path,
            )

            # Save tokenizer
            tokenizer_path = checkpoint_dir / f"tokenizer_{self.update_count}"
            self.tokenizer.save_pretrained(tokenizer_path)

            print(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def get_learning_stats(self) -> dict[str, Any]:
        """Get statistics about the learning process."""
        return {
            "model_loaded": self.model is not None,
            "update_count": self.update_count,
            "buffer_size": len(self.memory_buffer),
            "unused_examples": len(
                [ex for ex in self.memory_buffer if not ex["used_for_training"]]
            ),
            "high_confidence_examples": len(
                [ex for ex in self.memory_buffer if ex["confidence"] > 0.7]
            ),
            "last_update": self.last_update_time.isoformat(),
            "learning_rate": self.learning_rate,
            "max_updates": self.max_memory_updates,
        }

    def force_update(self):
        """Force a model update with current buffer."""
        if self.memory_buffer:
            unused_examples = [
                ex for ex in self.memory_buffer if not ex["used_for_training"]
            ]
            if unused_examples:
                self._update_model(unused_examples)

    def clear_buffer(self):
        """Clear the memory buffer."""
        self.memory_buffer.clear()
        print("Learning buffer cleared")

    def export_learning_trace(self) -> dict[str, Any]:
        """Export the complete learning trace."""
        return {
            "continuous_learning_session": {
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "stats": self.get_learning_stats(),
                "examples": self.memory_buffer,
                "model_path": str(self.base_model_path),
            }
        }
