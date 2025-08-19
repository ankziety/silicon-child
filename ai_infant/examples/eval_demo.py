#!/usr/bin/env python3
"""Demonstration of the LLM Jury evaluation system."""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup - noqa: E402
from ai_infant.data.store import Store  # noqa: E402
from ai_infant.learn.eval import (  # noqa: E402
    create_affordable_jury,
    create_diverse_jury,
    create_frontier_jury,
    create_mixed_jury,
    create_specialized_jury,
)
from scripts.promote import PromotionManager  # noqa: E402


def check_api_keys():
    """Check if required API keys are set."""
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY"]
    missing_keys = []

    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)

    if missing_keys:
        print("❌ Missing required API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease set these environment variables or add them to a .env file")
        print("Example .env file:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        print("ANTHROPIC_API_KEY=your_anthropic_api_key_here")
        print("COHERE_API_KEY=your_cohere_api_key_here")
        return False

    print("✅ All required API keys are set")
    return True


def main():
    """Run the LLM Jury evaluation demonstration."""
    print("AI-Infant LLM Jury Evaluation System")
    print("=" * 50)

    # Check API keys
    if not check_api_keys():
        return

    try:
        # Initialize components
        print("\nInitializing evaluation system...")
        store = Store(":memory:")  # Use in-memory database for demo

        # Create different jury configurations
        frontier_jury = create_frontier_jury()
        diverse_jury = create_diverse_jury()
        affordable_jury = create_affordable_jury()
        specialized_jury = create_specialized_jury()
        mixed_jury = create_mixed_jury()

        print(f"Frontier jury has {len(frontier_jury.judges)} judges:")
        for judge in frontier_jury.judges:
            print(f"  - {judge.name} ({judge.evaluation_type})")

        print(f"\nAffordable jury has {len(affordable_jury.judges)} judges:")
        for judge in affordable_jury.judges:
            print(f"  - {judge.name} ({judge.evaluation_type})")

        # Sample evaluation data
        prompt = "What is the capital of France and what makes it significant?"
        response = "The capital of France is Paris. It is significant as a major global center for art, fashion, gastronomy, and culture. Paris is home to iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It has been a center of political power, intellectual thought, and artistic innovation for centuries."
        context = "This is a geography and history question about European capitals."

        # Test different jury configurations
        juries = [
            ("Frontier Jury (GPT-5 + Claude Sonnet)", frontier_jury),
            ("Diverse Jury (Mixed Models)", diverse_jury),
            ("Affordable Jury (Cost-Effective)", affordable_jury),
            ("Specialized Jury (GPT-5 Only)", specialized_jury),
            ("Mixed Jury (High + Affordable)", mixed_jury),
        ]

        for jury_name, jury in juries:
            print(f"\n{'=' * 20} {jury_name} {'=' * 20}")

            try:
                # Run evaluation
                print(f"Running {jury_name} evaluation...")
                result = jury.evaluate(prompt, response, context, seed=42)

                print("✅ Evaluation completed!")
                print(f"Overall score: {result.candidate_score:.4f}")
                print(f"Aggregation method: {result.aggregation_method}")
                print(f"Seed used: {result.seed}")

                print("\nIndividual judge results:")
                for judge_result in result.judge_results:
                    print(f"  {judge_result.judge_name}: {judge_result.score:.4f}")
                    print(f"    Reasoning: {judge_result.reasoning[:100]}...")
                    print(f"    Model: {judge_result.metadata.get('model', 'Unknown')}")

            except Exception as e:
                print(f"❌ {jury_name} evaluation failed: {e}")

        # Test promotion system with frontier jury
        print(f"\n{'=' * 20} Promotion System {'=' * 20}")

        manager = PromotionManager(store, frontier_jury)

        print("Testing promotion with frontier jury...")
        promotion_result = manager.promote_candidate(
            model_path="demo_model_v1",
            prompt=prompt,
            response=response,
            context=context,
            seed=42,
        )

        if promotion_result["promoted"]:
            print("✅ Candidate promoted!")
            print(f"New incumbent score: {promotion_result['candidate_score']:.4f}")
        else:
            print("❌ Candidate not promoted")
            print(f"Candidate score: {promotion_result['candidate_score']:.4f}")
            print(f"Incumbent score: {promotion_result['incumbent_score']:.4f}")

        print(f"Total adapters in history: {promotion_result['adapters_count']}")

        # Show adapter history
        history = manager.get_adapter_history()
        if history:
            print("\nAdapter history:")
            for i, adapter in enumerate(history):
                print(
                    f"  {i + 1}. {adapter['model_path']} (score: {adapter['score']:.4f})"
                )

        # Test failure case
        print(f"\n{'=' * 20} Error Handling {'=' * 20}")
        print("Testing error handling with invalid response...")

        try:
            manager.evaluate_candidate(
                prompt="Test prompt",
                response="",  # Empty response should cause issues
                context="Test context",
                seed=42,
            )
        except Exception as e:
            print(f"✅ Expected failure caught: {type(e).__name__}: {e}")

        print("\n✅ LLM Jury demonstration completed successfully!")

        # Show cost comparison
        print(f"\n{'=' * 20} Cost Comparison {'=' * 20}")
        print("Model pricing (approximate per 1K tokens):")
        print("  GPT-4o-mini: $0.00015 (input) / $0.0006 (output)")
        print("  GPT-5: $0.005 (input) / $0.015 (output)")
        print("  Claude Haiku: $0.00025 (input) / $0.00125 (output)")
        print("  Claude Sonnet: $0.003 (input) / $0.015 (output)")
        print("  Command R+: $0.0005 (input) / $0.0015 (output)")
        print("\nAffordable jury uses ~70% less cost than frontier jury")

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
