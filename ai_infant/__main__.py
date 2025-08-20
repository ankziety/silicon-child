"""Main entry point for AI-Infant adaptive research agent."""

import argparse
import sys

from .core.adaptive_loop import AdaptiveResearchLoop
from .data import Store


def main():
    """Main entry point for the adaptive research agent."""
    parser = argparse.ArgumentParser(
        description="AI-Infant: Adaptive Research Agent with Reasoning and Learning"
    )
    parser.add_argument("question", help="Research question to investigate")
    parser.add_argument(
        "--store-path",
        type=str,
        default="data/ai_infant.db",
        help="Path to store database",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run browser in headless mode"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=20, help="Maximum research iterations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for research reports",
    )

    args = parser.parse_args()

    # Initialize store
    store = Store(args.store_path)

    # Initialize adaptive research loop
    research_loop = AdaptiveResearchLoop(store, headless=args.headless)
    research_loop.max_iterations = args.max_iterations

    interrupted = False
    try:
        print("🤖 AI-Infant Adaptive Research Agent")
        print("=" * 50)
        print(f"Question: {args.question}")
        print(f"Max iterations: {args.max_iterations}")
        print(f"Headless mode: {args.headless}")
        print("=" * 50)

        # Start research
        session = research_loop.research_question(args.question)

        # Display results
        print("\n" + "=" * 50)
        print("📊 RESEARCH RESULTS")
        print("=" * 50)
        print(f"Session ID: {session.id}")
        print(f"Status: {session.status}")
        print(f"Iterations: {session.total_iterations}")
        print(f"Sources used: {len(session.sources_used)}")
        print(f"Conclusions formed: {len(session.conclusions)}")
        print(f"Learning updates: {session.learning_stats.get('update_count', 0)}")

        print("\n🧠 REASONING SUMMARY:")
        reasoning = session.reasoning_summary
        print(f"  Total thoughts: {reasoning.get('total_thoughts', 0)}")
        print(
            f"  Knowledge gaps: {reasoning.get('knowledge_gaps', {}).get('total', 0)}"
        )
        print(f"  Filled gaps: {reasoning.get('knowledge_gaps', {}).get('filled', 0)}")

        print("\n📚 LEARNING STATS:")
        learning = session.learning_stats
        print(f"  Model loaded: {learning.get('model_loaded', False)}")
        print(f"  Buffer size: {learning.get('buffer_size', 0)}")
        print(
            f"  High confidence examples: {learning.get('high_confidence_examples', 0)}"
        )

        print("\n💡 FINAL ANSWER:")
        print("-" * 30)
        if session.final_answer:
            print(session.final_answer)
        else:
            print("No final answer generated")

        print(f"\n📁 Research report saved to: reports/research_{session.id}.json")

        return 0

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n⏹️  Research interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Research failed: {e}")
        return 1
    finally:
        # Avoid blocking close when interrupted by SIGINT; immediate exit preferred
        try:
            if not interrupted:
                research_loop.close()
            else:
                # If interrupted, attempt a fast teardown without blocking
                try:
                    if hasattr(research_loop, "browser") and getattr(
                        research_loop.browser, "playwright", None
                    ):
                        research_loop.browser.playwright.stop()
                except Exception:
                    pass
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
