"""Command-line interface for AI-Infant research agent."""

import argparse
import sys
from pathlib import Path

from .core.loop import ResearchLoop
from .data.store import Store


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Infant: Self-curious research agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start self-curious research (default behavior)
  python -m ai_infant
  
  # Run self-curious research for specific duration
  python -m ai_infant --duration 30
  
  # Run research on a specific question
  python -m ai_infant research "What are the latest developments in quantum computing?"
  
  # Run a timed session with predefined questions
  python -m ai_infant session --duration 15
  
  # Run with custom parameters
  python -m ai_infant research "How does machine learning work?" --max-iterations 10 --min-quotes 5
        """
    )
    
    # Default self-curious mode arguments
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Session duration in minutes (default: 60)"
    )
    parser.add_argument(
        "--db-path",
        default="data/ai_infant.db",
        help="Database path (default: data/ai_infant.db)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Research command
    research_parser = subparsers.add_parser(
        "research", 
        help="Run research on a specific question"
    )
    research_parser.add_argument(
        "question",
        help="Research question to investigate"
    )
    research_parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum research iterations (default: 20)"
    )
    research_parser.add_argument(
        "--min-quotes",
        type=int,
        default=3,
        help="Minimum quotes required (default: 3)"
    )
    research_parser.add_argument(
        "--db-path",
        default="data/ai_infant.db",
        help="Database path (default: data/ai_infant.db)"
    )
    
    # Session command
    session_parser = subparsers.add_parser(
        "session",
        help="Run a timed research session with predefined questions"
    )
    session_parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="Session duration in minutes (default: 15)"
    )
    session_parser.add_argument(
        "--questions",
        nargs="+",
        default=[
            "What are the latest developments in artificial intelligence?",
            "How does quantum computing work?",
            "What are the benefits of renewable energy?",
            "How do neural networks function?",
            "What is the future of blockchain technology?"
        ],
        help="List of questions to research during session"
    )
    session_parser.add_argument(
        "--db-path",
        default="data/ai_infant.db",
        help="Database path (default: data/ai_infant.db)"
    )
    
    args = parser.parse_args()
    
    # Initialize store
    store = Store(args.db_path)
    
    try:
        if args.command == "research":
            run_research(store, args.question, args.max_iterations, args.min_quotes)
        elif args.command == "session":
            run_session(store, args.duration, args.questions)
        else:
            # Default behavior: self-curious mode
            run_curious(store, args.duration)
    finally:
        store.close()


def run_research(store, question: str, max_iterations: int, min_quotes: int):
    """Run research on a specific question."""
    print(f"Starting research on: {question}")
    print(f"Max iterations: {max_iterations}, Min quotes: {min_quotes}")
    print("-" * 60)
    
    loop = ResearchLoop(store)
    try:
        answer = loop.research(
            question=question,
            max_iterations=max_iterations,
            min_quotes=min_quotes
        )
        
        if answer:
            print(f"\nResearch completed!")
            print(f"Answer: {answer.answer}")
            print(f"Quotes found: {len(answer.quotes)}")
            print(f"Documents used: {len(answer.documents_used)}")
            print(f"Iterations: {answer.trace_id}")
        else:
            print("Research failed")
    finally:
        loop.close()


def run_session(store, duration_minutes: int, questions: list[str]):
    """Run a timed research session."""
    import time
    import random
    
    print(f"Starting {duration_minutes}-minute research session")
    print(f"Questions available: {len(questions)}")
    print("-" * 60)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    loop = ResearchLoop(store)
    questions_researched = 0
    
    try:
        while time.time() < end_time:
            # Select a random question
            question = random.choice(questions)
            print(f"\nResearching: {question}")
            
            # Let the system decide how much to research
            answer = loop.research_autonomously(question)
            
            if answer:
                print(f"Completed: {len(answer.quotes)} quotes found")
                questions_researched += 1
            else:
                print("Failed")
            
            # Check if we have time for more
            remaining_time = end_time - time.time()
            if remaining_time < 60:  # Less than 1 minute left
                break
        
        elapsed_time = time.time() - start_time
        print(f"\nSession completed!")
        print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        print(f"Questions researched: {questions_researched}")
    finally:
        loop.close()


def run_curious(store, duration_minutes: int):
    """Run in self-curious mode, generating own research questions."""
    import time
    
    print(f"Starting {duration_minutes}-minute self-curious research session")
    print("Letting the AI-Infant explore and research autonomously...")
    print("-" * 60)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    loop = ResearchLoop(store)
    questions_researched = 0
    
    try:
        # Initial curiosity questions to bootstrap autonomous exploration
        initial_questions = [
            "What are the most important discoveries in the last decade?",
            "How do emerging technologies work?",
            "What are the biggest challenges facing humanity?",
            "What breakthroughs are happening in research?",
            "What new developments are occurring in science?",
            "How do complex systems function?",
            "What are the latest innovations?",
            "How do different fields connect?",
            "What problems are being solved?",
            "What new knowledge is being created?"
        ]
        
        # Let the system explore naturally
        for question in initial_questions:
            if time.time() >= end_time:
                break
                
            print(f"\nResearching: {question}")
            
            # Let the system decide how much to research
            answer = loop.research_autonomously(question)
            
            if answer:
                print(f"Completed: {len(answer.quotes)} quotes found")
                questions_researched += 1
                
                # Generate follow-up questions based on discovered content
                follow_up_questions = generate_follow_up_questions(answer)
                
                # Let it explore follow-ups naturally
                for follow_up in follow_up_questions:
                    if time.time() >= end_time:
                        break
                        
                    print(f"\nFollowing up: {follow_up}")
                    
                    follow_up_answer = loop.research_autonomously(follow_up)
                    
                    if follow_up_answer:
                        print(f"Follow-up completed: {len(follow_up_answer.quotes)} quotes found")
                        questions_researched += 1
                    else:
                        print("Follow-up failed")
            else:
                print("Failed")
        
        elapsed_time = time.time() - start_time
        print(f"\nSelf-curious session completed!")
        print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        print(f"Questions researched: {questions_researched}")
    finally:
        loop.close()


def generate_follow_up_questions(answer):
    """Generate follow-up questions based on research findings."""
    # Simple heuristic-based question generation
    # In a production system, this would use LLM-based question generation
    
    follow_ups = []
    
    # Extract key topics from the answer
    answer_text = answer.answer.lower()
    
    # Look for topics mentioned in the research
    topics = []
    if "technology" in answer_text or "innovation" in answer_text:
        topics.append("technology")
    if "science" in answer_text or "research" in answer_text:
        topics.append("science")
    if "discovery" in answer_text or "breakthrough" in answer_text:
        topics.append("discoveries")
    if "problem" in answer_text or "challenge" in answer_text:
        topics.append("problems")
    if "system" in answer_text or "function" in answer_text:
        topics.append("systems")
    if "development" in answer_text or "progress" in answer_text:
        topics.append("development")
    if "field" in answer_text or "area" in answer_text:
        topics.append("fields")
    if "application" in answer_text or "use" in answer_text:
        topics.append("applications")
    
    # Generate follow-up questions for discovered topics
    for topic in topics[:3]:  # Limit to 3 topics
        if topic == "technology":
            follow_ups.extend([
                "How is this technology being applied?",
                "What are the implications of this technology?",
                "How does this connect to other technologies?"
            ])
        elif topic == "science":
            follow_ups.extend([
                "What are the latest developments in this field?",
                "How does this research connect to other areas?",
                "What questions remain unanswered?"
            ])
        elif topic == "discoveries":
            follow_ups.extend([
                "What led to this discovery?",
                "How will this discovery be used?",
                "What are the broader implications?"
            ])
        elif topic == "problems":
            follow_ups.extend([
                "What solutions are being developed?",
                "How are people approaching this challenge?",
                "What progress has been made?"
            ])
        elif topic == "systems":
            follow_ups.extend([
                "How do the components interact?",
                "What makes this system work?",
                "How is this system evolving?"
            ])
        elif topic == "development":
            follow_ups.extend([
                "What's driving this progress?",
                "What obstacles remain?",
                "How will this develop further?"
            ])
        elif topic == "fields":
            follow_ups.extend([
                "How do different fields connect?",
                "What's happening at the boundaries?",
                "What new areas are emerging?"
            ])
        elif topic == "applications":
            follow_ups.extend([
                "How is this being used in practice?",
                "What new applications are possible?",
                "How will this impact society?"
            ])
    
    # Add some general curiosity questions if we don't have enough
    if len(follow_ups) < 2:
        follow_ups.extend([
            "What are the implications of this?",
            "How does this connect to other areas?",
            "What questions does this raise?",
            "How will this develop further?",
            "What are the broader impacts?"
        ])
    
    return follow_ups[:3]  # Return max 3 follow-up questions


if __name__ == "__main__":
    main()
