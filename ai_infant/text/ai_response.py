"""AI response generator for contextual, learning-based conversations."""

import os
from typing import Any, Optional

from pydantic import BaseModel

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class LearningContext(BaseModel):
    """Context about the AI-Infant's learning state."""

    recent_jobs: list[dict[str, Any]]
    recent_quotes: list[dict[str, Any]]
    recent_discoveries: list[dict[str, Any]]
    learning_topics: list[str]
    success_rate: float
    total_research_sessions: int


class AIResponseGenerator:
    """Generates contextual AI responses based on learning data."""

    def __init__(self, store: Any):
        """Initialize AI response generator."""
        self.store = store
        self.llm_client = self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client based on available API keys."""
        # Try OpenAI first
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                print("Using OpenAI API for AI responses")
                return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                print("OpenAI library not available")

        # Try Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic

                print("Using Anthropic API for AI responses")
                return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                print("Anthropic library not available")

        # Try Cohere
        if os.getenv("COHERE_API_KEY"):
            try:
                import cohere

                print("Using Cohere API for AI responses")
                return cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
            except ImportError:
                print("Cohere library not available")

        # Fallback to local model if available
        try:
            from transformers import pipeline

            print("Using local GPT-2 model (limited capabilities)")
            return pipeline("text-generation", model="gpt2", device=-1)
        except ImportError:
            print("No AI models available - using fallback responses")
            return None

    def _get_learning_context(self) -> LearningContext:
        """Get current learning context from the store."""
        try:
            recent_jobs = self.store.get_recent_jobs(limit=20)

            # Extract learning data
            research_jobs = [
                job
                for job in recent_jobs
                if job["type"] in ["research", "parse", "fetch"]
            ]
            successful_jobs = [
                job for job in research_jobs if job["status"] == "completed"
            ]

            # Extract quotes and discoveries
            recent_quotes = []
            recent_discoveries = []
            learning_topics = []

            for job in successful_jobs:
                if "output" in job and job["output"]:
                    if "quotes" in job["output"]:
                        recent_quotes.extend(job["output"]["quotes"])
                    if "quotes_count" in job["output"]:
                        recent_discoveries.append(
                            {
                                "topic": job["input"].get("question", "unknown"),
                                "quotes_found": job["output"]["quotes_count"],
                                "timestamp": job["created_at"],
                            }
                        )

                # Extract topics
                if "input" in job:
                    if "question" in job["input"]:
                        learning_topics.append(job["input"]["question"])
                    elif "url" in job["input"]:
                        learning_topics.append(f"webpage: {job['input']['url']}")

            success_rate = (
                len(successful_jobs) / len(research_jobs) if research_jobs else 0.0
            )

            return LearningContext(
                recent_jobs=recent_jobs,
                recent_quotes=recent_quotes[:10],  # Last 10 quotes
                recent_discoveries=recent_discoveries[:5],  # Last 5 discoveries
                learning_topics=learning_topics[:10],  # Last 10 topics
                success_rate=success_rate,
                total_research_sessions=len(research_jobs),
            )

        except Exception:
            # Return minimal context if there's an error
            return LearningContext(
                recent_jobs=[],
                recent_quotes=[],
                recent_discoveries=[],
                learning_topics=[],
                success_rate=0.0,
                total_research_sessions=0,
            )

    def _generate_ai_response(self, prompt: str, context: LearningContext) -> str:
        """Generate AI response using available LLM."""
        if not self.llm_client:
            return self._fallback_response(prompt, context)

        try:
            # Build context-aware prompt
            full_prompt = self._build_context_prompt(prompt, context)

            # Generate response based on LLM type
            if hasattr(self.llm_client, "chat"):  # OpenAI
                response = self.llm_client.chat.completions.create(
                    model="gpt-5-mini",  # Default to cheaper OpenAI model
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI-Infant, a curious learning AI that explores and discovers new information. You must respond with valid XML that matches the requested format exactly.",
                        },
                        {"role": "user", "content": full_prompt},
                    ],
                    max_tokens=2000,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()

            elif hasattr(self.llm_client, "messages"):  # Anthropic
                response = self.llm_client.messages.create(
                    model="claude-3-5-sonnet-20241022",  # Latest Claude model
                    max_tokens=2000,
                    temperature=0.3,
                    system="You are an AI-Infant, a curious learning AI that explores and discovers new information. You must respond with valid XML that matches the requested format exactly.",
                    messages=[{"role": "user", "content": full_prompt}],
                )
                return response.content[0].text.strip()

            elif hasattr(self.llm_client, "generate"):  # Cohere
                response = self.llm_client.generate(
                    model="command-r-plus",  # Latest Cohere model
                    prompt=full_prompt,
                    max_tokens=2000,
                    temperature=0.3,
                )
                return response.generations[0].text.strip()

            else:  # Local model
                response = self.llm_client(
                    full_prompt, max_length=2000, temperature=0.3
                )
                return response[0]["generated_text"].strip()

        except Exception:
            return self._fallback_response(prompt, context)

    def _build_context_prompt(self, user_input: str, context: LearningContext) -> str:
        """Build a context-aware prompt for the AI."""
        prompt_parts = [
            "You are an AI-Infant, a curious learning AI. Respond naturally to the user's input.",
            f"User input: {user_input}",
        ]

        # Add learning context
        if context.learning_topics:
            topics_text = ", ".join(context.learning_topics[:5])
            prompt_parts.append(f"Recently, you've been learning about: {topics_text}")

        if context.recent_discoveries:
            discoveries_text = ", ".join(
                [d["topic"] for d in context.recent_discoveries[:3]]
            )
            prompt_parts.append(f"Your recent discoveries include: {discoveries_text}")

        if context.recent_quotes:
            quote_text = (
                context.recent_quotes[0]["text"][:100] + "..."
                if len(context.recent_quotes[0]["text"]) > 100
                else context.recent_quotes[0]["text"]
            )
            prompt_parts.append(f'One interesting thing you learned: "{quote_text}"')

        prompt_parts.append(f"Your learning success rate: {context.success_rate:.1%}")
        prompt_parts.append(
            f"Total research sessions: {context.total_research_sessions}"
        )

        prompt_parts.append(
            "Respond naturally, incorporating your learning experiences and showing genuine curiosity and enthusiasm for discovery."
        )

        return "\n".join(prompt_parts)

    def _fallback_response(self, user_input: str, context: LearningContext) -> str:
        """Fallback response when AI is not available."""
        user_input_lower = user_input.lower()

        if any(
            word in user_input_lower for word in ["learning", "learned", "discovered"]
        ):
            if context.learning_topics:
                return f"I've been learning about {context.learning_topics[0]}! It's fascinating how much there is to discover."
            else:
                return "I'm ready to learn new things! What should I explore?"

        elif any(word in user_input_lower for word in ["curious", "wonder"]):
            if context.recent_discoveries:
                return f"I'm curious about how {context.recent_discoveries[0]['topic']} connects to other things I'm learning!"
            else:
                return "I'm naturally curious about everything! What interests you?"

        else:
            return "I'm here to learn and explore! What would you like me to discover?"

    def generate_response(self, user_input: str) -> str:
        """Generate a contextual AI response based on learning data."""
        context = self._get_learning_context()
        return self._generate_ai_response(user_input, context)

    def generate_direct_response(self, prompt: str) -> str:
        """Generate a direct AI response without learning context."""
        # Try each available API in order
        apis_to_try = [
            ("OpenAI", self._try_openai),
            ("Anthropic", self._try_anthropic),
            ("Cohere", self._try_cohere),
            ("Local", self._try_local),
        ]

        for api_name, api_func in apis_to_try:
            try:
                result = api_func(prompt)
                if result and result != "AI response generation failed":
                    return result
            except Exception as e:
                print(f"{api_name} API failed: {e}")
                continue

        return "AI response generation failed"

    def _try_openai(self, prompt: str) -> str:
        """Try OpenAI API."""
        if not os.getenv("OPENAI_API_KEY"):
            raise Exception("OpenAI API key not available")

        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-5-mini",  # Default to cheaper OpenAI model
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI research assistant. You must respond with valid XML that exactly matches the requested format. Do not include any text outside the XML structure. Ensure all XML tags are properly closed and the response is parseable.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}") from e

    def _try_anthropic(self, prompt: str) -> str:
        """Try Anthropic API."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise Exception("Anthropic API key not available")

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Latest Claude model
                max_tokens=2000,
                temperature=0.3,
                system="You are an AI research assistant. You must respond with valid XML that exactly matches the requested format. Do not include any text outside the XML structure. Ensure all XML tags are properly closed and the response is parseable.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}") from e

    def _try_cohere(self, prompt: str) -> str:
        """Try Cohere API."""
        if not os.getenv("COHERE_API_KEY"):
            raise Exception("Cohere API key not available")

        try:
            import cohere

            client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
            response = client.generate(
                model="command-r-plus",  # Latest Cohere model
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3,
            )
            return response.generations[0].text.strip()
        except Exception as e:
            raise Exception(f"Cohere API error: {e}") from e

    def _try_local(self, prompt: str) -> str:
        """Try local model."""
        try:
            from transformers import pipeline

            generator = pipeline("text-generation", model="gpt2", device=-1)
            response = generator(prompt, max_length=2000, temperature=0.3)
            return response[0]["generated_text"].strip()
        except Exception as e:
            raise Exception(f"Local model error: {e}") from e

    def generate_learning_summary(self) -> str:
        """Generate a summary of recent learning."""
        context = self._get_learning_context()

        if not context.learning_topics:
            return "I'm just getting started with my learning journey!"

        prompt = f"Summarize your recent learning experiences. You've been exploring {len(context.learning_topics)} topics and made {len(context.recent_discoveries)} discoveries."
        return self._generate_ai_response(prompt, context)

    def generate_curiosity_response(self, topic: Optional[str] = None) -> str:
        """Generate a curiosity-driven response."""
        context = self._get_learning_context()

        if topic:
            prompt = f"Express your curiosity about {topic} and how it relates to your recent learning."
        else:
            prompt = "Express your natural curiosity and what you're most interested in learning about next."

        return self._generate_ai_response(prompt, context)
