"""Vision-based browser automation module using external vision models for intelligent interaction."""

import base64
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import re
import os
import zipfile
from pydantic import BaseModel, Field

from .browser import Browser, PageState, InteractiveElement
from ..text.image_analysis import VisionAnalysis, VisionAction, ImageAnalyzer
from ..text.ai_response import AIResponseGenerator
from ..learn.llm_aggregator import AggregatorManager
from ..learn.eval import create_affordable_jury, LLMJury


class VisionModelConfig(BaseModel):
    """Configuration for vision model integration."""
    
    model_provider: str = Field(description="Vision model provider (openai, anthropic, local)")
    model_name: str = Field(description="Specific model name")
    api_key: Optional[str] = Field(description="API key for the model")
    api_base: Optional[str] = Field(description="API base URL for custom endpoints")
    max_tokens: int = Field(default=1000, description="Maximum tokens for vision analysis")
    temperature: float = Field(default=0.1, description="Temperature for vision analysis")


class VisionBrowserAction(BaseModel):
    """Action executed by vision-based browser automation."""
    
    action_type: str  # click, type, scroll, navigate, wait, screenshot
    target_description: str
    coordinates: Optional[Tuple[int, int]]
    text_input: Optional[str]
    selector: Optional[str]
    confidence: float
    reasoning: str
    timestamp: datetime
    success: bool
    error_message: Optional[str]


class VisionBrowserSession(BaseModel):
    """Session for vision-based browser automation."""
    
    session_id: str
    start_time: datetime
    current_url: str
    actions_performed: List[VisionBrowserAction]
    screenshots_taken: List[str]
    vision_analyses: List[VisionAnalysis]
    user_goal: str
    status: str  # active, completed, failed


class VisionBrowser(Browser):
    """Enhanced browser with vision-based automation capabilities."""

    def __init__(
        self, 
        store: Any, 
        vision_config: Optional[VisionModelConfig] = None,
        user_agent: str = "AI-Infant/0.1.0", 
        headless: bool = False
    ):
        """Initialize vision-based browser."""
        super().__init__(store, user_agent, headless)
        
        # Vision model configuration
        self.vision_config = vision_config or self._get_default_vision_config()
        
        # AI response generator with fallback support
        self.ai_generator = AIResponseGenerator(store)

        # Aggregator manager: required (no fallback allowed)
        try:
            pref = os.getenv("AGGREGATOR_PREFERENCE", "llmz,openrouter")
            self.aggregator = AggregatorManager(preference=pref, enforce_no_fallback=True)
            print("LLM Aggregator initialized")
        except Exception as e:
            # Fail fast if aggregator is not configured since user requested no fallback
            raise
        
        # LLM Jury for evaluating actions
        try:
            self.jury = create_affordable_jury()  # Use affordable jury for cost efficiency
            print("LLM Jury initialized for action evaluation")
        except Exception as e:
            print(f"Failed to initialize LLM Jury: {e}")
            self.jury = None
        
        # Session tracking
        self.current_session: Optional[VisionBrowserSession] = None
        
        # Action history for vision-based actions
        self.vision_actions: List[VisionBrowserAction] = []

        # Deliberation records for decisions about which actions to run
        self.deliberations: List[Dict[str, Any]] = []
        
        # Screenshot directory for vision analysis
        self.vision_screenshot_dir = Path("data/vision_screenshots")
        self.vision_screenshot_dir.mkdir(parents=True, exist_ok=True)

    def _get_default_vision_config(self) -> VisionModelConfig:
        """Get default vision model configuration."""
        return VisionModelConfig(
            model_provider="openai",
            model_name="gpt-4o-mini",
            api_key=None,  # Will be loaded from environment
            max_tokens=1000,
            temperature=0.1
        )

        # Warn about missing API keys for providers at init time
    def _validate_vision_config(self) -> None:
        try:
            import os

            if self.vision_config.model_provider == "openai":
                if not (self.vision_config.api_key or os.getenv("OPENAI_API_KEY")):
                    print("Warning: OpenAI API key not set; openai vision calls will be disabled")
            if self.vision_config.model_provider == "anthropic":
                if not (self.vision_config.api_key or os.getenv("ANTHROPIC_API_KEY")):
                    print("Warning: Anthropic API key not set; anthropic vision calls will be disabled")
        except Exception:
            # non-critical
            pass

    def start_vision_session(self, user_goal: str, initial_url: Optional[str] = None) -> str:
        """Start a new vision-based browser automation session."""
        session_id = f"vision-session-{int(time.time() * 1000)}"
        
        self.current_session = VisionBrowserSession(
            session_id=session_id,
            start_time=datetime.utcnow(),
            current_url=initial_url or "",
            actions_performed=[],
            screenshots_taken=[],
            vision_analyses=[],
            user_goal=user_goal,
            status="active"
        )
        
        if initial_url:
            self.navigate_to(initial_url)
        
        print(f"Started vision session: {session_id}")
        print(f"Goal: {user_goal}")
        
        return session_id

    def end_vision_session(self) -> VisionBrowserSession:
        """End the current vision session and return results."""
        if not self.current_session:
            raise ValueError("No active vision session")
        
        self.current_session.status = "completed"
        self.current_session.actions_performed = self.vision_actions.copy()

        # Archive screenshots for this session
        try:
            screenshots = self.current_session.screenshots_taken or []
            if screenshots:
                backup_dir = Path("data/vision_backups")
                backup_dir.mkdir(parents=True, exist_ok=True)
                archive_name = f"{self.current_session.session_id}.zip"
                archive_path = backup_dir / archive_name

                # Create ZIP archive
                with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for s in screenshots:
                        try:
                            if os.path.exists(s):
                                zf.write(s, arcname=Path(s).name)
                        except Exception:
                            continue

                print(f"Archived {len(screenshots)} screenshots to {archive_path}")

                # Remove the original screenshots after archiving
                for s in screenshots:
                    try:
                        if os.path.exists(s):
                            os.remove(s)
                    except Exception:
                        continue

                # Rotate backups: keep only the most recent archive, remove older ones
                try:
                    archives = sorted(backup_dir.glob("vision-session-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
                    # Keep the most recent archive only
                    for old in archives[1:]:
                        try:
                            old.unlink()
                        except Exception:
                            continue
                except Exception:
                    pass

        except Exception as e:
            print(f"Error archiving screenshots: {e}")

        print(f"Ended vision session: {self.current_session.session_id}")
        print(f"Actions performed: {len(self.vision_actions)}")

        return self.current_session

    def analyze_page_with_vision(self, user_goal: str = "") -> Optional[VisionAnalysis]:
        """Analyze current page using vision model."""
        if not self.page:
            print("No active page to analyze")
            return None
        
        try:
            # Take screenshot for vision analysis
            screenshot_path = self._take_vision_screenshot()
            if not screenshot_path:
                return None
            
            # Use vision model to analyze the screenshot
            vision_analysis = self._call_vision_model(screenshot_path, user_goal)
            if vision_analysis:
                # Store the analysis
                if self.current_session:
                    self.current_session.vision_analyses.append(vision_analysis)
                    self.current_session.screenshots_taken.append(screenshot_path)
                
                print(f"Vision analysis completed: {len(vision_analysis.recommended_actions)} actions recommended")
                return vision_analysis
            
        except Exception as e:
            print(f"Error in vision analysis: {e}")
        
        return None

    def execute_vision_action(self, action: VisionAction) -> bool:
        """Execute an action recommended by the vision model."""
        try:
            # Create vision browser action record
            vision_action = VisionBrowserAction(
                action_type=action.action_type,
                target_description=action.target_description,
                coordinates=action.coordinates,
                text_input=action.text_input,
                selector=action.element_selector,
                confidence=action.confidence,
                reasoning=action.reasoning,
                timestamp=datetime.utcnow(),
                success=False,
                error_message=None
            )

            result_record: Dict[str, Any] = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": vision_action.model_dump() if hasattr(vision_action, 'model_dump') else vision_action.dict(),
                "success": False,
                "error": None,
                "page_state": None,
                "screenshot_path": None,
            }

            # Execute the action based on type
            success = False

            if action.action_type == "click":
                if action.coordinates:
                    success = self._click_at_coordinates(action.coordinates)
                elif action.element_selector:
                    success = self.click_element(action.element_selector)
                else:
                    success = self._click_by_description(action.target_description)

            elif action.action_type == "type":
                if action.text_input:
                    if action.element_selector:
                        success = self._type_in_element(action.element_selector, action.text_input)
                    elif action.coordinates:
                        if self._click_at_coordinates(action.coordinates):
                            success = self._type_text(action.text_input)
                    else:
                        success = self._type_by_description(action.target_description, action.text_input)

            elif action.action_type == "scroll":
                success = self._scroll_page()

            elif action.action_type == "navigate":
                success = self._navigate_by_description(action.target_description)

            elif action.action_type == "wait":
                time.sleep(2)
                success = True

            # Update records
            vision_action.success = success
            if not success:
                vision_action.error_message = f"Failed to execute {action.action_type} action"
                result_record["error"] = vision_action.error_message

            # Capture page state and screenshot after attempt
            try:
                page_state = self.get_page_state()
                result_record["page_state"] = {
                    "url": page_state.url,
                    "title": page_state.title,
                    "screenshot_path": page_state.screenshot_path,
                }
                result_record["screenshot_path"] = page_state.screenshot_path
            except Exception:
                # Non-critical
                pass

            result_record["success"] = success

            # Store the action and result
            self.vision_actions.append(vision_action)
            if self.current_session:
                # also append to session actions
                try:
                    self.current_session.actions_performed.append(vision_action)
                except Exception:
                    pass

            # Evaluate action with LLM jury
            if self.current_session:
                jury_score = self._evaluate_action_success(vision_action, self.current_session.user_goal)
                print(f"Jury action evaluation: {jury_score:.2f}")

            if success:
                print(f"Successfully executed: {action.action_type} - {action.target_description}")
            else:
                print(f"Failed to execute: {action.action_type} - {action.target_description}")

            return result_record

        except Exception as e:
            print(f"Error executing vision action: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def automate_with_vision(self, user_goal: str, max_actions: int = 10) -> VisionBrowserSession:
        """Automate browser interaction using vision model analysis."""
        session_id = self.start_vision_session(user_goal)
        
        try:
            action_count = 0
            
            while action_count < max_actions:
                # Analyze current page
                vision_analysis = self.analyze_page_with_vision(user_goal)
                if not vision_analysis:
                    print("No vision analysis available")
                    break
                
                # Get recommended actions
                recommended_actions = vision_analysis.recommended_actions
                if not recommended_actions:
                    print("No recommended actions")
                    break

                # Ask the agent to consider available actions and choose one
                chosen_action = None
                decision_record: Optional[Dict[str, Any]] = None
                try:
                    chosen_action, decision_record = self.consider_actions(recommended_actions, user_goal)
                except Exception as e:
                    print(f"Error during deliberation: {e}")

                # Fallback: if deliberation failed or returned nothing, pick highest-confidence
                if not chosen_action:
                    best_action = max(recommended_actions, key=lambda a: a.confidence)
                    if best_action.confidence < 0.3:
                        print(f"Low confidence action ({best_action.confidence}), stopping")
                        break
                    chosen_action = best_action

                if decision_record:
                    # store deliberation for session analysis
                    self.deliberations.append(decision_record)

                # Execute the chosen action with retries and possible LLM-driven repair
                success, attempt_record = self._execute_action_with_retries(chosen_action, user_goal, max_retries=3)
                action_count += 1

                # Attach attempt record to deliberations for later inspection
                if decision_record is None:
                    decision_record = {}
                decision_record["attempts"] = attempt_record
                
                if not success:
                    print("Action failed, trying next action")
                    continue
                
                # Wait for page to load
                time.sleep(2)
                
                # Check if goal is achieved
                if self._check_goal_achievement(user_goal):
                    print("Goal appears to be achieved")
                    break
            
            print(f"Automation completed: {action_count} actions performed")
            
        except Exception as e:
            print(f"Error in vision automation: {e}")
            if self.current_session:
                self.current_session.status = "failed"
        
        return self.end_vision_session()

    def _take_vision_screenshot(self) -> Optional[str]:
        """Take a screenshot specifically for vision analysis."""
        try:
            # Wait for page to load
            self.page.wait_for_load_state("networkidle", timeout=10000)
            
            # Generate screenshot filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            screenshot_filename = f"vision_{timestamp}.png"
            screenshot_path = self.vision_screenshot_dir / screenshot_filename
            
            # Take full page screenshot
            self.page.screenshot(
                path=str(screenshot_path),
                full_page=True
            )
            
            return str(screenshot_path)
            
        except Exception as e:
            print(f"Failed to take vision screenshot: {e}")
            return None

    def _call_vision_model(self, screenshot_path: str, user_goal: str) -> Optional[VisionAnalysis]:
        """Call vision models with automatic fallback support."""
        try:
            # Read and encode screenshot
            with open(screenshot_path, "rb") as f:
                image_data = f.read()
                base64_image = base64.b64encode(image_data).decode("utf-8")
            
            # Prepare prompt for vision model
            prompt = self._create_vision_prompt(user_goal)
            
            # Try providers in order of preference with fallback
            providers_to_try = [
                ("openai", self._call_openai_vision),
                ("anthropic", self._call_anthropic_vision),
                ("local", self._call_local_vision)
            ]
            
            for provider_name, provider_func in providers_to_try:
                try:
                    print(f"Trying {provider_name} vision model...")
                    
                    if provider_name == "local":
                        result = provider_func(screenshot_path, prompt)
                    else:
                        result = provider_func(base64_image, prompt)
                    
                    if result:
                        print(f"{provider_name} vision analysis successful")
                        
                        # Use LLM jury to evaluate the analysis quality
                        if self.jury and provider_name != "local":
                            evaluation = self._evaluate_vision_analysis(result, user_goal)
                            print(f"Jury evaluation score: {evaluation:.2f}")
                            
                            # If quality is too low, try next provider
                            if evaluation < 0.6:
                                print(f"Low quality analysis, trying next provider...")
                                continue
                        
                        return result
                    else:
                        print(f"❌ {provider_name} returned no result")
                        
                except Exception as e:
                    print(f"❌ {provider_name} vision model failed: {e}")
                    continue
            
            print("❌ All vision models failed")
            return None
                
        except Exception as e:
            print(f"Error calling vision model: {e}")
            return None

    def _create_vision_prompt(self, user_goal: str) -> str:
        """Create a prompt for vision model analysis."""
        return f"""
Analyze this web page screenshot and provide detailed information for browser automation.

User Goal: {user_goal}

Please provide:
1. Page description
2. Interactive elements (buttons, links, form fields)
3. Recommended actions to achieve the user goal
4. Navigation opportunities
5. Page type classification

Focus on elements that would help achieve: {user_goal}

Return the analysis in JSON format with the following structure:
{{
    "page_description": "Human-readable description",
    "interactive_elements": [
        {{
            "type": "button|link|form_field",
            "description": "Element description",
            "coordinates": [x, y],
            "confidence": 0.0-1.0
        }}
    ],
    "recommended_actions": [
        {{
            "action_type": "click|type|scroll|navigate|wait",
            "target_description": "What to click/type",
            "confidence": 0.0-1.0,
            "coordinates": [x, y],
            "text_input": "text to type (if applicable)",
            "reasoning": "Why this action is recommended"
        }}
    ],
    "page_type": "login|search|content|navigation",
    "navigation_opportunities": ["list of navigation options"]
}}
"""

    def _call_openai_vision(self, base64_image: str, prompt: str) -> Optional[VisionAnalysis]:
        """Call OpenAI vision model."""
        try:
            import os
            api_key = self.vision_config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("OpenAI API key not found")
                return None
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Use provider-agnostic parameters: some OpenAI endpoints expect max_completion_tokens
            data = {
                "model": self.vision_config.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        ],
                    }
                ],
                "temperature": self.vision_config.temperature,
            }

            # prefer max_completion_tokens if supported
            if hasattr(self.vision_config, "max_tokens") and self.vision_config.max_tokens:
                data["max_completion_tokens"] = self.vision_config.max_tokens
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
            )

            # handle empties and parse safely
            if response.status_code != 200:
                print(f"OpenAI API error: {response.status_code} - {response.text}")
                return None

            try:
                result = response.json()
            except ValueError:
                print("OpenAI returned non-JSON response")
                return None

            # Support multiple response shapes
            choices = result.get("choices") or []
            if not choices:
                print("OpenAI returned no choices")
                return None

            message = choices[0].get("message") or {}
            # content may be stringified JSON or direct JSON
            content = message.get("content")
            if not content:
                print("OpenAI message content empty")
                return None

            # If content is list/dict, try to extract text
            if isinstance(content, list):
                # gather text blocks
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("content")
                        if text:
                            parts.append(text)
                content_text = "\n".join(parts)
            elif isinstance(content, dict):
                content_text = json.dumps(content)
            else:
                content_text = str(content)

            # try to find JSON in content_text
            try:
                analysis_data = json.loads(content_text)
            except Exception:
                # fallback: try to extract JSON substring
                m = re.search(r"\{[\s\S]*\}", content_text)
                if m:
                    try:
                        analysis_data = json.loads(m.group(0))
                    except Exception:
                        print("Failed to parse JSON from OpenAI content")
                        return None
                else:
                    print("No JSON found in OpenAI content")
                    return None

            return self._parse_vision_response(analysis_data)
                
        except Exception as e:
            print(f"Error calling OpenAI vision: {e}")
            return None

    def _call_anthropic_vision(self, base64_image: str, prompt: str) -> Optional[VisionAnalysis]:
        """Call Anthropic vision model using official SDK."""
        try:
            import os
            import anthropic
            api_key = self.vision_config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("Anthropic API key not found")
                return None

            # Use Anthropic SDK but defensively handle errors and response shapes
            client = anthropic.Anthropic(api_key=api_key)
            # The messages SDK may differ between versions; attempt a generic call
            try:
                message = client.messages.create(
                    model=self.vision_config.model_name,
                    messages=[{"type": "input_text", "text": prompt}],
                    temperature=self.vision_config.temperature,
                )
            except Exception as e:
                print(f"Anthropic SDK call failed: {e}")
                return None

            # Try to extract text from message
            try:
                content_text = ""
                if hasattr(message, "content"):
                    # message.content might be a list or string
                    if isinstance(message.content, str):
                        content_text = message.content
                    elif isinstance(message.content, list):
                        parts = []
                        for part in message.content:
                            if isinstance(part, dict):
                                text = part.get("text") or part.get("content")
                                if text:
                                    parts.append(text)
                        content_text = "\n".join(parts)

                if not content_text:
                    print("Anthropic returned empty content")
                    return None

                analysis_data = json.loads(content_text)
                return self._parse_vision_response(analysis_data)
            except Exception as e:
                print(f"Error parsing Anthropic response: {e}")
                return None

        except Exception as e:
            print(f"Error calling Anthropic vision: {e}")
            return None

    def _call_local_vision(self, screenshot_path: str, prompt: str) -> Optional[VisionAnalysis]:
        """Call local vision model (placeholder for local deployment)."""
        # Provide a deterministic mock analysis for local/offline testing to avoid repeated API errors.
        try:
            analyzer = ImageAnalyzer(self.store)
            analysis = analyzer.analyze_for_automation(screenshot_path, page_url="", user_goal=prompt)
            if not analysis:
                print("Local analyzer returned no result")
                return None

            # Convert ImageAnalysis to VisionAnalysis recommended structure
            # Create a simple recommended action: click first detected button or navigate to first link
            recommended = []
            for btn in (analysis.ui_components or []):
                if btn == "search_bar":
                    recommended.append(VisionAction(action_type="type", target_description="search input", confidence=0.8, reasoning="Detected search bar"))
                elif btn == "button":
                    recommended.append(VisionAction(action_type="click", target_description="button", confidence=0.7, reasoning="Detected button"))

            # Build VisionAnalysis from image analyzer outputs
            va = VisionAnalysis(
                image_path=screenshot_path,
                page_description=analysis.content_type if hasattr(analysis, 'content_type') else "local page",
                interactive_elements=[],
                recommended_actions=recommended,
                page_type=analysis.content_type if hasattr(analysis, 'content_type') else "unknown",
                navigation_opportunities=[],
                form_fields=[],
                buttons=[],
                links=[],
                checksum=getattr(analysis, 'checksum', ''),
                analysis_time=datetime.utcnow(),
            )
            return va
        except Exception as e:
            print(f"Local vision analysis failed: {e}")
            return None

    def _parse_vision_response(self, analysis_data: Dict[str, Any]) -> VisionAnalysis:
        """Parse vision model response into VisionAnalysis object."""
        try:
            # Convert interactive elements
            interactive_elements = []
            for elem in analysis_data.get("interactive_elements", []):
                interactive_elements.append({
                    "type": elem.get("type", "unknown"),
                    "description": elem.get("description", ""),
                    "coordinates": elem.get("coordinates", [0, 0]),
                    "confidence": elem.get("confidence", 0.5)
                })
            
            # Convert recommended actions
            recommended_actions = []
            for action in analysis_data.get("recommended_actions", []):
                vision_action = VisionAction(
                    action_type=action.get("action_type", "click"),
                    target_description=action.get("target_description", ""),
                    confidence=action.get("confidence", 0.5),
                    coordinates=tuple(action.get("coordinates", [0, 0])) if action.get("coordinates") else None,
                    text_input=action.get("text_input"),
                    reasoning=action.get("reasoning", "")
                )
                recommended_actions.append(vision_action)
            
            # Create VisionAnalysis object
            return VisionAnalysis(
                image_path="",  # Will be set by caller
                page_description=analysis_data.get("page_description", ""),
                interactive_elements=interactive_elements,
                recommended_actions=recommended_actions,
                page_type=analysis_data.get("page_type", "unknown"),
                navigation_opportunities=analysis_data.get("navigation_opportunities", []),
                form_fields=[],
                buttons=[],
                links=[],
                checksum="",
                analysis_time=datetime.utcnow()
            )
            
        except Exception as e:
            print(f"Error parsing vision response: {e}")
            return None

    def _click_at_coordinates(self, coordinates: Tuple[int, int]) -> bool:
        """Click at specific screen coordinates."""
        try:
            self.page.click(coordinates[0], coordinates[1])
            return True
        except Exception as e:
            print(f"Error clicking at coordinates: {e}")
            return False

    def consider_actions(self, recommended_actions: List[VisionAction], user_goal: str) -> tuple[Optional[VisionAction], Dict[str, Any]]:
        """Deliberate over recommended actions and return a chosen action plus a decision record.

        This function uses the AIResponseGenerator (LLM) to score or pick actions when available,
        falling back to a simple heuristic if no LLM is available.
        """
        # Build a concise prompt describing the actions
        try:
            actions_summary = []
            for i, a in enumerate(recommended_actions):
                actions_summary.append({
                    "index": i,
                    "action_type": a.action_type,
                    "target": a.target_description,
                    "confidence": float(a.confidence),
                    "reasoning": a.reasoning,
                })

            prompt = {
                "user_goal": user_goal,
                "actions": actions_summary,
                "instruction": "Select the best action index to execute that most directly advances the user goal. Return a JSON object {\"chosen_index\": INT, \"rationale\": \"text\"}."
            }

            decision_record: Dict[str, Any] = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_goal": user_goal,
                "actions": actions_summary,
            }

            # If we have an AI generator, ask it to pick; else, use heuristic
            chosen_action: Optional[VisionAction] = None
            if self.ai_generator and self.ai_generator.llm_client:
                # Use the direct response generator which tries available APIs
                json_prompt = json.dumps(prompt)
                raw = self.ai_generator.generate_direct_response(json_prompt)
                # Try to parse JSON out of the response
                try:
                    parsed = json.loads(raw)
                    idx = int(parsed.get("chosen_index"))
                    rationale = parsed.get("rationale", "")
                    if 0 <= idx < len(recommended_actions):
                        chosen_action = recommended_actions[idx]
                        decision_record["chosen_index"] = idx
                        decision_record["rationale"] = rationale
                    else:
                        decision_record["error"] = "invalid_index_from_llm"
                except Exception:
                    # If parsing failed, store the raw response and fall back
                    decision_record["llm_raw"] = raw

            if not chosen_action:
                # Heuristic: prefer highest confidence, prefer 'type' or 'click' over others when confidence similar
                sorted_actions = sorted(recommended_actions, key=lambda a: (a.confidence, a.action_type == "type" or a.action_type == "click"), reverse=True)
                chosen_action = sorted_actions[0]
                decision_record["chosen_index"] = recommended_actions.index(chosen_action)
                decision_record["rationale"] = "heuristic: highest confidence, prefer typing/clicking"

            return chosen_action, decision_record

        except Exception as e:
            return None, {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def _execute_action_with_retries(self, action: VisionAction, user_goal: str, max_retries: int = 3) -> tuple[bool, List[Dict[str, Any]]]:
        """Execute a VisionAction with retries. On failure, allow the LLM to inspect the error and propose a fix up to max_retries times.

        Returns (success, attempt_records).
        """
        attempt_records: List[Dict[str, Any]] = []

        for attempt in range(1, max_retries + 1):
            attempt_info: Dict[str, Any] = {"attempt": attempt, "action": {
                "action_type": action.action_type,
                "target": action.target_description,
                "confidence": float(action.confidence)
            }}

            try:
                result = self.execute_vision_action(action)
                # execute_vision_action now returns a structured dict
                success = bool(result.get("success", False)) if isinstance(result, dict) else bool(result)
                attempt_info["result"] = result
                attempt_info["success"] = success
                attempt_records.append(attempt_info)

                if success:
                    return True, attempt_records

                # If failed and we have an AI generator, ask it to propose a fix
                if self.ai_generator and self.ai_generator.llm_client:
                    # Build repair prompt including tool output
                    repair_prompt = {
                        "user_goal": user_goal,
                        "failed_action": attempt_info["action"],
                        "tool_output": result,
                        "instruction": "Propose a modified action to retry (change selector, use coordinates, or change action_type). Return JSON {\"action\": {\"action_type\":..., \"target_description\":..., \"coordinates\": [x,y]|null, \"text_input\": null}}"
                    }

                    raw = self.ai_generator.generate_direct_response(json.dumps(repair_prompt))
                    attempt_info["llm_raw"] = raw
                    try:
                        parsed = json.loads(raw)
                        proposed = parsed.get("action")
                        if proposed:
                            # Apply proposed changes into the action copy
                            new_action = VisionAction(
                                action_type=proposed.get("action_type", action.action_type),
                                target_description=proposed.get("target_description", action.target_description),
                                confidence=proposed.get("confidence", action.confidence),
                                coordinates=tuple(proposed.get("coordinates")) if proposed.get("coordinates") else action.coordinates,
                                text_input=proposed.get("text_input", action.text_input),
                                reasoning=proposed.get("reasoning", action.reasoning)
                            )
                            attempt_info["proposed_action"] = proposed
                            # Replace action to retry
                            action = new_action
                            attempt_records.append(attempt_info)
                            continue  # retry with new action
                        else:
                            attempt_info["llm_error"] = "no_action_proposed"
                            attempt_records.append(attempt_info)
                            continue
                    except Exception:
                        attempt_info["llm_error"] = "json_parse_failed"
                        attempt_records.append(attempt_info)
                        continue

                # No ai_generator or LLM couldn't help, continue to next attempt to follow heuristic
                attempt_records.append(attempt_info)
                continue

            except Exception as e:
                attempt_info["error"] = str(e)
                attempt_records.append(attempt_info)
                continue

        # All attempts failed
        return False, attempt_records

    def _click_by_description(self, description: str) -> bool:
        """Click element by description."""
        try:
            # Try to find element by text content
            return self.click_element_by_text(description)
        except Exception as e:
            print(f"Error clicking by description: {e}")
            return False

    def _type_in_element(self, selector: str, text: str) -> bool:
        """Type text in a specific element."""
        try:
            element = self.page.query_selector(selector)
            if element:
                element.fill(text)
                return True
            return False
        except Exception as e:
            print(f"Error typing in element: {e}")
            return False

    def _type_text(self, text: str) -> bool:
        """Type text at current focus."""
        try:
            self.page.keyboard.type(text)
            return True
        except Exception as e:
            print(f"Error typing text: {e}")
            return False

    def _type_by_description(self, description: str, text: str) -> bool:
        """Type text by finding element by description."""
        try:
            # Try to find input field by description
            elements = self.find_elements_by_text(description)
            for element in elements:
                if element.element_type.startswith("input_"):
                    return self.click_element(element.selector) and self._type_text(text)
            return False
        except Exception as e:
            print(f"Error typing by description: {e}")
            return False

    def _scroll_page(self) -> bool:
        """Scroll the page."""
        try:
            self.page.evaluate("window.scrollBy(0, 500)")
            return True
        except Exception as e:
            print(f"Error scrolling page: {e}")
            return False

    def _navigate_by_description(self, description: str) -> bool:
        """Navigate by description."""
        try:
            # Try to find and click a link by description
            return self.click_element_by_text(description)
        except Exception as e:
            print(f"Error navigating by description: {e}")
            return False

    def _check_goal_achievement(self, user_goal: str) -> bool:
        """Check if the user goal has been achieved."""
        try:
            # Simple heuristic: check if page content contains goal-related keywords
            page_content = self.page.content().lower()
            goal_keywords = user_goal.lower().split()
            
            # Check if most goal keywords appear in page content
            matching_keywords = sum(1 for keyword in goal_keywords if keyword in page_content)
            return matching_keywords >= len(goal_keywords) * 0.7
            
        except Exception as e:
            print(f"Error checking goal achievement: {e}")
            return False

    def _evaluate_vision_analysis(self, analysis: VisionAnalysis, user_goal: str) -> float:
        """Evaluate vision analysis quality using LLM jury."""
        try:
            if not self.jury:
                return 0.8  # Default score if no jury available
            
            # Create evaluation prompt
            prompt = f"""
            Evaluate the quality of this vision analysis for browser automation.
            
            User Goal: {user_goal}
            
            Analysis:
            - Page Description: {analysis.page_description}
            - Recommended Actions: {len(analysis.recommended_actions)} actions
            - Page Type: {analysis.page_type}
            - Interactive Elements: {len(analysis.interactive_elements)} elements
            
            Rate the analysis quality from 0.0 to 1.0 based on:
            1. Relevance to user goal
            2. Actionability of recommendations
            3. Accuracy of page understanding
            4. Completeness of analysis
            """
            
            # Create mock response for jury evaluation
            response = f"Vision Analysis Quality Assessment for goal: {user_goal}"
            
            # Get jury evaluation
            jury_result = self.jury.evaluate(prompt, response)
            return jury_result.candidate_score
            
        except Exception as e:
            print(f"Error evaluating vision analysis: {e}")
            return 0.7  # Default score on error

    def _evaluate_action_success(self, action: VisionBrowserAction, user_goal: str) -> float:
        """Evaluate action success using LLM jury."""
        try:
            if not self.jury:
                return 0.8 if action.success else 0.2
            
            # Create evaluation prompt
            prompt = f"""
            Evaluate the success and appropriateness of this browser automation action.
            
            User Goal: {user_goal}
            Action: {action.action_type} - {action.target_description}
            Success: {action.success}
            Reasoning: {action.reasoning}
            Confidence: {action.confidence}
            
            Rate from 0.0 to 1.0 based on:
            1. Technical success
            2. Progress toward goal
            3. Appropriateness of action
            4. Potential for goal achievement
            """
            
            response = f"Action Evaluation: {action.action_type} was {'successful' if action.success else 'unsuccessful'}"
            
            # Get jury evaluation
            jury_result = self.jury.evaluate(prompt, response)
            return jury_result.candidate_score
            
        except Exception as e:
            print(f"Error evaluating action success: {e}")
            return 0.7 if action.success else 0.3

    def get_vision_session_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current vision session."""
        if not self.current_session:
            return None
        
        return {
            "session_id": self.current_session.session_id,
            "goal": self.current_session.user_goal,
            "status": self.current_session.status,
            "actions_performed": len(self.current_session.actions_performed),
            "screenshots_taken": len(self.current_session.screenshots_taken),
            "vision_analyses": len(self.current_session.vision_analyses),
            "current_url": self.current_session.current_url,
            "start_time": self.current_session.start_time.isoformat()
        }
    
    def search(self, query: str, max_results: int = 10) -> list[str]:
        """Vision-based search using intelligent browser automation."""
        print(f"🔍 VISION SEARCH: {query}")
        
        # Start a vision session for this search
        session_id = self.start_vision_session(
            user_goal=f"Search for: {query}",
            initial_url="https://www.google.com"
        )
        
        discovered_urls = []
        
        try:
            # Navigate to Google search
            self.navigate_to("https://www.google.com")
            
            # Analyze the search page with vision
            search_analysis = self.analyze_page_with_vision(
                user_goal=f"Find the search box and enter: {query}"
            )
            
            if search_analysis and search_analysis.recommended_actions:
                # Execute search actions (click search box, type query, click search button)
                for action in search_analysis.recommended_actions:
                    if action.confidence > 0.6:
                        print(f"   Executing search action: {action.action_type} - {action.target_description}")
                        success = self.execute_vision_action(action)
                        
                        if success:
                            # Wait for search results
                            time.sleep(3)
                            
                            # Analyze search results page
                            results_analysis = self.analyze_page_with_vision(
                                user_goal=f"Find relevant search result links for: {query}"
                            )
                            
                            if results_analysis:
                                # Extract URLs from search results
                                urls = self._extract_urls_from_vision_analysis(results_analysis, query)
                                discovered_urls.extend(urls[:max_results])
                            
                            break  # Stop after first successful search action
            
            # If no URLs found, try alternative search engines
            if not discovered_urls:
                print("   Trying alternative search engines...")
                alternative_engines = [
                    "https://duckduckgo.com",
                    "https://www.bing.com",
                    "https://en.wikipedia.org"
                ]
                
                for engine in alternative_engines:
                    if len(discovered_urls) >= max_results:
                        break
                    
                    try:
                        self.navigate_to(engine)
                        engine_analysis = self.analyze_page_with_vision(
                            user_goal=f"Search for: {query}"
                        )
                        
                        if engine_analysis:
                            urls = self._extract_urls_from_vision_analysis(engine_analysis, query)
                            discovered_urls.extend(urls[:max_results - len(discovered_urls)])
                    
                    except Exception as e:
                        print(f"   Failed with {engine}: {e}")
                        continue
        
        finally:
            # End the vision session
            self.end_vision_session()
        
        print(f"📊 VISION SEARCH RESULTS: {len(discovered_urls)} URLs found")
        return discovered_urls[:max_results]
    
    def _extract_urls_from_vision_analysis(self, analysis: VisionAnalysis, query: str) -> list[str]:
        """Extract URLs from vision analysis of search results."""
        urls = []
        
        if not analysis.recommended_actions:
            return urls
        
        # Look for navigation actions that might be search result links
        for action in analysis.recommended_actions:
            if action.action_type == "navigate" and action.confidence > 0.5:
                # Try to extract URL from the action description or target
                potential_url = self._extract_url_from_action(action)
                if potential_url and self._is_relevant_url(potential_url, query):
                    urls.append(potential_url)
        
        return urls
    
    def _extract_url_from_action(self, action: VisionAction) -> Optional[str]:
        """Extract URL from a vision action."""
        # This is a simplified implementation
        # In a real system, you'd need more sophisticated URL extraction
        if hasattr(action, 'target_description'):
            # Look for URL patterns in the description
            import re
            url_pattern = r'https?://[^\s<>"]+'
            matches = re.findall(url_pattern, action.target_description)
            if matches:
                return matches[0]
        
        return None
    
    def _is_relevant_url(self, url: str, query: str) -> bool:
        """Check if URL is relevant to the search query."""
        # Simple relevance check
        query_terms = query.lower().split()
        url_lower = url.lower()
        
        # Check if query terms appear in URL
        relevance_score = sum(1 for term in query_terms if term in url_lower)
        return relevance_score > 0
