"""Image analysis module for analyzing and classifying screenshots with vision-based browser automation."""

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field


class ImageAnalysis(BaseModel):
    """Result of image analysis."""

    image_path: str
    content_type: str = Field(
        description="Type of content detected (e.g., 'text', 'form', 'data_table')"
    )
    ui_components: list[str] = Field(description="UI components detected")
    text_regions: list[dict[str, Any]] = Field(description="Text regions found")
    color_scheme: dict[str, str] = Field(description="Dominant colors detected")
    layout_type: str = Field(
        description="Layout classification (e.g., 'single_column', 'multi_column')"
    )
    checksum: str = Field(description="SHA-256 checksum of image")
    analysis_time: datetime


class VisionAction(BaseModel):
    """Action recommended by vision model for browser automation."""

    action_type: str  # click, type, scroll, navigate, wait
    target_description: str  # Human-readable description of target
    confidence: float  # Confidence score 0-1
    reasoning: str = Field(description="Why this action was recommended")
    coordinates: Optional[Tuple[int, int]] = Field(
        default=None, description="Screen coordinates for action"
    )
    text_input: Optional[str] = Field(
        default=None, description="Text to type if action is 'type'"
    )
    element_selector: Optional[str] = Field(
        default=None, description="CSS selector if available"
    )


class VisionAnalysis(BaseModel):
    """Comprehensive vision analysis for browser automation."""

    image_path: str
    page_description: str = Field(description="Human-readable description of the page")
    interactive_elements: list[dict[str, Any]] = Field(
        description="Interactive elements detected"
    )
    recommended_actions: list[VisionAction] = Field(
        description="Recommended actions for automation"
    )
    page_type: str = Field(description="Type of page (login, search, content, etc.)")
    navigation_opportunities: list[str] = Field(
        description="Available navigation options"
    )
    form_fields: list[dict[str, Any]] = Field(description="Form fields detected")
    buttons: list[dict[str, Any]] = Field(description="Buttons detected")
    links: list[dict[str, Any]] = Field(description="Links detected")
    checksum: str = Field(description="SHA-256 checksum of image")
    analysis_time: datetime


class ImageAnalyzer:
    """Analyzer for classifying and analyzing web page screenshots with vision-based automation."""

    def __init__(self, store: Any):
        """Initialize image analyzer with storage."""
        self.store = store
        self.content_types = [
            "text_article",
            "form_login",
            "form_search",
            "data_table",
            "navigation",
            "error_page",
            "landing_page",
            "product_page",
            "blog_post",
            "dashboard",
        ]

        self.ui_components = [
            "navigation_menu",
            "search_bar",
            "button",
            "form_field",
            "table",
            "image",
            "video",
            "advertisement",
            "footer",
            "header",
        ]

    def _log_job(
        self,
        job_type: str,
        input_data: dict[str, Any],
        output_data: Optional[dict[str, Any]] = None,
        error_data: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log a job to the store."""
        job_id = f"{job_type}-{int(time.time() * 1000)}"

        job_data = {
            "id": job_id,
            "type": job_type,
            "status": "failed" if error_data else "completed",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "started_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "input": input_data,
            "output": output_data,
            "error": error_data,
            "metadata": {
                "version": "0.1.0",
                "priority": 5,
                "retries": 0,
                "max_retries": 3,
                "timeout_seconds": 30,
            },
        }

        self.store.store_job(job_data)
        return job_id

    def _detect_text_regions(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect text regions in the image using OCR."""
        try:
            # Convert to grayscale for better text detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Use morphological operations to find text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by size to avoid noise
                if w > 20 and h > 10:
                    text_regions.append(
                        {"x": x, "y": y, "width": w, "height": h, "area": w * h}
                    )

            return text_regions

        except Exception as e:
            print(f"Error detecting text regions: {e}")
            return []

    def _detect_ui_components(self, image: np.ndarray) -> list[str]:
        """Detect UI components in the image."""
        components = []

        try:
            # Convert to different color spaces for better detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect buttons (rectangular shapes with text)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Button-like shapes
                if 0.5 < aspect_ratio < 4 and w > 30 and h > 20:
                    components.append("button")

            # Detect search bars (horizontal rectangles)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Search bar-like shapes
                if aspect_ratio > 3 and w > 100 and h < 50:
                    components.append("search_bar")

            # Detect form fields (small rectangles)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Form field-like shapes
                if 0.5 < aspect_ratio < 3 and 20 < w < 200 and 20 < h < 50:
                    components.append("form_field")

            # Remove duplicates
            components = list(set(components))

        except Exception as e:
            print(f"Error detecting UI components: {e}")

        return components

    def _analyze_color_scheme(self, image: np.ndarray) -> dict[str, str]:
        """Analyze the color scheme of the image."""
        try:
            # Convert to RGB for better color analysis
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Reshape image to 2D array of pixels
            pixels = rgb.reshape(-1, 3)

            # Calculate mean color
            mean_color = np.mean(pixels, axis=0)

            # Calculate dominant colors using k-means
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)

            colors = kmeans.cluster_centers_
            labels = kmeans.labels_

            # Count labels to find dominant colors
            unique, counts = np.unique(labels, return_counts=True)
            dominant_colors = colors[unique[np.argsort(counts)[::-1]]]

            return {
                "primary": f"rgb({int(dominant_colors[0][0])}, {int(dominant_colors[0][1])}, {int(dominant_colors[0][2])})",
                "secondary": f"rgb({int(dominant_colors[1][0])}, {int(dominant_colors[1][1])}, {int(dominant_colors[1][2])})",
                "background": f"rgb({int(mean_color[0])}, {int(mean_color[1])}, {int(mean_color[2])})",
            }

        except Exception as e:
            print(f"Error analyzing color scheme: {e}")
            return {
                "primary": "rgb(0, 0, 0)",
                "secondary": "rgb(128, 128, 128)",
                "background": "rgb(255, 255, 255)",
            }

    def _classify_layout(
        self, image: np.ndarray, text_regions: list[dict[str, Any]]
    ) -> str:
        """Classify the layout of the page."""
        try:
            if not text_regions:
                return "unknown"

            # Analyze text region distribution
            x_coords = [region["x"] for region in text_regions]
            y_coords = [region["y"] for region in text_regions]

            # Check if text is distributed in columns
            x_variance = np.var(x_coords)
            y_variance = np.var(y_coords)

            if x_variance > y_variance * 2:
                return "multi_column"
            elif len(text_regions) > 20:
                return "content_heavy"
            else:
                return "single_column"

        except Exception as e:
            print(f"Error classifying layout: {e}")
            return "unknown"

    def _classify_content_type(
        self,
        image: np.ndarray,
        ui_components: list[str],
        text_regions: list[dict[str, Any]],
    ) -> str:
        """Classify the type of content on the page."""
        try:
            # Check for specific UI components
            if "search_bar" in ui_components:
                return "form_search"
            elif (
                "form_field" in ui_components
                and len([c for c in ui_components if c == "form_field"]) > 2
            ):
                return "form_login"
            elif "button" in ui_components and len(text_regions) < 10:
                return "navigation"
            else:
                return "landing_page"

        except Exception as e:
            print(f"Error classifying content type: {e}")
            return "landing_page"

    def analyze_image(self, image_path: str) -> Optional[ImageAnalysis]:
        """Analyze an image and return classification results."""
        start_time = datetime.utcnow()

        try:
            # Check if image exists
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Calculate image checksum
            with open(image_path, "rb") as f:
                image_data = f.read()
                checksum = hashlib.sha256(image_data).hexdigest()

            # Perform analysis
            text_regions = self._detect_text_regions(image)
            ui_components = self._detect_ui_components(image)
            color_scheme = self._analyze_color_scheme(image)
            layout_type = self._classify_layout(image, text_regions)
            content_type = self._classify_content_type(
                image, ui_components, text_regions
            )

            result = ImageAnalysis(
                image_path=image_path,
                content_type=content_type,
                ui_components=ui_components,
                text_regions=text_regions,
                color_scheme=color_scheme,
                layout_type=layout_type,
                checksum=checksum,
                analysis_time=start_time,
            )

            # Log successful job
            output_data = {
                "content_type": content_type,
                "ui_components_count": len(ui_components),
                "text_regions_count": len(text_regions),
                "layout_type": layout_type,
                "checksum": checksum,
            }
            self._log_job("image_analysis", {"image_path": image_path}, output_data)

            return result

        except Exception as e:
            error_data = {
                "type": "image_analysis_error",
                "message": str(e),
                "stack": None,
            }
            self._log_job(
                "image_analysis", {"image_path": image_path}, error_data=error_data
            )
            return None

    def analyze_for_automation(
        self, image_path: str, page_url: str = "", user_goal: str = ""
    ) -> Optional[VisionAnalysis]:
        """Analyze image for browser automation using vision models."""
        start_time = datetime.utcnow()

        try:
            # Check if image exists
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Calculate image checksum
            with open(image_path, "rb") as f:
                image_data = f.read()
                checksum = hashlib.sha256(image_data).hexdigest()

            # Perform basic analysis
            text_regions = self._detect_text_regions(image)
            ui_components = self._detect_ui_components(image)
            content_type = self._classify_content_type(
                image, ui_components, text_regions
            )

            # Generate page description
            page_description = self._generate_page_description(
                image, ui_components, text_regions, content_type
            )

            # Detect interactive elements for automation
            interactive_elements = self._detect_interactive_elements(
                image, text_regions
            )

            # Generate recommended actions
            recommended_actions = self._generate_recommended_actions(
                image, interactive_elements, user_goal, content_type
            )

            # Categorize elements
            buttons = [
                elem for elem in interactive_elements if elem.get("type") == "button"
            ]
            links = [
                elem for elem in interactive_elements if elem.get("type") == "link"
            ]
            form_fields = [
                elem
                for elem in interactive_elements
                if elem.get("type") == "form_field"
            ]

            # Generate navigation opportunities
            navigation_opportunities = self._generate_navigation_opportunities(
                links, buttons
            )

            result = VisionAnalysis(
                image_path=image_path,
                page_description=page_description,
                interactive_elements=interactive_elements,
                recommended_actions=recommended_actions,
                page_type=content_type,
                navigation_opportunities=navigation_opportunities,
                form_fields=form_fields,
                buttons=buttons,
                links=links,
                checksum=checksum,
                analysis_time=start_time,
            )

            # Log successful job
            output_data = {
                "page_type": content_type,
                "interactive_elements_count": len(interactive_elements),
                "recommended_actions_count": len(recommended_actions),
                "navigation_opportunities_count": len(navigation_opportunities),
                "checksum": checksum,
            }
            self._log_job(
                "vision_analysis",
                {"image_path": image_path, "url": page_url, "goal": user_goal},
                output_data,
            )

            return result

        except Exception as e:
            error_data = {
                "type": "vision_analysis_error",
                "message": str(e),
                "stack": None,
            }
            self._log_job(
                "vision_analysis",
                {"image_path": image_path, "url": page_url, "goal": user_goal},
                error_data=error_data,
            )
            return None

    def _generate_page_description(
        self,
        image: np.ndarray,
        ui_components: List[str],
        text_regions: List[Dict[str, Any]],
        content_type: str,
    ) -> str:
        """Generate a human-readable description of the page."""
        try:
            description_parts = []

            # Add content type
            if content_type == "form_login":
                description_parts.append("This appears to be a login page")
            elif content_type == "form_search":
                description_parts.append("This appears to be a search page")
            elif content_type == "navigation":
                description_parts.append("This appears to be a navigation page")
            else:
                description_parts.append("This appears to be a content page")

            # Add UI components
            if ui_components:
                component_descriptions = []
                for component in ui_components:
                    if component == "search_bar":
                        component_descriptions.append("search bar")
                    elif component == "button":
                        component_descriptions.append("buttons")
                    elif component == "form_field":
                        component_descriptions.append("form fields")

                if component_descriptions:
                    description_parts.append(
                        f"with {', '.join(component_descriptions)}"
                    )

            # Add text content info
            if text_regions:
                description_parts.append(f"containing {len(text_regions)} text regions")

            return ". ".join(description_parts) + "."

        except Exception as e:
            print(f"Error generating page description: {e}")
            return "A web page with various interactive elements."

    def _detect_interactive_elements(
        self, image: np.ndarray, text_regions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Detect interactive elements in the image for automation."""
        elements = []

        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Detect buttons (rectangular shapes with text)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Button-like shapes
                if 0.5 < aspect_ratio < 4 and w > 30 and h > 20:
                    # Check if there's text in this region
                    has_text = any(
                        abs(region["x"] - x) < 10 and abs(region["y"] - y) < 10
                        for region in text_regions
                    )

                    if has_text:
                        elements.append(
                            {
                                "type": "button",
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "coordinates": (x + w // 2, y + h // 2),
                                "description": f"Button at ({x}, {y})",
                            }
                        )

            # Detect form fields (small rectangles)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Form field-like shapes
                if 0.5 < aspect_ratio < 3 and 20 < w < 200 and 20 < h < 50:
                    elements.append(
                        {
                            "type": "form_field",
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "coordinates": (x + w // 2, y + h // 2),
                            "description": f"Form field at ({x}, {y})",
                        }
                    )

            # Detect links (text regions that might be clickable)
            for region in text_regions:
                # Simple heuristic: text regions that are not too large
                if region["width"] < 300 and region["height"] < 50:
                    elements.append(
                        {
                            "type": "link",
                            "x": region["x"],
                            "y": region["y"],
                            "width": region["width"],
                            "height": region["height"],
                            "coordinates": (
                                region["x"] + region["width"] // 2,
                                region["y"] + region["height"] // 2,
                            ),
                            "description": f"Link at ({region['x']}, {region['y']})",
                        }
                    )

        except Exception as e:
            print(f"Error detecting interactive elements: {e}")

        return elements

    def _generate_recommended_actions(
        self,
        image: np.ndarray,
        interactive_elements: list[dict[str, Any]],
        user_goal: str,
        content_type: str,
    ) -> list[VisionAction]:
        """Generate recommended actions based on the image and user goal."""
        actions = []

        try:
            # Add actions based on content type
            if content_type == "form_login":
                actions.append(
                    VisionAction(
                        action_type="type",
                        target_description="username field",
                        confidence=0.9,
                        reasoning="This is a login page, likely needs username input",
                    )
                )
                actions.append(
                    VisionAction(
                        action_type="type",
                        target_description="password field",
                        confidence=0.9,
                        reasoning="This is a login page, likely needs password input",
                    )
                )
                actions.append(
                    VisionAction(
                        action_type="click",
                        target_description="login button",
                        confidence=0.8,
                        reasoning="This is a login page, needs to submit the form",
                    )
                )

            elif content_type == "form_search":
                actions.append(
                    VisionAction(
                        action_type="type",
                        target_description="search field",
                        confidence=0.9,
                        reasoning="This is a search page, needs search query input",
                    )
                )
                actions.append(
                    VisionAction(
                        action_type="click",
                        target_description="search button",
                        confidence=0.8,
                        reasoning="This is a search page, needs to submit the search",
                    )
                )

            # Add actions based on interactive elements
            for element in interactive_elements:
                if element["type"] == "button":
                    actions.append(
                        VisionAction(
                            action_type="click",
                            target_description=element["description"],
                            confidence=0.7,
                            coordinates=element["coordinates"],
                            reasoning=f"Detected clickable button: {element['description']}",
                        )
                    )

                elif element["type"] == "form_field":
                    actions.append(
                        VisionAction(
                            action_type="type",
                            target_description=element["description"],
                            confidence=0.6,
                            coordinates=element["coordinates"],
                            reasoning=f"Detected form field: {element['description']}",
                        )
                    )

                elif element["type"] == "link":
                    actions.append(
                        VisionAction(
                            action_type="click",
                            target_description=element["description"],
                            confidence=0.5,
                            coordinates=element["coordinates"],
                            reasoning=f"Detected clickable link: {element['description']}",
                        )
                    )

            # Add goal-specific actions
            if user_goal:
                if "search" in user_goal.lower():
                    actions.append(
                        VisionAction(
                            action_type="type",
                            target_description="search input",
                            confidence=0.8,
                            text_input=user_goal,
                            reasoning=f"User wants to search for: {user_goal}",
                        )
                    )
                elif "login" in user_goal.lower():
                    actions.append(
                        VisionAction(
                            action_type="click",
                            target_description="login or sign in button",
                            confidence=0.7,
                            reasoning="User wants to log in",
                        )
                    )

        except Exception as e:
            print(f"Error generating recommended actions: {e}")

        return actions

    def _generate_navigation_opportunities(
        self, links: list[dict[str, Any]], buttons: list[dict[str, Any]]
    ) -> list[str]:
        """Generate navigation opportunities from detected elements."""
        opportunities = []

        try:
            # Add opportunities from links
            for link in links:
                opportunities.append(f"Click link at ({link['x']}, {link['y']})")

            # Add opportunities from buttons
            for button in buttons:
                opportunities.append(f"Click button at ({button['x']}, {button['y']})")

            # Add general navigation suggestions
            if links:
                opportunities.append("Navigate to linked pages")
            if buttons:
                opportunities.append("Interact with buttons")

        except Exception as e:
            print(f"Error generating navigation opportunities: {e}")

        return opportunities
