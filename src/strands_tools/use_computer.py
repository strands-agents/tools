"""
Cross-platform computer automation tool for controlling mouse, keyboard, and screen interactions.

This module provides a comprehensive set of utilities for programmatically controlling
a computer through various input methods (mouse, keyboard) and screen analysis capabilities.
It's designed to work across multiple operating systems (Windows, macOS, Linux) with
appropriate fallbacks and platform-specific optimizations.

Features:
- Mouse control: positioning, clicking, dragging
- Keyboard input: typing, key presses, hotkeys
- Screen analysis: OCR-based text extraction from screen regions
- Application management: opening, closing, and focusing applications

The module uses PyAutoGUI for most operations, with platform-specific enhancements
for macOS (using Quartz), Windows, and Linux where needed. It includes comprehensive
error handling, input validation, and user consent mechanisms.

For OCR functionality, the module uses Tesseract OCR via the pytesseract library,
with image preprocessing optimizations to improve text recognition accuracy.
"""

import inspect
import logging
import os
import platform
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import psutil
import pyautogui
import pytesseract
from PIL import Image
from strands import tool

from strands_tools.utils.user_input import get_user_input

# Import libraries for macOS
if platform.system().lower() == "darwin":
    from Quartz.CoreGraphics import (
        CGEventCreateMouseEvent,
        CGEventPost,
        CGEventSetIntegerValueField,
        kCGEventLeftMouseDown,
        kCGEventLeftMouseUp,
        kCGHIDEventTap,
        kCGMouseButtonLeft,
        kCGMouseEventClickState,
    )

logger = logging.getLogger(__name__)


class UseComputerMethods:
    """
    Core implementation of computer automation methods for mouse, keyboard, and screen interactions.

    This class provides the underlying implementation for the use_computer tool,
    with methods for controlling mouse movement, clicks, keyboard input, and
    screen analysis. It handles platform-specific differences and includes
    appropriate error handling and validation.

    The class is designed with cross-platform compatibility in mind, with special
    handling for macOS, Windows, and Linux where necessary. It uses PyAutoGUI
    for most operations but falls back to platform-specific APIs when needed for
    better reliability or functionality.
    """

    def __init__(self):
        """
        Initialize the UseComputerMethods instance with safety settings.

        Sets up PyAutoGUI with failsafe mode enabled (moving mouse to corner aborts)
        and adds a small delay between actions for stability across platforms.
        """
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1  # Add small delay between actions for stability

    # Basic Computer Automation Actions
    def mouse_position(self):
        """
        Get the current mouse cursor position.

        Returns:
            str: String representation of current mouse coordinates in the format:
                 "Mouse position: (x, y)"
        """
        x, y = pyautogui.position()
        return f"Mouse position: ({x}, {y})"

    def click(self, x: int, y: int, click_type: str = "left") -> str:
        """Handle mouse clicks."""
        x, y = self._prepare_mouse_position(x, y)
        system = platform.system().lower()

        if click_type == "left":
            pyautogui.click()
        elif click_type == "right":
            pyautogui.rightClick()
        elif click_type == "double":
            if system == "darwin":
                self._native_mac_double_click(x, y)
            else:
                pyautogui.click(clicks=2, interval=0.2)
            time.sleep(0.1)
        elif click_type == "middle":
            pyautogui.middleClick()
        else:
            raise ValueError(f"Unknown click type: {click_type}")

        return f"{click_type.title()} clicked at ({x}, {y})"

    def move_mouse(self, x: int, y: int) -> str:
        """Move mouse to specified coordinates."""
        x, y = self._prepare_mouse_position(x, y, duration=0.5)
        return f"Moved mouse to ({x}, {y})"

    def drag(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        drag_to_x: Optional[int] = None,
        drag_to_y: Optional[int] = None,
        duration: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Perform a drag operation from one point to another.

        Args:
            x (Optional[int]): Starting X coordinate. If None, uses current mouse position.
            y (Optional[int]): Starting Y coordinate. If None, uses current mouse position.
            drag_to_x (int): Ending X coordinate.
            drag_to_y (int): Ending Y coordinate.
            duration (float): Duration of the drag operation in seconds.

        Returns:
            str: Description of the drag operation performed.
        """
        if drag_to_x is None or drag_to_y is None:
            raise ValueError("Missing drag destination coordinates")

        # If x and y are provided, move to that position first
        if x is not None and y is not None:
            x, y = self._prepare_mouse_position(x, y, duration=0.3)
        else:
            # If x and y are not provided, use current mouse position
            x, y = pyautogui.position()

        try:
            # Use pyautogui.drag() which handles the complete drag operation
            pyautogui.drag(drag_to_x - x, drag_to_y - y, duration=duration, button="left")
            return f"Dragged from ({x}, {y}) to ({drag_to_x}, {drag_to_y})"
        except Exception as e:
            raise Exception(f"Drag operation failed: {str(e)}") from e

    def scroll(
        self,
        x: Optional[int],
        y: Optional[int],
        app_name: Optional[str],
        scroll_direction: str = "up",
        scroll_amount: int = 15,
        click_first: bool = True,
    ) -> str:
        """Handle scrolling actions."""
        if x is None or y is None:
            if app_name:
                screen_width, screen_height = pyautogui.size()
                x = screen_width // 2
                y = screen_height // 2
                logger.info(f"No coordinates provided for scroll, using app center: ({x}, {y})")
            else:
                raise ValueError(
                    "Missing x or y coordinates for scrolling. "
                    "For scrolling to work, mouse must be over the scrollable area."
                )

        pyautogui.moveTo(x, y, duration=0.3)

        # Click to ensure the scrollable area has focus
        if click_first:
            pyautogui.click()
            time.sleep(0.1)

        if scroll_direction in ["up", "down"]:
            scroll_value = scroll_amount if scroll_direction == "up" else -scroll_amount
            pyautogui.scroll(scroll_value)

        elif scroll_direction in ["left", "right"]:
            # horizontal scrolling is handled differently on mac
            if platform.system().lower() == "darwin":
                # Use keycode for macOS
                keycode = 124 if scroll_direction == "right" else 123  # macOS keycodes
                for _ in range(scroll_amount):
                    subprocess.run(
                        ["osascript", "-e", f'tell application "System Events" to key code {keycode}'], check=False
                    )
                    time.sleep(0.01)
            else:
                # Use hscroll for Windows/Linux
                scroll_value = scroll_amount if scroll_direction == "right" else -scroll_amount
                pyautogui.hscroll(scroll_value)

        return f"Scrolled {scroll_direction} by {scroll_amount} steps at coordinates ({x}, {y})"

    def type(self, text: str) -> str:
        """Type specified text."""
        if not text:
            raise ValueError("No text provided for typing")
        pyautogui.typewrite(text)
        return f"Typed: {text}"

    def key_press(self, key: str, modifier_keys: Optional[List[str]] = None) -> str:
        """Handle key press actions."""
        if not key:
            raise ValueError("No key specified for key press")

        if modifier_keys:
            keys_to_press = modifier_keys + [key]
            pyautogui.hotkey(*keys_to_press)
            return f"Pressed key combination: {'+'.join(keys_to_press)}"
        else:
            pyautogui.press(key)
            return f"Pressed key: {key}"

    def key_hold(
        self, key: Optional[str] = None, modifier_keys: Optional[List[str]] = None, hold_duration: float = 0.1, **kwargs
    ) -> str:
        if not key:
            raise ValueError("No key specified for key hold")

        if modifier_keys:
            # Hold modifier keys and press main key
            for mod_key in modifier_keys:
                pyautogui.keyDown(mod_key)

            pyautogui.press(key)

            for mod_key in reversed(modifier_keys):
                pyautogui.keyUp(mod_key)

            return f"Held {'+'.join(modifier_keys)} and pressed {key}"
        else:
            pyautogui.keyDown(key)
            time.sleep(0.1)
            pyautogui.keyUp(key)
            return f"Held and released key: {key}"

    def hotkey(self, hotkey_str: str) -> str:
        """Handle hotkey combinations."""
        if not hotkey_str:
            raise ValueError("No hotkey string provided for hotkey action")

        keys = hotkey_str.split("+")

        if platform.system().lower() == "darwin":  # macOS
            keys = ["command" if k.lower() == "cmd" else k for k in keys]

        pyautogui.hotkey(*keys)
        logger.info(f"Executing hotkey combination: {keys}")

        return f"Pressed hotkey combination: {hotkey_str}"

    def analyze_screen(
        self,
        screenshot_path: Optional[str] = None,
        region: Optional[List[int]] = None,
        min_confidence: float = 0.5,
        send_screenshot: bool = False,
    ) -> Dict:
        """
        Capture a screenshot and analyze it for text content using OCR.

        This method takes a screenshot of the current screen (or a specified region),
        extracts text using OCR, and returns both the text analysis and optionally
        the screenshot itself.

        Args:
            screenshot_path: Path to an existing screenshot file to analyze instead of
                           capturing a new one. If None, a new screenshot is taken.
            region: Optional list of [left, top, width, height] defining the screen
                  region to capture. If None, the entire screen is captured.
            min_confidence: Minimum confidence threshold (0.0-1.0) for OCR text detection.
                          Higher values improve precision but may miss some text.
            send_screenshot: Whether to include the actual screenshot image in the return value.
                           Set to True if you want the screenshot to be sent to the model/agent
                           for visual inspection. Set to False to only return the text analysis,
                           which is useful for privacy or when bandwidth/tokens are a concern.

        Returns:
            Dict: Dictionary containing status and content with the following structure:
                {
                    "status": "success" or "error",
                    "content": [
                        {"text": "Text analysis results"},
                        {"image": {...}}  # Only included if send_screenshot=True
                    ]
                }

        Note:
            Large screenshots (>5MB) will automatically disable send_screenshot to prevent
            exceeding model context limits, regardless of the parameter value.
        """
        # Get text analysis results using Tesseract OCR
        analysis_results = handle_analyze_screenshot_pytesseract(screenshot_path, region, min_confidence)

        # Prepare text analysis result
        text_result = analysis_results.get("text_result", "No text analysis available")

        # Prepare image for the LLM only if send_screenshot is True
        image_path = analysis_results.get("image_path")
        image_content = None

        if send_screenshot:
            # Check the file size first as a quick filter
            if os.path.exists(image_path):
                # File size check - consider base64 encoding overhead (approximately 33%)
                # Base64 encoding increases size by ~33% (4/3) plus some additional overhead
                raw_size = os.path.getsize(image_path)
                estimated_encoded_size = int(raw_size * 1.37)  # Base64 size + buffer
                logger.info(
                    f"Raw image size: {raw_size / 1024 / 1024:.2f}MB, \
                    estimated encoded size: {estimated_encoded_size / 1024 / 1024:.2f}MB"
                )

                if estimated_encoded_size > 5 * 1024 * 1024:
                    logger.info(
                        f"Image size after base64 encoding would exceed 5MB limit \
                        ({estimated_encoded_size} bytes), disabling screenshot"
                    )
                    send_screenshot = False
                else:
                    # Only read and prepare the image if it's likely to be within size limits
                    image_content = handle_sending_results_to_llm(image_path)

                    # Get actual bytes length (this is the important check)
                    if (
                        "image" in image_content
                        and "source" in image_content["image"]
                        and "bytes" in image_content["image"]["source"]
                    ):
                        actual_bytes_length = len(image_content["image"]["source"]["bytes"])
                        logger.info(f"Actual image bytes size: {actual_bytes_length / 1024 / 1024:.2f}MB")
                        if actual_bytes_length > 5 * 1024 * 1024:
                            logger.info(
                                f"Image bytes exceed 5MB limit ({actual_bytes_length} bytes), disabling screenshot"
                            )
                            send_screenshot = False
                            image_content = {"text": "Image too large to display (exceeds 5MB limit)"}

        # Clean up if needed
        should_delete = analysis_results.get("should_delete", False)
        if should_delete and os.path.exists(image_path):
            delete_screenshot(image_path)

        # Create content list, conditionally including image based on send_screenshot parameter
        content_list = [{"text": text_result}]  # Always include text analysis results

        # Add image content only if send_screenshot is True and we have valid image content
        if send_screenshot and image_content:
            logger.info("Adding screenshot to the content being returned")
            content_list.append(image_content)

        return {
            "status": "success",
            "content": content_list,
        }

    def screen_size(self) -> str:
        """
        Get the screen dimensions of the primary display.

        Returns:
            str: String representation of screen dimensions in the format:
                 "Screen size: widthxheight"
        """
        width, height = pyautogui.size()
        return f"Screen size: {width}x{height}"

    def open_app(self, app_name):
        logger.info(f"Opening application: {app_name}")
        if not app_name:
            raise ValueError("No application name provided")
        return open_application(app_name)

    def close_app(self, app_name):
        if not app_name:
            raise ValueError("No application name provided")
        return close_application(app_name)

    # I cannot find a way to double click using pyautoguis built in functions on macos
    # This function uses lower level mac functions to double click
    def _native_mac_double_click(self, x: int, y: int):
        """
        Perform a native macOS double-click operation using Quartz APIs.

        This method provides a more reliable double-click implementation for
        macOS compared to PyAutoGUI's implementation, using the native Quartz
        CoreGraphics framework to generate hardware-level mouse events.

        Args:
            x: X-coordinate for the double-click position
            y: Y-coordinate for the double-click position

        Note:
            This is a private helper method used internally by the click method
            when running on macOS and when a double-click is requested.
        """

        for i in range(2):
            click_down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
            click_up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)

            # Set click state: 1 = first click, 2 = second click
            CGEventSetIntegerValueField(click_down, kCGMouseEventClickState, i + 1)
            CGEventSetIntegerValueField(click_up, kCGMouseEventClickState, i + 1)

            CGEventPost(kCGHIDEventTap, click_down)
            CGEventPost(kCGHIDEventTap, click_up)

            # Small delay between clicks for proper double-click timing
            if i == 0:
                time.sleep(0.05)

    def _prepare_mouse_position(self, x: int, y: int, duration: float = 0.1) -> tuple[int, int]:
        """Move mouse to specified coordinates with error handling."""
        if x is None or y is None:
            raise ValueError("Missing x or y coordinates")
        pyautogui.moveTo(x, y, duration=duration)
        time.sleep(0.05)  # Let pointer settle
        return x, y


def create_screenshot(region: Optional[List[int]] = None) -> str:
    """
    Create and save a screenshot to disk.

    Takes a screenshot of the entire screen or a specified region and saves it
    to the 'screenshots' directory with a timestamped filename. Creates the
    directory if it doesn't exist.

    Args:
        region: Optional list of [left, top, width, height] specifying screen region
               to capture. If None, captures the entire screen.

    Returns:
        str: Path to the saved screenshot file.
    """
    screenshots_dir = "screenshots"
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join(screenshots_dir, filename)

    if region:
        screenshot = pyautogui.screenshot(region=region)
    else:
        screenshot = pyautogui.screenshot()

    screenshot.save(filepath)
    return filepath


# Helper function to sort the text extracted from the screenshots
def group_text_by_lines(text_data: List[Dict[str, Any]], line_threshold: int = 10) -> List[List[Dict[str, Any]]]:
    """
    Group extracted text elements into lines based on vertical proximity.

    This function organizes OCR-extracted text elements into logical lines
    by analyzing their y-coordinates. Text elements are considered part of
    the same line if their vertical positions are within the specified threshold.
    Elements in each line are then sorted horizontally (by x-coordinate) to
    preserve proper reading order.

    Args:
        text_data: List of text elements with coordinate information.
        line_threshold: Maximum vertical distance (in pixels) for two elements
                      to be considered part of the same line. Default is 10 pixels.

    Returns:
        List of lists, where each inner list contains text elements belonging to
        the same line, sorted from left to right.
    """
    if not text_data:
        return []

    # Sort by y-coordinate
    sorted_data = sorted(text_data, key=lambda x: x["coordinates"]["y"])

    lines = []
    current_line = [sorted_data[0]]

    for item in sorted_data[1:]:
        # If y-coordinate is close to the previous item, keep in the same line
        if abs(item["coordinates"]["y"] - current_line[-1]["coordinates"]["y"]) <= line_threshold:
            current_line.append(item)
        else:
            # Sort current line by x-coordinate to get words in order
            current_line.sort(key=lambda x: x["coordinates"]["x"])
            lines.append(current_line)
            current_line = [item]

    # For the last line
    if current_line:
        current_line.sort(key=lambda x: x["coordinates"]["x"])
        lines.append(current_line)

    return lines


def extract_text_from_image(image_path: str, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """
    Extract text and coordinates from an image using Tesseract OCR.

    Args:
        image_path: Path to the image file
        min_confidence: Minimum confidence level for OCR text detection (0.0-1.0)

    Returns:
        List of dictionaries with text and its coordinates
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions for potential scaling adjustments
    img_height, img_width = img.shape[:2]

    # Scale image if it's too small for good OCR (upscale by 2x if smaller than 1000px)
    scale_factor = 1.0
    if img_width < 1000 or img_height < 1000:
        scale_factor = 2.0
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Apply preprocessing to improve OCR accuracy
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply noise reduction
    denoised = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Contrast Limited Adaptive Histogram Equalization
    enhanced = clahe.apply(denoised)

    # Apply sharpening kernel to improve text clarity
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    gray = sharpened  # Use the enhanced image

    # Try multiple OCR configurations for better text detection
    # Include character whitelist for common characters to reduce noise
    char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]{}\"'-_@#$%^&*+=<>/|\\~`"

    configs = [
        f"--oem 3 --psm 11 -c tessedit_char_whitelist={char_whitelist}",  # Sparse text with whitelist
        "--oem 3 --psm 11",  # Sparse text without whitelist
        "--oem 3 --psm 6",  # Single uniform block
        "--oem 3 --psm 3",  # Fully automatic page segmentation
        "--oem 3 --psm 8",  # Single word
    ]

    all_results = []
    for config in configs:
        try:
            data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
            all_results.append(data)
        except Exception:
            continue

    # Use the configuration that detected the most text
    if not all_results:
        raise ValueError("OCR failed with all configurations")

    data = max(all_results, key=lambda d: len([t for t in d["text"] if t.strip()]))

    # Check for potential scaling issues by comparing with screen resolution
    screen_width, screen_height = pyautogui.size()
    scale_factor_x = 1.0
    scale_factor_y = 1.0

    # If the image dimensions don't match the screen dimensions, calculate scaling factors
    if abs(img_width - screen_width) > 5 or abs(img_height - screen_height) > 5:
        scale_factor_x = screen_width / img_width
        scale_factor_y = screen_height / img_height

    # Extract text and coordinates
    results = []
    for i in range(len(data["text"])):
        if data["text"][i].strip() and float(data["conf"][i]) > min_confidence * 100:  # Tesseract confidence is 0-100
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

            # Apply scaling if necessary (account for both image upscaling and screen scaling)
            adjusted_x = int((x / scale_factor) * scale_factor_x)
            adjusted_y = int((y / scale_factor) * scale_factor_y)
            adjusted_w = int((w / scale_factor) * scale_factor_x)
            adjusted_h = int((h / scale_factor) * scale_factor_y)

            # Calculate center with safety bounds checking
            center_x = adjusted_x + adjusted_w // 2
            center_y = adjusted_y + adjusted_h // 2

            # Ensure coordinates are within screen bounds
            center_x = max(0, min(center_x, screen_width))
            center_y = max(0, min(center_y, screen_height))

            results.append(
                {
                    "text": data["text"][i],
                    "coordinates": {
                        "x": adjusted_x,
                        "y": adjusted_y,
                        "width": adjusted_w,
                        "height": adjusted_h,
                        "center_x": center_x,
                        "center_y": center_y,
                        "raw_x": x,  # Store original coordinates for debugging
                        "raw_y": y,
                        "scaling_applied": (scale_factor_x != 1.0 or scale_factor_y != 1.0),
                    },
                    "confidence": float(data["conf"][i]) / 100,
                }
            )

    # Group text into lines for better organization
    lines = group_text_by_lines(results)

    # Add line information to each text element
    for line_idx, line in enumerate(lines):
        for item in line:
            item["line_number"] = line_idx
            item["line_text"] = " ".join([text_item["text"] for text_item in line])

    return results


def open_application(app_name: str) -> str:
    """
    Launch an application cross-platform.

    Attempts to open the specified application using platform-appropriate methods.
    Includes support for common application name variations and aliases through
    an internal mapping system.

    Args:
        app_name: Name of the application to open. Common variations are mapped
                to their standard names (e.g., "chrome" to "Google Chrome").

    Returns:
        str: Success or error message detailing the result of the operation.

    Platform Support:
        - Windows: Uses the 'start' command
        - macOS: Uses the 'open -a' command
        - Linux: Attempts to run app_name directly as a command
    """
    system = platform.system().lower()

    # Map common app name variations to their actual names
    app_mappings = {
        "outlook": "Microsoft Outlook",
        "word": "Microsoft Word",
        "excel": "Microsoft Excel",
        "powerpoint": "Microsoft PowerPoint",
        "chrome": "Google Chrome",
        "firefox": "Firefox",
        "safari": "Safari",
        "notes": "Notes",
        "calculator": "Calculator",
        "terminal": "Terminal",
        "finder": "Finder",
    }

    # Use mapped name if available, otherwise use original
    actual_app_name = app_mappings.get(app_name.lower(), app_name)

    try:
        if system == "windows":
            result = subprocess.run(f"start {actual_app_name}", shell=True, capture_output=True, text=True)
        elif system == "darwin":  # macOS
            result = subprocess.run(["open", "-a", actual_app_name], capture_output=True, text=True)
        elif system == "linux":
            result = subprocess.run([actual_app_name.lower()], capture_output=True, text=True)

        if result.returncode == 0:
            return f"Launched {actual_app_name}"
        else:
            return f"Unable to find application named '{actual_app_name}'"
    except Exception as e:
        return f"Error launching {actual_app_name}: {str(e)}"


def close_application(app_name: str) -> str:
    """Helper function to close applications cross-platform."""
    if not psutil:
        return "psutil not available - cannot close applications"

    try:
        closed_count = 0
        for proc in psutil.process_iter(["pid", "name"]):
            if app_name.lower() in proc.info["name"].lower():
                proc.terminate()
                closed_count += 1

        if closed_count > 0:
            return f"Closed {closed_count} instance(s) of {app_name}"
        else:
            return f"No running instances of {app_name} found"
    except Exception as e:
        return f"Error closing {app_name}: {str(e)}"


def focus_application(app_name: str, timeout: float = 2.0) -> bool:
    """
    Focus on (bring to foreground) the specified application window with timeout.

    Uses platform-specific methods to activate and bring the specified application
    to the foreground, enabling subsequent interaction with its windows.

    Args:
        app_name: Name of the application to focus on.
        timeout: Maximum time in seconds to wait for focus operation (default: 2.0).
                 If focusing takes longer than this, the function will return False.

    Returns:
        bool: True if the focus operation was successful, False otherwise.

    Platform Support:
        - macOS: Uses AppleScript's 'activate' command
        - Windows: Uses PowerShell's AppActivate method
        - Linux: Attempts to use wmctrl if available
    """
    system = platform.system().lower()
    start_time = time.time()

    try:
        if system == "darwin":  # macOS
            # Use AppleScript to bring app to front with timeout
            script = f'tell application "{app_name}" to activate'

            # Set up a process with timeout
            try:
                result = subprocess.run(["osascript", "-e", script], check=True, capture_output=True, timeout=timeout)
                if result.returncode != 0:
                    logger.warning(f"Focus application returned non-zero exit code: {result.returncode}")
                    return False

                # Brief pause for window to focus, but respect overall timeout
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time > 0:
                    time.sleep(min(0.2, remaining_time))
                return True
            except subprocess.TimeoutExpired:
                logger.warning(f"Focus operation timed out after {timeout} seconds for app: {app_name}")
                return False

        elif system == "windows":
            # Use PowerShell to focus window
            script = (
                f"Add-Type -AssemblyName Microsoft.VisualBasic; "
                f"[Microsoft.VisualBasic.Interaction]::AppActivate('{app_name}')"
            )
            try:
                result = subprocess.run(
                    ["powershell", "-Command", script], check=True, capture_output=True, timeout=timeout
                )
                if result.returncode != 0:
                    return False

                # Brief pause for window to focus, but respect overall timeout
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time > 0:
                    time.sleep(min(0.2, remaining_time))
                return True
            except subprocess.TimeoutExpired:
                logger.warning(f"Focus operation timed out after {timeout} seconds for app: {app_name}")
                return False

        elif system == "linux":
            # Use wmctrl if available
            try:
                result = subprocess.run(["wmctrl", "-a", app_name], check=True, capture_output=True, timeout=timeout)
                if result.returncode != 0:
                    return False

                # Brief pause for window to focus, but respect overall timeout
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time > 0:
                    time.sleep(min(0.2, remaining_time))
                return True
            except subprocess.TimeoutExpired:
                logger.warning(f"Focus operation timed out after {timeout} seconds for app: {app_name}")
                return False
    except Exception as e:
        logger.warning(f"Error focusing application {app_name}: {str(e)}")
        return False

    return False


def delete_screenshot(filepath: str) -> None:
    """
    Delete a screenshot file from disk.

    Attempts to remove the specified file, handling errors gracefully without
    interrupting program flow. Errors are logged as warnings.

    Args:
        filepath: Path to the screenshot file to be deleted.

    Returns:
        None

    Note:
        Errors during deletion are logged but do not raise exceptions to avoid
        interrupting the main operation flow.
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        # Log the error but continue execution
        logger.warning(f"Failed to delete screenshot file '{filepath}': {str(e)}")
        # We don't want to fail the entire operation just because of a cleanup issue


def handle_sending_results_to_llm(image_path: str) -> dict:
    """
    Prepare the screenshot image to be sent to the LLM.

    Args:
        image_path: Path to the screenshot image

    Returns:
        Dictionary containing the image data formatted for the Converse API
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return {"text": f"Screenshot image not found at path: {image_path}"}

        # Read the image file as binary data
        with open(image_path, "rb") as file:
            file_bytes = file.read()

        # Determine image format using PIL
        with Image.open(image_path) as img:
            image_format = img.format.lower()
            if image_format not in ["png", "jpeg", "jpg", "gif", "webp"]:
                image_format = "png"  # Default to PNG if format is not recognized

        # Return the image data in the format expected by the Converse API
        return {"image": {"format": image_format, "source": {"bytes": file_bytes}}}
    except Exception as e:
        return {"text": f"Error preparing image for LLM: {str(e)}"}


def handle_analyze_screenshot_pytesseract(
    screenshot_path: Optional[str], region: Optional[List[int]], min_confidence: float = 0.5
) -> dict:
    """Extract text and coordinates from screenshot using Tesseract OCR."""
    # Check if screenshot_path was given then do not delete the screenshot
    if screenshot_path:
        if not os.path.exists(screenshot_path):
            raise ValueError(f"Screenshot not found at {screenshot_path}")
        image_path = screenshot_path
        should_delete = False
    else:
        image_path = create_screenshot(region)
        should_delete = True

    try:
        text_data = extract_text_from_image(image_path, min_confidence)
        if not text_data:
            result = f"No text detected in screenshot {image_path}"
        else:
            formatted_result = f"Detected {len(text_data)} text elements in {image_path}:\n\n"
            for idx, item in enumerate(text_data, 1):
                coords = item["coordinates"]
                formatted_result += (
                    f"{idx}. Text: '{item['text']}'\n"
                    f"   Confidence: {item['confidence']:.2f}\n"
                    f"   Position: X={coords['x']}, Y={coords['y']}, "
                    f"W={coords['width']}, H={coords['height']}\n"
                    f"   Center: ({coords['center_x']}, {coords['center_y']})\n\n"
                )
            result = formatted_result

        # Return the text result and keep the image path for sending to LLM
        return {"text_result": result, "image_path": image_path, "should_delete": should_delete}

    except Exception as e:
        if should_delete:
            delete_screenshot(image_path)
        raise RuntimeError(f"Error analyzing screenshot: {str(e)}") from e


@tool
def use_computer(
    action: str,
    x: Optional[int] = None,
    y: Optional[int] = None,
    text: Optional[str] = None,
    key: Optional[str] = None,
    region: Optional[List[int]] = None,
    app_name: Optional[str] = None,
    click_type: Optional[str] = None,
    modifier_keys: Optional[List[str]] = None,
    scroll_direction: Optional[str] = None,
    scroll_amount: Optional[int] = None,
    drag_to_x: Optional[int] = None,
    drag_to_y: Optional[int] = None,
    screenshot_path: Optional[str] = None,
    hotkey_str: Optional[str] = None,
    min_confidence: Optional[float] = 0.5,
    send_screenshot: Optional[bool] = False,
    focus_timeout: Optional[float] = 2.0,
) -> Dict:
    """
    Control computer using mouse, keyboard, and capture screenshots.
    IMPORTANT: When performing actions within an application (clicking, typing, etc.),
    always provide the app_name parameter to ensure proper focus on the target application.

    NOTE ON SCREENSHOTS: Do NOT include send_screenshot=True unless the user has EXPLICITLY
    requested to see the actual screenshot. By default, only text analysis is returned.

    Args:
        action (str): The action to perform. Must be one of:
            - mouse_position: Get current mouse coordinates
            - click: Click at specified coordinates (requires app_name when clicking in application)
            - move_mouse: Move mouse to specified coordinates (requires app_name when moving to application elements)
            - drag: Click and drag from current position (requires app_name when dragging in application)
            - scroll: Scroll in specified direction
                (requires x,y coordinates and app_name when scrolling in application)
            - type: Type specified text (requires app_name)
            - key_press: Press specified key (requires app_name)
            - key_hold: Hold key combination (requires app_name)
            - hotkey: Press a hotkey combination (requires app_name)
            - analyze_screen: Capture screenshot and extract text in a single operation (recommended)
            - screen_size: Get screen dimensions
            - open_app: Open specified application
            - close_app: Close specified application

        app_name (str): Name of application to focus on before performing actions.
            Required for all actions that interact with application windows
            (clicking, typing, key presses, etc.). Examples: "Chrome", "Firefox", "Notepad"
        x (int, optional): X coordinate for mouse actions
        y (int, optional): Y coordinate for mouse actions
        text (str, optional): Text to type
        key (str, optional): Key to press (e.g., 'enter', 'tab', 'space')
        region (List[int], optional): Region for screenshot [left, top, width, height]
        click_type (str, optional): Type of click ('left', 'right', 'double', 'middle')
        modifier_keys (List[str], optional): Modifier keys to hold ('shift', 'ctrl', 'alt', 'command')
        scroll_direction (str, optional): Scroll direction ('up', 'down', 'left', 'right')
        scroll_amount (int, optional): Number of scroll steps (default: 3)
        drag_to_x (int, optional): X coordinate to drag to
        drag_to_y (int, optional): Y coordinate to drag to
        screenshot_path (str, optional): Path to screenshot file for analysis
        hotkey_str (str, optional): Hotkey combination string (e.g., 'ctrl+c', 'alt+tab', 'ctrl+shift+esc')
        min_confidence (float, optional): Minimum confidence level for OCR text detection (default: 0.5)
        send_screenshot (bool, optional): Whether to send the screenshot to the model (default: False).
            IMPORTANT: Only set this to True when a user EXPLICITLY asks to see the screenshot.
            Setting this parameter increases token usage significantly and may expose sensitive
            information from the user's screen. Default is False which returns only text analysis.
            Large screenshots (>5MB) will be automatically rejected to prevent context overflow.
            Set to True to include the actual screenshot image in the return value,
            allowing the agent to visually inspect the screen. Set to False to only
            return the text analysis results, which is more privacy-conscious and uses
            fewer tokens. Note: Large images (>5MB) will not be sent regardless of
            this setting to prevent exceeding model context limits.
        focus_timeout (float, optional): Maximum time in seconds to wait for application focus.
            Default is 2.0 seconds. If focusing takes longer than this, the function will
            proceed with the action anyway but will issue a warning. This is especially
            useful for menu interactions which can sometimes get stuck.

    Returns:
        Dict: For most actions, returns a simple dictionary with status and text content.
              For analyze_screen, returns both text analysis results and the image content
              in a format that can be processed by the model.
    """
    all_params = locals()
    params = [
        f"{k}: {v}"
        for k, v in all_params.items()
        if v is not None
        and not (k == "min_confidence" and v == 0.5)
        and not (k == "send_screenshot" and v is False)
        and not (k == "focus_timeout" and v == 2.0)
    ]

    strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    if not strands_dev:
        params_str = "\n ".join(params)
        user_input = get_user_input(f"Do you want to proceed with {params_str}? (y/n)")
        if user_input.lower().strip() != "y":
            cancellation_reason = (
                user_input if user_input.strip() != "n" else get_user_input("Please provide a reason for cancellation:")
            )
            error_message = f"Python code execution cancelled by the user. Reason: {cancellation_reason}"
            return {
                "status": "error",
                "content": [{"text": error_message}],
            }

    # Special handling for menu interactions - longer timeout for "File", "Edit", "View", etc.
    if action == "click" and app_name and (y is not None and y < 50):
        # Top menu bar typically is at the top of the screen, with y < 50
        logger.info(f"Detected potential menu bar interaction at y={y}. Using extended focus timeout.")
        focus_timeout = max(focus_timeout, 3.0)  # Use at least 3 seconds for menu interactions

    # Auto-focus on target app before performing actions (except for certain actions)
    actions_requiring_focus = [
        "click",
        "type",
        "key_press",
        "key_hold",
        "hotkey",
        "drag",
        "scroll",
        "scroll_to_bottom",
        "screenshot",
        "analyze_screen",
    ]
    if action in actions_requiring_focus and app_name:
        # Use the timeout parameter
        focus_success = focus_application(app_name, timeout=focus_timeout)
        if not focus_success:
            warning_message = (
                f"Warning: Could not focus on {app_name} within {focus_timeout} seconds. Proceeding with action anyway."
            )
            logger.warning(warning_message)
            # For menu interactions, if focus fails, take a screenshot to help diagnose what's happening

    logger.info(f"Performing action: {action} in app: {app_name}")

    computer = UseComputerMethods()

    # This is so we only pass the parameters that are called with use_computer
    method_params = {
        "x": x,
        "y": y,
        "text": text,
        "key": key,
        "region": region,
        "app_name": app_name,
        "click_type": click_type,
        "modifier_keys": modifier_keys,
        "scroll_direction": scroll_direction,
        "scroll_amount": scroll_amount,
        "drag_to_x": drag_to_x,
        "drag_to_y": drag_to_y,
        "screenshot_path": screenshot_path,
        "hotkey_str": hotkey_str,
        "min_confidence": min_confidence,
        "send_screenshot": send_screenshot,
    }
    # Remove None values
    method_params = {k: v for k, v in method_params.items() if v is not None}

    try:
        method = getattr(computer, action, None)
        if method:
            # Get method signature to only pass valid parameters
            sig = inspect.signature(method)
            valid_params = {k: v for k, v in method_params.items() if k in sig.parameters}
            result = method(**valid_params)

            # If it's already a dictionary with the expected format, return it directly
            if isinstance(result, dict) and "status" in result and "content" in result:
                return result

            # Otherwise, wrap the result in our standard format
            return {"status": "success", "content": [{"text": result}]}
        else:
            return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}
