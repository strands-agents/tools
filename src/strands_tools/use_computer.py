import os
from datetime import datetime
from typing import List, Optional

import pyautogui
from strands import tool

# Initialize pyautogui safely
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # Add small delay between actions for stability


@tool
def use_computer(
    action: str,
    x: Optional[int] = None,
    y: Optional[int] = None,
    text: Optional[str] = None,
    key: Optional[str] = None,
    region: Optional[List[int]] = None,
) -> str:
    """
    Control computer using mouse, keyboard, and capture screenshots.

    Args:
        action (str): The action to perform. Must be one of:
            - mouse_position: Get current mouse coordinates
            - click: Click at specified coordinates
            - move_mouse: Move mouse to specified coordinates
            - type: Type specified text
            - key_press: Press specified key
            - screenshot: Capture screen (optionally in specified region)
            - screen_size: Get screen dimensions
        x (int, optional): X coordinate for mouse actions
        y (int, optional): Y coordinate for mouse actions
        text (str, optional): Text to type
        key (str, optional): Key to press (e.g., 'enter', 'tab', 'space')
        region (List[int], optional): Region for screenshot [left, top, width, height]

    Returns:
        str: Description of the action result or error message
    """
    try:
        if action == "mouse_position":
            x, y = pyautogui.position()
            return f"Mouse position: ({x}, {y})"

        elif action == "screenshot":
            # Create screenshots directory if it doesn't exist
            screenshots_dir = "screenshots"
            if not os.path.exists(screenshots_dir):
                os.makedirs(screenshots_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(screenshots_dir, filename)

            # Take screenshot with optional region
            screenshot = pyautogui.screenshot(region=region) if region else pyautogui.screenshot()

            # Save locally
            screenshot.save(filepath)
            return f"Screenshot saved to {filepath}"

        elif action == "type":
            if not text:
                raise ValueError("No text provided for typing")
            pyautogui.typewrite(text)
            return f"Typed: {text}"

        elif action == "click":
            if x is None or y is None:
                raise ValueError("Missing x or y coordinates for click")

            # Move mouse smoothly to position and click
            pyautogui.moveTo(x, y, duration=0.5)
            pyautogui.click()
            return f"Clicked at ({x}, {y})"

        elif action == "move_mouse":
            if x is None or y is None:
                raise ValueError("Missing x or y coordinates for mouse movement")

            pyautogui.moveTo(x, y, duration=0.5)
            return f"Moved mouse to ({x}, {y})"

        elif action == "key_press":
            if not key:
                raise ValueError("No key specified for key press")
            pyautogui.press(key)
            return f"Pressed key: {key}"

        elif action == "screen_size":
            width, height = pyautogui.size()
            return f"Screen size: {width}x{height}"

        else:
            raise ValueError(f"Unknown action: {action}")

    except Exception as e:
        return f"Error: {str(e)}"
