import os
import platform
import subprocess
import time
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import cv2
import pyautogui
import pytesseract
import numpy as np
from strands import tool
from PIL import Image    
import psutil

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # Add small delay between actions for stability

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
    
    # Apply preprocessing to improve OCR accuracy
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optional: Apply slight Gaussian blur to reduce noise
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Optional: Apply adaptive thresholding for better text contrast
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Use pytesseract with improved configuration
    # PSM 11 = Sparse text. Find as much text as possible without assuming a particular structure.
    # PSM 6 = Assume a single uniform block of text (might be better for paragraphs)
    custom_config = f'--oem 3 --psm 11'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    
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
    for i in range(len(data['text'])):
        # Skip empty text
        if data['text'][i].strip() and float(data['conf'][i]) > min_confidence * 100:  # Tesseract confidence is 0-100
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Apply scaling if necessary
            adjusted_x = int(x * scale_factor_x)
            adjusted_y = int(y * scale_factor_y)
            adjusted_w = int(w * scale_factor_x)
            adjusted_h = int(h * scale_factor_y)
            
            # Calculate center with safety bounds checking
            center_x = adjusted_x + adjusted_w//2
            center_y = adjusted_y + adjusted_h//2
            
            # Ensure coordinates are within screen bounds
            center_x = max(0, min(center_x, screen_width))
            center_y = max(0, min(center_y, screen_height))
            
            results.append({
                'text': data['text'][i],
                'coordinates': {
                    'x': adjusted_x,
                    'y': adjusted_y,
                    'width': adjusted_w,
                    'height': adjusted_h,
                    'center_x': center_x,
                    'center_y': center_y,
                    'raw_x': x,  # Store original coordinates for debugging
                    'raw_y': y,
                    'scaling_applied': (scale_factor_x != 1.0 or scale_factor_y != 1.0)
                },
                'confidence': float(data['conf'][i]) / 100
            })
    
    return results

def open_application(app_name: str) -> str:
    """Helper function to open applications cross-platform."""
    system = platform.system().lower()

    try:
        if system == "windows":
            subprocess.Popen(f"start {app_name}", shell=True)
        elif system == "darwin":  # macOS
            subprocess.Popen(["open", "-a", app_name])
        elif system == "linux":
            subprocess.Popen(app_name.lower())
        return f"Launched {app_name}"
    except Exception as e:
        return f"Error launching {app_name}: {str(e)}"


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


def focus_application(app_name: str) -> bool:
    """Focus on the specified application window."""
    system = platform.system().lower()
    
    try:
        print("HELLO, we are attempting to focus on the target app")
        if system == "darwin":  # macOS
            # Use AppleScript to bring app to front
            script = f'tell application "{app_name}" to activate'
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
            time.sleep(0.2)  # Brief pause for window to focus
            return True
        elif system == "windows":
            # Use PowerShell to focus window
            script = f"Add-Type -AssemblyName Microsoft.VisualBasic; [Microsoft.VisualBasic.Interaction]::AppActivate('{app_name}')"
            subprocess.run(["powershell", "-Command", script], check=True, capture_output=True)
            time.sleep(0.2)
            return True
        elif system == "linux":
            # Use wmctrl if available
            subprocess.run(["wmctrl", "-a", app_name], check=True, capture_output=True)
            time.sleep(0.2)
            return True
    except Exception:
        return False
    
    return False


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
) -> str:
    """
    Control computer using mouse, keyboard, and capture screenshots.
    IMPORTANT: When performing actions within an application (clicking, typing, etc.), 
    always provide the app_name parameter to ensure proper focus on the target application.

    Args:
        action (str): The action to perform. Must be one of:
            - mouse_position: Get current mouse coordinates
            - click: Click at specified coordinates (requires app_name when clicking in application)
            - move_mouse: Move mouse to specified coordinates (requires app_name when moving to application elements)
            - drag: Click and drag from current position (requires app_name when dragging in application)
            - scroll: Scroll in specified direction (requires app_name when scrolling in application)
            - type: Type specified text (requires app_name)
            - key_press: Press specified key (requires app_name)
            - key_hold: Hold key combination (requires app_name)
            - hotkey: Press a hotkey combination (requires app_name)
            - screenshot: Capture screen (optionally in specified region or active window)
            - analyze_screenshot: Extract text and coordinates from screenshot
            - find_element: Find UI element by description and return coordinates
            - screen_size: Get screen dimensions
            - open_app: Open specified application
            - close_app: Close specified application
            - window_control: Control window (minimize, maximize, close)

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

    Returns:
        str: Description of the action result or error message
    
    Example Usage:
        # Correct usage - with app_name for application interaction
        use_computer(action="type", text="Hello world", app_name="Notepad")
        use_computer(action="click", x=100, y=200, app_name="Chrome")

        # Actions that don't require app_name
        use_computer(action="screen_size")
        use_computer(action="mouse_position")
    """

    # Auto-focus on target app before performing actions (except for certain actions)
    actions_requiring_focus = ["click", "type", "key_press", "key_hold", "hotkey", "drag", "scroll", "screenshot"]
    if action in actions_requiring_focus and app_name:
        focus_success = focus_application(app_name)
        if not focus_success:
            return f"Warning: Could not focus on {app_name}. Proceeding with action anyway."
    print(f"performing action: {action} in app: {app_name}") 
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

            # Handle different screenshot types
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()

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

            # Move mouse smoothly to position
            pyautogui.moveTo(x, y, duration=0.5)

            # Handle different click types
            click_type = click_type or "left"
            if click_type == "left":
                pyautogui.click()
            elif click_type == "right":
                pyautogui.rightClick()
            elif click_type == "double":
                pyautogui.doubleClick()
            elif click_type == "middle":
                pyautogui.middleClick()
            else:
                raise ValueError(f"Unknown click type: {click_type}")

            return f"{click_type.title()} clicked at ({x}, {y})"

        # drag does not currently work
        elif action == "drag":
            if drag_to_x is None or drag_to_y is None:
                raise ValueError("Missing drag destination coordinates")

            if x is not None and y is not None:
                pyautogui.moveTo(x, y, duration=0.3)

            pyautogui.dragTo(drag_to_x, drag_to_y, duration=1.0)
            return f"Dragged to ({drag_to_x}, {drag_to_y})"

        elif action == "scroll":
            direction = scroll_direction or "up"
            amount = scroll_amount or 3

            if x is not None and y is not None:
                pyautogui.moveTo(x, y, duration=0.3)

            if direction in ["up", "down"]:
                scroll_value = amount if direction == "up" else -amount
                pyautogui.scroll(scroll_value)
            elif direction in ["left", "right"]:
                pyautogui.hscroll(amount if direction == "right" else -amount)

            return f"Scrolled {direction} by {amount} steps"

        elif action == "move_mouse":
            if x is None or y is None:
                raise ValueError("Missing x or y coordinates for mouse movement")

            pyautogui.moveTo(x, y, duration=0.5)
            return f"Moved mouse to ({x}, {y})"

        elif action == "key_press":
            if not key:
                raise ValueError("No key specified for key press")

            if modifier_keys:
                # Handle key combinations
                keys_to_press = modifier_keys + [key]
                pyautogui.hotkey(*keys_to_press)
                return f"Pressed key combination: {'+'.join(keys_to_press)}"
            else:
                pyautogui.press(key)
                return f"Pressed key: {key}"

        elif action == "key_hold":
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
                
        elif action == "hotkey":
            if not hotkey_str:
                raise ValueError("No hotkey string provided for hotkey action")
                
            # Split the keys by '+'
            keys = hotkey_str.split('+')
            # Press the hotkey combination
            pyautogui.hotkey(*keys)
            print(f"clicked keys: {keys}")
            
            return f"Pressed hotkey combination: {hotkey_str}"

        elif action == "screen_size":
            width, height = pyautogui.size()
            return f"Screen size: {width}x{height}"

        elif action == "open_app":
            if not app_name:
                raise ValueError("No application name provided")
            return open_application(app_name)

        elif action == "close_app":
            if not app_name:
                raise ValueError("No application name provided")
            return close_application(app_name)
            
        elif action == "analyze_screenshot":
            # Use an existing screenshot or take a new one
            if screenshot_path:
                if not os.path.exists(screenshot_path):
                    raise ValueError(f"Screenshot not found at {screenshot_path}")
                image_path = screenshot_path
            else:
                # Create screenshots directory if it doesn't exist
                screenshots_dir = "screenshots"
                if not os.path.exists(screenshots_dir):
                    os.makedirs(screenshots_dir)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                filepath = os.path.join(screenshots_dir, filename)

                # Handle different screenshot types
                if region:
                    screenshot = pyautogui.screenshot(region=region)
                else:
                    screenshot = pyautogui.screenshot()

                # Save locally
                screenshot.save(filepath)
                image_path = filepath
                
            # Extract text and coordinates
            try:
                text_data = extract_text_from_image(image_path, min_confidence or 0.5)
                if not text_data:
                    return f"No text detected in screenshot {image_path}"
                
                # Format the result
                formatted_result = f"Detected {len(text_data)} text elements in {image_path}:\n\n"
                for idx, item in enumerate(text_data, 1):
                    coords = item['coordinates']
                    formatted_result += (
                        f"{idx}. Text: '{item['text']}'\n"
                        f"   Confidence: {item['confidence']:.2f}\n"
                        f"   Position: X={coords['x']}, Y={coords['y']}, W={coords['width']}, H={coords['height']}\n"
                        f"   Center: ({coords['center_x']}, {coords['center_y']})\n\n"
                    )
                
                return formatted_result
                
            except Exception as e:
                return f"Error analyzing screenshot: {str(e)}"

        else:
            raise ValueError(f"Unknown action: {action}")

    except Exception as e:
        return f"Error: {str(e)}"