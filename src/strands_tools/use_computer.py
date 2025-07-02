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

def group_text_by_lines(text_data: List[Dict[str, Any]], line_threshold: int = 10) -> List[List[Dict[str, Any]]]:
    """Group text elements into lines based on y-coordinate proximity."""
    if not text_data:
        return []
    
    # Sort by y-coordinate
    sorted_data = sorted(text_data, key=lambda x: x['coordinates']['y'])
    
    lines = []
    current_line = [sorted_data[0]]
    
    for item in sorted_data[1:]:
        # If y-coordinate is close to the previous item, add to current line
        if abs(item['coordinates']['y'] - current_line[-1]['coordinates']['y']) <= line_threshold:
            current_line.append(item)
        else:
            # Sort current line by x-coordinate for proper reading order
            current_line.sort(key=lambda x: x['coordinates']['x'])
            lines.append(current_line)
            current_line = [item]
    
    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda x: x['coordinates']['x'])
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
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Apply sharpening kernel to improve text clarity
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Use the enhanced image for OCR
    gray = sharpened
    
    # Try multiple OCR configurations for better text detection
    # Include character whitelist for common characters to reduce noise
    char_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]{}"\'-_@#$%^&*+=<>/|\\~`'
    
    configs = [
        f'--oem 3 --psm 11 -c tessedit_char_whitelist={char_whitelist}',  # Sparse text with whitelist
        '--oem 3 --psm 11',  # Sparse text without whitelist
        '--oem 3 --psm 6',   # Single uniform block
        '--oem 3 --psm 3',   # Fully automatic page segmentation
        '--oem 3 --psm 8'    # Single word
    ]
    
    all_results = []
    for config in configs:
        try:
            data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
            all_results.append(data)
        except:
            continue
    
    # Use the configuration that detected the most text
    if not all_results:
        raise ValueError("OCR failed with all configurations")
    
    data = max(all_results, key=lambda d: len([t for t in d['text'] if t.strip()]))
    
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
            
            # Apply scaling if necessary (account for both image upscaling and screen scaling)
            adjusted_x = int((x / scale_factor) * scale_factor_x)
            adjusted_y = int((y / scale_factor) * scale_factor_y)
            adjusted_w = int((w / scale_factor) * scale_factor_x)
            adjusted_h = int((h / scale_factor) * scale_factor_y)
            
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
    
    # Group text into lines for better organization
    lines = group_text_by_lines(results)
    
    # Add line information to each text element
    for line_idx, line in enumerate(lines):
        for item in line:
            item['line_number'] = line_idx
            item['line_text'] = ' '.join([text_item['text'] for text_item in line])
    
    return results

def open_application(app_name: str) -> str:
    """Helper function to open applications cross-platform."""
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
        "finder": "Finder"
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


def handle_mouse_position() -> str:
    """Get current mouse coordinates."""
    x, y = pyautogui.position()
    return f"Mouse position: ({x}, {y})"


def handle_screenshot(region: Optional[List[int]] = None) -> str:
    """Capture screen screenshot."""
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
    return f"Screenshot saved to {filepath}"


def handle_type(text: str) -> str:
    """Type specified text."""
    if not text:
        raise ValueError("No text provided for typing")
    pyautogui.typewrite(text)
    return f"Typed: {text}"


def handle_click(x: int, y: int, click_type: str = "left") -> str:
    """Handle mouse clicks."""
    if x is None or y is None:
        raise ValueError("Missing x or y coordinates for click")

    pyautogui.moveTo(x, y, duration=0.5)

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


def handle_scroll(x: Optional[int], y: Optional[int], app_name: Optional[str], 
                 scroll_direction: str = "up", scroll_amount: int = 15) -> str:
    """Handle scrolling actions."""
    if x is None or y is None:
        if app_name:
            screen_width, screen_height = pyautogui.size()
            x = screen_width // 2
            y = screen_height // 2
            print(f"No coordinates provided for scroll, using app center: ({x}, {y})")
        else:
            raise ValueError("Missing x or y coordinates for scrolling. For scrolling to work, mouse must be over the scrollable area.")
    
    pyautogui.moveTo(x, y, duration=0.3)
    
    if scroll_direction in ["up", "down"]:
        scroll_value = scroll_amount if scroll_direction == "up" else -scroll_amount
        pyautogui.scroll(scroll_value)
    elif scroll_direction in ["left", "right"]:
        scroll_value = scroll_amount if scroll_direction == "right" else -scroll_amount
        pyautogui.hscroll(scroll_value)

    return f"Scrolled {scroll_direction} by {scroll_amount} steps at coordinates ({x}, {y})"


def handle_move_mouse(x: int, y: int) -> str:
    """Move mouse to specified coordinates."""
    if x is None or y is None:
        raise ValueError("Missing x or y coordinates for mouse movement")
    pyautogui.moveTo(x, y, duration=0.5)
    return f"Moved mouse to ({x}, {y})"


def handle_key_press(key: str, modifier_keys: Optional[List[str]] = None) -> str:
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


def handle_hotkey(hotkey_str: str) -> str:
    """Handle hotkey combinations."""
    if not hotkey_str:
        raise ValueError("No hotkey string provided for hotkey action")

    keys = hotkey_str.split('+')
    
    if platform.system().lower() == "darwin":  # macOS
        keys = ["command" if k.lower() == "cmd" else k for k in keys]
    
    pyautogui.hotkey(*keys)
    print(f"clicked keys: {keys}")
    
    return f"Pressed hotkey combination: {hotkey_str}"


def handle_screen_size() -> str:
    """Get screen dimensions."""
    width, height = pyautogui.size()
    return f"Screen size: {width}x{height}"


def handle_scroll_to_bottom(x: Optional[int], y: Optional[int], app_name: Optional[str], min_confidence: float = 0.5) -> str:
    """Continuously scroll down while taking screenshots and extracting text until reaching bottom."""
    if x is None or y is None:
        if app_name:
            screen_width, screen_height = pyautogui.size()
            x = screen_width // 2
            y = screen_height // 2
        else:
            raise ValueError("Missing x or y coordinates for scrolling. For scrolling to work, mouse must be over the scrollable area.")
    
    pyautogui.moveTo(x, y, duration=0.3)
    
    scroll_count = 0
    max_attempts = 100
    
    screenshots_dir = "screenshots"
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)
        
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_text_data = []
    all_screenshots = []
    
    prev_screenshot = pyautogui.screenshot()
    initial_filename = f"scroll_page_{base_timestamp}_0.png"
    initial_filepath = os.path.join(screenshots_dir, initial_filename)
    prev_screenshot.save(initial_filepath)
    all_screenshots.append(initial_filepath)
    
    try:
        initial_text_data = extract_text_from_image(initial_filepath, min_confidence)
        if initial_text_data:
            all_text_data.extend(initial_text_data)
    except Exception as e:
        print(f"Error extracting text from initial screenshot: {str(e)}")
    
    # Use moderate scroll steps that move about 80% of a screen worth of content
    # Typical scroll step moves ~3 lines, so we calculate based on screen height
    screen_width, screen_height = pyautogui.size()
    # Estimate ~20 pixels per line of text, so screen_height/20 gives total lines
    # Then scroll 80% of that in steps of ~3 lines each
    estimated_lines_per_screen = screen_height // 20
    scroll_step = max(5, int(estimated_lines_per_screen * 0.8 // 3))  # Min 5 steps, max ~80% of screen
    similarity_threshold = 0.95
    consecutive_similar_screens = 0
    required_similar_screens = 2
    
    while scroll_count < max_attempts:
        pyautogui.scroll(-scroll_step)
        time.sleep(1.0)
        
        current_screenshot = pyautogui.screenshot()
        current_filename = f"scroll_page_{base_timestamp}_{scroll_count + 1}.png"
        current_filepath = os.path.join(screenshots_dir, current_filename)
        current_screenshot.save(current_filepath)
        all_screenshots.append(current_filepath)
        
        try:
            current_text_data = extract_text_from_image(current_filepath, min_confidence)
            if current_text_data:
                all_text_data.extend(current_text_data)
        except Exception as e:
            print(f"Error extracting text from screenshot {scroll_count + 1}: {str(e)}")
        
        prev_array = np.array(prev_screenshot)
        current_array = np.array(current_screenshot)
        
        if prev_array.shape == current_array.shape:
            similarity = np.mean(prev_array == current_array)
            
            if similarity > similarity_threshold:
                consecutive_similar_screens += 1
                if consecutive_similar_screens >= required_similar_screens:
                    print(f"Reached bottom after {scroll_count + 1} scrolls (similarity: {similarity:.2f})")
                    break
            else:
                consecutive_similar_screens = 0
        else:
            consecutive_similar_screens = 0
        
        prev_screenshot = current_screenshot
        scroll_count += 1
    
    # Group text into lines for better readability
    ordered_text_data = sorted(all_text_data, key=lambda x: (x['coordinates']['raw_y'], x['coordinates']['raw_x']))
    
    # Group by lines (y-coordinate proximity)
    lines = []
    current_line = []
    last_y = None
    line_threshold = 25
    
    for item in ordered_text_data:
        if last_y is None or abs(item['coordinates']['raw_y'] - last_y) <= line_threshold:
            current_line.append(item)
        else:
            if current_line:
                lines.append(current_line)
            current_line = [item]
        last_y = item['coordinates']['raw_y']
    
    if current_line:
        lines.append(current_line)
    
    # Reconstruct text with proper line breaks
    ordered_text = ""
    seen_lines = set()
    
    for line in lines:
        line.sort(key=lambda x: x['coordinates']['raw_x'])
        line_text = " ".join([item['text'].strip() for item in line if item['text'].strip()])
        
        line_key = line_text.lower().strip()
        if line_key and line_key not in seen_lines and len(line_key) > 2:
            seen_lines.add(line_key)
            ordered_text += line_text + "\n"
    
    compiled_text = ordered_text
    
    summary_filename = f"scroll_summary_{base_timestamp}.txt"
    summary_filepath = os.path.join(screenshots_dir, summary_filename)
    
    ordered_summary_filename = f"scroll_summary_ordered_{base_timestamp}.txt"
    ordered_summary_filepath = os.path.join(screenshots_dir, ordered_summary_filename)
    
    with open(summary_filepath, 'w') as f:
        f.write(compiled_text)
        
    with open(ordered_summary_filepath, 'w') as f:
        f.write(ordered_text)
    print(f"{ordered_text}")
    return (
        f"Scrolled to bottom after {scroll_count} scroll actions at coordinates ({x}, {y})\n"
        f"Captured {len(all_screenshots)} screenshots\n"
        f"Extracted {len(lines)} text lines\n\n"
        f"EXTRACTED TEXT:\n{ordered_text}\n\n"
        f"Files saved: {summary_filepath}, {ordered_summary_filepath}"
    )


def handle_analyze_screenshot(screenshot_path: Optional[str], region: Optional[List[int]], min_confidence: float = 0.5) -> str:
    """Extract text and coordinates from screenshot."""
    if screenshot_path:
        if not os.path.exists(screenshot_path):
            raise ValueError(f"Screenshot not found at {screenshot_path}")
        image_path = screenshot_path
    else:
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
        image_path = filepath
        
    try:
        text_data = extract_text_from_image(image_path, min_confidence)
        if not text_data:
            return f"No text detected in screenshot {image_path}"
        
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
            - scroll: Scroll in specified direction (requires x,y coordinates and app_name when scrolling in application)
            - scroll_to_bottom: Continuously scroll down while taking screenshots and extracting text until reaching bottom (requires app_name)
            - type: Type specified text (requires app_name)
            - key_press: Press specified key (requires app_name)
            - key_hold: Hold key combination (requires app_name)
            - hotkey: Press a hotkey combination (requires app_name)
            - screenshot: Capture screen (optionally in specified region or active window)(requires app_name when screenshotting an app)
            - analyze_screenshot: Extract text and coordinates from screenshot
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
    actions_requiring_focus = ["click", "type", "key_press", "key_hold", "hotkey", "drag", "scroll", "scroll_to_bottom", "screenshot"]
    if action in actions_requiring_focus and app_name:
        focus_success = focus_application(app_name)
        if not focus_success:
            return f"Warning: Could not focus on {app_name}. Proceeding with action anyway."
    print(f"performing action: {action} in app: {app_name}") 
    try:
        if action == "mouse_position":
            return handle_mouse_position()

        elif action == "screenshot":
            return handle_screenshot(region)

        elif action == "type":
            return handle_type(text)

        elif action == "click":
            return handle_click(x, y, click_type or "left")

        # drag does not currently work
        elif action == "drag":
            if drag_to_x is None or drag_to_y is None:
                raise ValueError("Missing drag destination coordinates")

            if x is not None and y is not None:
                pyautogui.moveTo(x, y, duration=0.3)

            pyautogui.dragTo(drag_to_x, drag_to_y, duration=1.0)
            return f"Dragged to ({drag_to_x}, {drag_to_y})"

        elif action == "scroll":
            return handle_scroll(x, y, app_name, scroll_direction or "up", scroll_amount or 15)

        elif action == "scroll_to_bottom":
            return handle_scroll_to_bottom(x, y, app_name, min_confidence or 0.5)

        elif action == "move_mouse":
            return handle_move_mouse(x, y)

        elif action == "key_press":
            return handle_key_press(key, modifier_keys)

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
            return handle_hotkey(hotkey_str)

        elif action == "screen_size":
            return handle_screen_size()

        elif action == "open_app":
            if not app_name:
                raise ValueError("No application name provided")
            return open_application(app_name)

        elif action == "close_app":
            if not app_name:
                raise ValueError("No application name provided")
            return close_application(app_name)
            
        elif action == "analyze_screenshot":
            return handle_analyze_screenshot(screenshot_path, region, min_confidence or 0.5)

        else:
            raise ValueError(f"Unknown action: {action}")

    except Exception as e:
        return f"Error: {str(e)}"