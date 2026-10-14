# flake8: noqa: E402

import sys
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch

sys.modules["pyautogui"] = mock.MagicMock()
sys.modules["mouseinfo"] = mock.MagicMock()
sys.modules["cv2"] = mock.MagicMock()

import platform
import unittest

import numpy as np
import pytest

from src.strands_tools.use_computer import (
    UseComputerMethods,
    close_application,
    extract_text_from_image,
    group_text_by_lines,
    handle_analyze_screenshot_pytesseract,
    handle_sending_results_to_llm,
    open_application,
    use_computer,
)


class TestUseComputerConsent:
    """Tests for use_computer consent handling"""

    def test_use_computer_with_bypass_consent(self, monkeypatch):
        """Test use_computer proceeds immediately if BYPASS_TOOL_CONSENT is set"""
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        with patch("src.strands_tools.use_computer.UseComputerMethods.mouse_position") as mock_mouse:
            mock_mouse.return_value = "Mouse position: (100, 200)"
            result = use_computer(action="mouse_position")
            assert result == {"status": "success", "content": [{"text": "Mouse position: (100, 200)"}]}

    def test_use_computer_without_bypass_consent_user_says_no(self, monkeypatch):
        """Test use_computer cancels with user input = 'n'"""
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "false")

        with patch("src.strands_tools.use_computer.get_user_input") as mock_input:
            mock_input.side_effect = ["n", "Just testing"]
            result = use_computer(action="mouse_position")
            assert isinstance(result, dict)
            assert result["status"] == "error"
            assert "Just testing" in result["content"][0]["text"]

    def test_use_computer_without_bypass_consent_user_says_yes(self, monkeypatch):
        """Test use_computer runs if user confirms 'y'"""
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "false")

        with patch("src.strands_tools.use_computer.get_user_input", return_value="y"):
            with patch("src.strands_tools.use_computer.UseComputerMethods.mouse_position") as mock_mouse:
                mock_mouse.return_value = "Mouse position: (50, 60)"
                result = use_computer(action="mouse_position")
                assert result == {"status": "success", "content": [{"text": "Mouse position: (50, 60)"}]}


class TestUseComputerMethods:
    """Tests for UseComputerMethods class - basic functionality"""

    @pytest.fixture
    def computer(self):
        return UseComputerMethods()

    def test_mouse_position(self, computer):
        with patch("pyautogui.position", return_value=(100, 200)):
            result = computer.mouse_position()
            assert result == "Mouse position: (100, 200)"

    @pytest.mark.parametrize(
        "click_type,expected",
        [
            ("left", "Left clicked at (100, 100)"),
            ("right", "Right clicked at (100, 100)"),
            ("double", "Double clicked at (100, 100)"),
            ("middle", "Middle clicked at (100, 100)"),
        ],
    )
    def test_click_types(self, computer, click_type, expected):
        with (
            patch("pyautogui.moveTo"),
            patch("pyautogui.click"),
            patch("pyautogui.rightClick"),
            patch("pyautogui.middleClick"),
            patch("time.sleep"),
            patch("platform.system", return_value="windows"),
        ):
            result = computer.click(100, 100, click_type=click_type)
            assert result == expected

    def test_move_mouse(self, computer):
        with patch("pyautogui.moveTo"):
            result = computer.move_mouse(100, 200)
            assert result == "Moved mouse to (100, 200)"

    def test_type_text(self, computer):
        with patch("pyautogui.typewrite"):
            result = computer.type("Hello World")
            assert result == "Typed: Hello World"

    def test_type_text_empty(self, computer):
        with pytest.raises(ValueError):
            computer.type("")

    def test_key_press(self, computer):
        with patch("pyautogui.press"):
            result = computer.key_press("enter")
            assert result == "Pressed key: enter"

    def test_key_press_with_modifiers(self, computer):
        with patch("pyautogui.hotkey"):
            result = computer.key_press("a", modifier_keys=["ctrl", "shift"])
            assert result == "Pressed key combination: ctrl+shift+a"

    def test_key_hold_simple(self, computer):
        with patch("pyautogui.keyDown"), patch("pyautogui.keyUp"), patch("time.sleep"):
            result = computer.key_hold("a")
            assert result == "Held and released key: a"

    def test_key_hold_with_modifiers(self, computer):
        with patch("pyautogui.keyDown"), patch("pyautogui.keyUp"), patch("pyautogui.press"):
            result = computer.key_hold("v", modifier_keys=["ctrl", "shift"])
            assert result == "Held ctrl+shift and pressed v"

    def test_key_hold_no_key(self, computer):
        with pytest.raises(ValueError) as exc_info:
            computer.key_hold(None)
        assert str(exc_info.value) == "No key specified for key hold"

    @pytest.mark.parametrize(
        "system,hotkey_str,expected_keys",
        [
            ("windows", "ctrl+c", ["ctrl", "c"]),
            ("darwin", "cmd+c", ["command", "c"]),
            ("linux", "alt+tab", ["alt", "tab"]),
        ],
    )
    def test_hotkey(self, computer, system, hotkey_str, expected_keys):
        with (
            patch("platform.system", return_value=system),
            patch("pyautogui.hotkey") as mock_hotkey,
            patch("builtins.print"),
        ):  # Mock print to avoid output
            result = computer.hotkey(hotkey_str)
            mock_hotkey.assert_called_once_with(*expected_keys)
            assert result == f"Pressed hotkey combination: {hotkey_str}"

    def test_hotkey_empty_string(self, computer):
        with pytest.raises(ValueError) as exc_info:
            computer.hotkey("")
        assert str(exc_info.value) == "No hotkey string provided for hotkey action"

    def test_screen_size(self, computer):
        with patch("pyautogui.size", return_value=(1920, 1080)):
            result = computer.screen_size()
            assert result == "Screen size: 1920x1080"

    def test_open_app_success(self, computer):
        with (
            patch("src.strands_tools.use_computer.open_application", return_value="Launched TestApp") as mock_open,
            patch("builtins.print"),
        ):  # Mock print statement
            result = computer.open_app("TestApp")
            assert result == "Launched TestApp"
            mock_open.assert_called_once_with("TestApp")

    def test_open_app_no_name(self, computer):
        with pytest.raises(ValueError) as exc_info:
            computer.open_app("")
        assert str(exc_info.value) == "No application name provided"

    def test_open_app_none_name(self, computer):
        with pytest.raises(ValueError) as exc_info:
            computer.open_app(None)
        assert str(exc_info.value) == "No application name provided"

    def test_close_app_success(self, computer):
        with patch(
            "src.strands_tools.use_computer.close_application", return_value="Closed 1 instance(s) of TestApp"
        ) as mock_close:
            result = computer.close_app("TestApp")
            assert result == "Closed 1 instance(s) of TestApp"
            mock_close.assert_called_once_with("TestApp")

    def test_close_app_no_name(self, computer):
        with pytest.raises(ValueError) as exc_info:
            computer.close_app("")
        assert str(exc_info.value) == "No application name provided"

    def test_close_app_none_name(self, computer):
        with pytest.raises(ValueError) as exc_info:
            computer.close_app(None)
        assert str(exc_info.value) == "No application name provided"


class TestScreenshotAndAnalysis:
    """Tests for screenshot and analysis functionality"""

    @pytest.fixture
    def computer(self):
        return UseComputerMethods()

    def test_analyze_screen(self, computer):
        mock_image_content = {"image": {"format": "png", "source": {"bytes": b"test image data"}}}

        with (
            patch("src.strands_tools.use_computer.handle_analyze_screenshot_pytesseract") as mock_analyze,
            patch("src.strands_tools.use_computer.handle_sending_results_to_llm") as mock_send_results,
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000),  # Mock file size to be small enough
        ):
            # Set up the return values
            mock_analyze.return_value = {
                "text_result": "Detected 1 text elements in screenshot",
                "image_path": "test.png",
                "should_delete": True,
            }
            mock_send_results.return_value = mock_image_content

            # Call the analyze_screen method without send_screenshot (default behavior now returns only text)
            result = computer.analyze_screen(screenshot_path="test.png")

            # Verify the result format
            assert result["status"] == "success"
            assert len(result["content"]) == 1
            assert result["content"][0]["text"] == "Detected 1 text elements in screenshot"

            # Verify the called functions (should not call mock_send_results when send_screenshot=False)
            mock_analyze.assert_called_once_with("test.png", None, 0.5)

            # Now test with send_screenshot=True to verify both results are returned
            mock_analyze.reset_mock()
            result_with_screenshot = computer.analyze_screen(screenshot_path="test.png", send_screenshot=True)

            # Verify the result format with screenshot included
            assert result_with_screenshot["status"] == "success"
            assert len(result_with_screenshot["content"]) == 2
            assert result_with_screenshot["content"][0]["text"] == "Detected 1 text elements in screenshot"
            assert result_with_screenshot["content"][1] == mock_image_content

            # Now send_results should be called
            mock_send_results.assert_called_once_with("test.png")

    def test_analyze_screen_with_region_and_confidence(self, computer):
        with (
            patch("src.strands_tools.use_computer.handle_analyze_screenshot_pytesseract") as mock_analyze,
            patch("src.strands_tools.use_computer.handle_sending_results_to_llm") as mock_send_results,
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000),  # Mock file size to be small enough
        ):
            # Set up the return values
            mock_analyze.return_value = {
                "text_result": "Analysis result text",
                "image_path": "test.png",
                "should_delete": True,
            }
            mock_send_results.return_value = {"image": {"format": "png", "source": {"bytes": b"image data"}}}

            # Call the analyze_screen method with custom parameters but without send_screenshot
            computer.analyze_screen(region=[0, 0, 100, 100], min_confidence=0.8)

            # Verify analyze is called with the correct parameters
            mock_analyze.assert_called_once_with(None, [0, 0, 100, 100], 0.8)

            # handle_sending_results_to_llm should not be called when send_screenshot=False
            mock_send_results.assert_not_called()

            # Now test with send_screenshot=True
            mock_analyze.reset_mock()
            mock_send_results.reset_mock()

            computer.analyze_screen(region=[0, 0, 100, 100], min_confidence=0.8, send_screenshot=True)

            # Now verify that send_results is called
            mock_analyze.assert_called_once_with(None, [0, 0, 100, 100], 0.8)
            mock_send_results.assert_called_once_with("test.png")

    def test_analyze_screen_pytesseract_with_no_text(self):
        with (
            patch("os.path.exists", return_value=True),
            patch("src.strands_tools.use_computer.extract_text_from_image", return_value=[]),
        ):
            result = handle_analyze_screenshot_pytesseract("test.png", None)
            assert "No text detected" in result["text_result"]

    @patch("os.path.exists", return_value=True)
    def test_analyze_screen_pytesseract_with_text(self, mock_exists):
        mock_text_data = [
            {
                "text": "test",
                "coordinates": {"x": 0, "y": 0, "width": 10, "height": 10, "center_x": 5, "center_y": 5},
                "confidence": 0.9,
            }
        ]
        with patch("src.strands_tools.use_computer.extract_text_from_image", return_value=mock_text_data):
            result = handle_analyze_screenshot_pytesseract("test.png", None)
            assert "Detected 1 text elements" in result["text_result"]

    def test_handle_sending_results_to_llm(self):
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=b"test image data")),
            patch("PIL.Image.open") as mock_image_open,
        ):
            # Mock the PIL Image object
            mock_img = MagicMock()
            mock_img.format = "PNG"
            mock_image_open.return_value.__enter__.return_value = mock_img

            # Call the function
            result = handle_sending_results_to_llm("test.png")

            # Verify the result
            assert "image" in result
            assert result["image"]["format"] == "png"
            assert "bytes" in result["image"]["source"]

    def test_handle_sending_results_to_llm_nonexistent_file(self):
        with patch("os.path.exists", return_value=False):
            result = handle_sending_results_to_llm("nonexistent.png")
            assert "text" in result
            assert "not found" in result["text"]


class TestApplicationManagement:
    """Tests for application management functionality"""

    @pytest.mark.parametrize("system", ["windows", "darwin", "linux"])
    def test_open_application(self, system):
        with patch("platform.system", return_value=system), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = open_application("test_app")
            assert "Launched" in result

    def test_close_application(self):
        mock_process = MagicMock()
        mock_process.info = {"name": "test_app"}
        mock_process.terminate = MagicMock()

        with patch("psutil.process_iter", return_value=[mock_process]):
            result = close_application("test_app")
            assert "Closed" in result


class TestMouseOperations:
    """Tests for mouse operations like drag and scroll"""

    @pytest.fixture
    def computer(self):
        return UseComputerMethods()

    def test_drag_operation(self, computer):
        with (
            patch("pyautogui.moveTo") as mock_move,
            patch("pyautogui.drag") as mock_drag,
            patch("pyautogui.position", return_value=(100, 100)),
            patch("time.sleep"),
        ):
            result = computer.drag(100, 100, drag_to_x=200, drag_to_y=200)

            mock_move.assert_called_once()
            mock_drag.assert_called_once()
            assert "Dragged from (100, 100) to (200, 200)" in result

    @pytest.mark.parametrize(
        "direction,amount",
        [
            ("up", 15),
            ("down", 15),
            ("left", 15),
            ("right", 15),
        ],
    )
    def test_scroll_operations(self, computer, direction, amount):
        with (
            patch("pyautogui.moveTo") as mock_move,
            patch("pyautogui.scroll") as mock_scroll,
            patch("pyautogui.hscroll") as mock_hscroll,
            patch("pyautogui.click"),
            patch("platform.system", return_value="windows"),
            patch("time.sleep"),
            patch("subprocess.run"),
        ):
            result = computer.scroll(100, 100, None, direction, amount)

            mock_move.assert_called_once_with(100, 100, duration=0.3)

            if direction in ["up", "down"]:
                scroll_value = amount if direction == "up" else -amount
                mock_scroll.assert_called_once_with(scroll_value)
            else:
                scroll_value = amount if direction == "right" else -amount
                mock_hscroll.assert_called_once_with(scroll_value)

            assert f"Scrolled {direction}" in result

    def test_scroll_operations_missing_coordinates_with_app_name(self, computer):
        """Test scroll when coordinates are missing but app_name is provided - should use screen center"""
        with (
            patch("pyautogui.moveTo") as mock_move,
            patch("pyautogui.scroll") as mock_scroll,
            patch("pyautogui.click"),
            patch("pyautogui.size", return_value=(1920, 1080)) as mock_size,
            patch("src.strands_tools.use_computer.logger.info") as mock_logger,
        ):
            result = computer.scroll(None, None, "TestApp", "up", 15)

            mock_size.assert_called_once()
            mock_move.assert_called_once_with(960, 540, duration=0.3)
            mock_logger.assert_called_once_with("No coordinates provided for scroll, using app center: (960, 540)")
            mock_scroll.assert_called_once_with(15)

            assert "Scrolled up by 15 steps at coordinates (960, 540)" in result

    def test_scroll_operations_missing_coordinates_without_app_name(self, computer):
        """Test scroll when coordinates are missing and no app_name - should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            computer.scroll(None, None, None, "up", 15)

        expected_message = (
            "Missing x or y coordinates for scrolling. For scrolling to work, mouse must be over the scrollable area."
        )
        assert str(exc_info.value) == expected_message

    def test_scroll_operations_missing_x_coordinate_only(self, computer):
        """Test scroll when only x coordinate is missing"""
        with patch("pyautogui.size", return_value=(1920, 1080)):
            with pytest.raises(ValueError) as exc_info:
                computer.scroll(None, 100, None, "up", 15)

            expected_message = (
                "Missing x or y coordinates for scrolling. "
                "For scrolling to work, mouse must be over the scrollable area."
            )
            assert str(exc_info.value) == expected_message

    def test_scroll_operations_missing_y_coordinate_only(self, computer):
        """Test scroll when only y coordinate is missing"""
        with patch("pyautogui.size", return_value=(1920, 1080)):
            with pytest.raises(ValueError) as exc_info:
                computer.scroll(100, None, None, "up", 15)

            expected_message = (
                "Missing x or y coordinates for scrolling. "
                "For scrolling to work, mouse must be over the scrollable area."
            )
            assert str(exc_info.value) == expected_message


class TestTextProcessing:
    @pytest.fixture
    def sample_text_data(self):
        return [
            {
                "text": "Hello",
                "coordinates": {"x": 10, "y": 100},
            },
            {
                "text": "World",
                "coordinates": {"x": 60, "y": 102},  # Same line as "Hello"
            },
            {
                "text": "New",
                "coordinates": {"x": 10, "y": 150},  # New line
            },
            {
                "text": "Line",
                "coordinates": {"x": 60, "y": 152},  # Same line as "New"
            },
        ]

    def test_group_text_by_lines_empty_input(self):
        result = group_text_by_lines([])
        assert result == []

    def test_group_text_by_lines_single_line(self, sample_text_data):
        # Only use first two items which should be on the same line
        result = group_text_by_lines(sample_text_data[:2])
        assert len(result) == 1  # One line
        assert len(result[0]) == 2  # Two words in the line
        assert result[0][0]["text"] == "Hello"
        assert result[0][1]["text"] == "World"

    def test_group_text_by_lines_multiple_lines(self, sample_text_data):
        result = group_text_by_lines(sample_text_data)
        assert len(result) == 2  # Two lines
        assert len(result[0]) == 2  # Two words in first line
        assert len(result[1]) == 2  # Two words in second line
        assert result[0][0]["text"] == "Hello"
        assert result[1][0]["text"] == "New"

    def test_group_text_by_lines_threshold(self):
        data = [
            {"text": "Word1", "coordinates": {"x": 0, "y": 100}},
            {"text": "Word2", "coordinates": {"x": 0, "y": 115}},  # > 10px difference
        ]
        # With default threshold (10), should be two lines
        result = group_text_by_lines(data)
        assert len(result) == 2

        # With larger threshold (20), should be one line
        result = group_text_by_lines(data, line_threshold=20)
        assert len(result) == 1


class TestExtractTextFromImage:
    """Tests for OCR text extraction functionality"""

    @pytest.fixture
    def mock_image(self):
        return np.zeros((1200, 1200), dtype=np.uint8)

    @pytest.fixture
    def mock_tesseract_data(self):
        return {
            "text": ["Hello", "World", "", "Test"],
            "conf": [95.0, 95.0, 0.0, 95.0],
            "left": [10, 60, 0, 10],
            "top": [100, 100, 0, 150],
            "width": [40, 40, 0, 40],
            "height": [20, 20, 0, 20],
        }

    def test_extract_text_from_image_basic(self, mock_image, mock_tesseract_data):
        with (
            patch("cv2.imread", return_value=mock_image),
            patch("cv2.cvtColor", return_value=mock_image),
            patch("cv2.medianBlur", return_value=mock_image),
            patch("cv2.createCLAHE") as mock_clahe,
            patch("cv2.filter2D", return_value=mock_image),
            patch("pytesseract.image_to_data", return_value=mock_tesseract_data),
            patch("pyautogui.size", return_value=(1920, 1080)),
        ):
            mock_clahe.return_value.apply.return_value = mock_image

            results = extract_text_from_image("test.png")

            assert len(results) == 3
            assert results[0]["text"] == "Hello"
            assert results[1]["text"] == "World"
            assert results[2]["text"] == "Test"

    def test_extract_text_from_image_no_image(self):
        with patch("cv2.imread", return_value=None):
            with pytest.raises(ValueError) as exc_info:
                extract_text_from_image("nonexistent.png")
            assert "Could not read image" in str(exc_info.value)

    def test_extract_text_from_image_scaling(self, mock_image, mock_tesseract_data):
        small_image = np.zeros((800, 800, 3), dtype=np.uint8)

        with (
            patch("cv2.imread", return_value=small_image),
            patch("cv2.resize", return_value=mock_image),
            patch("cv2.cvtColor", return_value=mock_image),
            patch("cv2.medianBlur", return_value=mock_image),
            patch("cv2.createCLAHE") as mock_clahe,
            patch("cv2.filter2D", return_value=mock_image),
            patch("pytesseract.image_to_data", return_value=mock_tesseract_data),
            patch("pyautogui.size", return_value=(1920, 1080)),
        ):
            mock_clahe.return_value.apply.return_value = mock_image

            results = extract_text_from_image("test.png")

            assert results[0]["coordinates"]["scaling_applied"]
            assert "center_x" in results[0]["coordinates"]
            assert "center_y" in results[0]["coordinates"]

    def test_extract_text_from_image_ocr_failure(self, mock_image):
        color_image = np.zeros((1200, 1200, 3), dtype=np.uint8)
        gray_image = np.zeros((1200, 1200), dtype=np.uint8)

        with (
            patch("cv2.imread", return_value=color_image),
            patch("cv2.cvtColor", return_value=gray_image),
            patch("cv2.medianBlur", return_value=gray_image),
            patch("cv2.createCLAHE") as mock_clahe,
            patch("cv2.filter2D", return_value=gray_image),
            patch("pytesseract.image_to_data", side_effect=Exception("OCR failed")),
            patch("pyautogui.size", return_value=(1920, 1080)),
        ):
            mock_clahe.return_value.apply.return_value = gray_image

            with pytest.raises(ValueError) as exc_info:
                extract_text_from_image("test.png")
            assert "OCR failed with all configurations" in str(exc_info.value)

    def test_extract_text_from_image_confidence_threshold(self, mock_image):
        color_image = np.zeros((1200, 1200, 3), dtype=np.uint8)
        gray_image = np.zeros((1200, 1200), dtype=np.uint8)

        test_data = {
            "text": ["High1", "High2", "Low", ""],
            "conf": [95.0, 92.0, 85.0, 0.0],
            "left": [10, 60, 110, 0],
            "top": [100, 100, 100, 0],
            "width": [40, 40, 40, 0],
            "height": [20, 20, 20, 0],
        }

        with (
            patch("cv2.imread", return_value=color_image),
            patch("cv2.cvtColor", return_value=gray_image),
            patch("cv2.medianBlur", return_value=gray_image),
            patch("cv2.createCLAHE") as mock_clahe,
            patch("cv2.filter2D", return_value=gray_image),
            patch("pytesseract.image_to_data", return_value=test_data),
            patch("pyautogui.size", return_value=(1920, 1080)),
        ):
            mock_clahe.return_value.apply.return_value = gray_image

            results = extract_text_from_image("test.png", min_confidence=0.9)

            assert len(results) == 2
            assert all(r["confidence"] >= 0.9 for r in results)
            assert results[0]["text"] == "High1"
            assert results[1]["text"] == "High2"


class TestFocusApplication:
    @pytest.mark.parametrize(
        "system,expected_command",
        [
            ("darwin", ["osascript", "-e", 'tell application "TestApp" to activate']),
            (
                "windows",
                [
                    "powershell",
                    "-Command",
                    (
                        "Add-Type -AssemblyName Microsoft.VisualBasic; "
                        "[Microsoft.VisualBasic.Interaction]::AppActivate('TestApp')"
                    ),
                ],
            ),
            ("linux", ["wmctrl", "-a", "TestApp"]),
        ],
    )
    def test_focus_application_success(self, system, expected_command):
        from src.strands_tools.use_computer import focus_application

        with patch("platform.system", return_value=system), patch("subprocess.run") as mock_run, patch("time.sleep"):
            # Set up the mock to return an object with returncode=0
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result  # Successful run with returncode 0

            result = focus_application("TestApp")

            assert result is True
            mock_run.assert_called_once_with(expected_command, check=True, capture_output=True, timeout=2.0)

    @pytest.mark.parametrize("system", ["darwin", "windows", "linux"])
    def test_focus_application_failure(self, system):
        from src.strands_tools.use_computer import focus_application

        with (
            patch("platform.system", return_value=system),
            patch("subprocess.run", side_effect=Exception("Command failed")),
            patch("time.sleep"),
        ):
            result = focus_application("TestApp")
            assert result is False

    def test_focus_application_unknown_system(self):
        from src.strands_tools.use_computer import focus_application

        with patch("platform.system", return_value="unknown"):
            result = focus_application("TestApp")
            assert result is False


class TestHandleAnalyzeScreenshotPytesseract:
    """Tests for screenshot analysis handling"""

    def test_handle_analyze_screenshot_pytesseract_with_existing_path(self):
        mock_text_data = [
            {
                "text": "Sample Text",
                "coordinates": {"x": 10, "y": 20, "width": 100, "height": 30, "center_x": 60, "center_y": 35},
                "confidence": 0.95,
            }
        ]

        with (
            patch("os.path.exists", return_value=True),
            patch("src.strands_tools.use_computer.extract_text_from_image", return_value=mock_text_data),
        ):
            result = handle_analyze_screenshot_pytesseract("test.png", None)

            assert "Detected 1 text elements" in result["text_result"]
            assert "Sample Text" in result["text_result"]
            assert "Confidence: 0.95" in result["text_result"]
            assert result["image_path"] == "test.png"
            assert result["should_delete"] is False

    def test_handle_analyze_screenshot_pytesseract_nonexistent_path(self):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(ValueError) as exc_info:
                handle_analyze_screenshot_pytesseract("nonexistent.png", None)
            assert "Screenshot not found at nonexistent.png" in str(exc_info.value)

    def test_handle_analyze_screenshot_pytesseract_no_path_provided(self):
        mock_text_data = [
            {
                "text": "Auto Screenshot",
                "coordinates": {"x": 5, "y": 10, "width": 50, "height": 20, "center_x": 30, "center_y": 20},
                "confidence": 0.8,
            }
        ]

        mock_screenshot = MagicMock()

        with (
            patch("os.path.exists", return_value=False),
            patch("pyautogui.screenshot", return_value=mock_screenshot),
            patch("src.strands_tools.use_computer.extract_text_from_image", return_value=mock_text_data),
            patch("datetime.datetime") as mock_datetime,
            patch("src.strands_tools.use_computer.create_screenshot") as mock_create_screenshot,
        ):
            # Mock datetime.now().strftime()
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            mock_create_screenshot.return_value = "screenshots/screenshot_20240101_120000.png"

            result = handle_analyze_screenshot_pytesseract(None, None)

            mock_create_screenshot.assert_called_once_with(None)
            assert "Detected 1 text elements" in result["text_result"]
            assert "Auto Screenshot" in result["text_result"]
            assert result["should_delete"] is True

    def test_handle_analyze_screenshot_pytesseract_with_region(self):
        mock_text_data = []

        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs"),
            patch("src.strands_tools.use_computer.create_screenshot") as mock_create_screenshot,
            patch("src.strands_tools.use_computer.extract_text_from_image", return_value=mock_text_data),
            patch("datetime.datetime") as mock_datetime,
        ):
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            mock_create_screenshot.return_value = "screenshots/screenshot_20240101_120000.png"

            result = handle_analyze_screenshot_pytesseract(None, [0, 0, 100, 100])

            # Verify screenshot was called with region
            mock_create_screenshot.assert_called_once_with([0, 0, 100, 100])
            assert "No text detected" in result["text_result"]

    def test_handle_analyze_screenshot_pytesseract_no_text_found(self):
        with (
            patch("os.path.exists", return_value=True),
            patch("src.strands_tools.use_computer.extract_text_from_image", return_value=[]),
        ):
            result = handle_analyze_screenshot_pytesseract("test.png", None)
            assert "No text detected in screenshot test.png" in result["text_result"]

    def test_handle_analyze_screenshot_pytesseract_analysis_error(self):
        with (
            patch("os.path.exists", return_value=True),
            patch("src.strands_tools.use_computer.extract_text_from_image", side_effect=Exception("OCR Error")),
        ):
            with pytest.raises(RuntimeError) as excinfo:
                handle_analyze_screenshot_pytesseract("test.png", None)
            assert "Error analyzing screenshot: OCR Error" in str(excinfo.value)

    def test_handle_analyze_screenshot_pytesseract_formatted_output(self):
        mock_text_data = [
            {
                "text": "First",
                "coordinates": {"x": 10, "y": 20, "width": 40, "height": 15, "center_x": 30, "center_y": 27},
                "confidence": 0.9,
            },
            {
                "text": "Second",
                "coordinates": {"x": 60, "y": 20, "width": 50, "height": 15, "center_x": 85, "center_y": 27},
                "confidence": 0.85,
            },
        ]

        with (
            patch("os.path.exists", return_value=True),
            patch("src.strands_tools.use_computer.extract_text_from_image", return_value=mock_text_data),
        ):
            result = handle_analyze_screenshot_pytesseract("test.png", None)

            # Check that both text elements are included with proper formatting
            assert "Detected 2 text elements" in result["text_result"]
            assert "1. Text: 'First'" in result["text_result"]
            assert "2. Text: 'Second'" in result["text_result"]
            assert "Confidence: 0.90" in result["text_result"]
            assert "Confidence: 0.85" in result["text_result"]
            assert "Position: X=10, Y=20, W=40, H=15" in result["text_result"]
            assert "Center: (30, 27)" in result["text_result"]


class TestUseComputerEdgeCases:
    """Tests for edge cases and error handling in use_computer"""

    def test_use_computer_unknown_action(self, monkeypatch):
        """Test use_computer with unknown action raises ValueError"""
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        result = use_computer(action="invalid_unknown_action")
        assert result == {"status": "error", "content": [{"text": "Unknown action: invalid_unknown_action"}]}

    def test_use_computer_method_exception(self, monkeypatch):
        """Test use_computer catches exceptions from methods and returns error message"""
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        with patch("src.strands_tools.use_computer.UseComputerMethods.mouse_position") as mock_mouse:
            mock_mouse.side_effect = RuntimeError("Test error message")
            result = use_computer(action="mouse_position")
            assert result == {"status": "error", "content": [{"text": "Error: Test error message"}]}

    def test_use_computer_method_different_exception_type(self, monkeypatch):
        """Test use_computer catches different exception types"""
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        with patch("src.strands_tools.use_computer.UseComputerMethods.click") as mock_click:
            mock_click.side_effect = ConnectionError("Network issue")
            result = use_computer(action="click", x=100, y=200)
            assert result == {"status": "error", "content": [{"text": "Error: Network issue"}]}

    def test_use_computer_getattr_returns_none(self, monkeypatch):
        """Test use_computer when getattr returns None (method doesn't exist)"""
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        result = use_computer(action="completely_nonexistent_method")
        assert result == {"status": "error", "content": [{"text": "Unknown action: completely_nonexistent_method"}]}

    def test_use_computer_focus_application_success(self, monkeypatch):
        """Test use_computer calls focus_application when action requires focus and app_name provided - success case"""
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        with (
            patch("src.strands_tools.use_computer.focus_application", return_value=True) as mock_focus,
            patch("src.strands_tools.use_computer.UseComputerMethods.click") as mock_click,
            patch("src.strands_tools.use_computer.logger.info") as mock_logger,
        ):
            mock_click.return_value = "Left clicked at (100, 200)"

            result = use_computer(action="click", x=100, y=200, app_name="TestApp")

            mock_focus.assert_called_once_with("TestApp", timeout=2.0)
            mock_click.assert_called_once()
            mock_logger.assert_called_with("Performing action: click in app: TestApp")

            assert result == {"status": "success", "content": [{"text": "Left clicked at (100, 200)"}]}

    def test_use_computer_focus_not_required_no_app_name(self, monkeypatch):
        """Test use_computer skips focus when action requires focus but no app_name provided"""
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        with (
            patch("src.strands_tools.use_computer.focus_application") as mock_focus,
            patch("src.strands_tools.use_computer.UseComputerMethods.scroll") as mock_scroll,
            patch("src.strands_tools.use_computer.logger.info") as mock_logger,
        ):
            mock_scroll.return_value = "Scrolled up by 15 steps at coordinates (100, 100)"

            result = use_computer(action="scroll", x=100, y=100, scroll_direction="up", scroll_amount=15)

            mock_focus.assert_not_called()
            mock_scroll.assert_called_once()
            mock_logger.assert_called_with("Performing action: scroll in app: None")

            assert result == {
                "status": "success",
                "content": [{"text": "Scrolled up by 15 steps at coordinates (100, 100)"}],
            }

    def test_use_computer_action_not_requiring_focus(self, monkeypatch):
        """Test use_computer skips focus for actions that don't require focus"""
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        with (
            patch("src.strands_tools.use_computer.focus_application") as mock_focus,
            patch("src.strands_tools.use_computer.UseComputerMethods.mouse_position") as mock_mouse,
            patch("src.strands_tools.use_computer.logger.info") as mock_logger,
        ):
            mock_mouse.return_value = "Mouse position: (100, 200)"

            result = use_computer(action="mouse_position", app_name="TestApp")

            mock_focus.assert_not_called()
            mock_mouse.assert_called_once()
            mock_logger.assert_called_with("Performing action: mouse_position in app: TestApp")

            assert result == {"status": "success", "content": [{"text": "Mouse position: (100, 200)"}]}

    def test_use_computer_analyze_screen_requires_focus(self, monkeypatch):
        """Test use_computer calls focus for analyze_screen action when app_name provided"""
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        with (
            patch("src.strands_tools.use_computer.focus_application", return_value=True) as mock_focus,
            patch("src.strands_tools.use_computer.UseComputerMethods.analyze_screen") as mock_analyze_screen,
        ):
            mock_analyze_screen.return_value = {
                "status": "success",
                "content": [
                    {"text": "Analysis results"},
                    {"image": {"format": "png", "source": {"bytes": b"image data"}}},
                ],
            }

            result = use_computer(action="analyze_screen", app_name="TestApp")

            mock_focus.assert_called_once_with("TestApp", timeout=2.0)
            mock_analyze_screen.assert_called_once()

            assert result["status"] == "success"
            assert len(result["content"]) == 2
            assert result["content"][0]["text"] == "Analysis results"


# Only run this test if on mac, because the tool has some mac specific way of performing actions
@pytest.mark.skipif(platform.system() != "Darwin", reason="Tests only valid on macOS")
class TestNativeMacDoubleClick:
    @pytest.fixture
    def computer(self):
        return UseComputerMethods()

    def test_native_mac_double_click_basic_functionality(self, computer):
        """Test that _native_mac_double_click calls all required Quartz functions"""
        # Mock all the Quartz CoreGraphics functions
        mock_create_mouse_event = MagicMock()
        mock_set_integer_value_field = MagicMock()
        mock_event_post = MagicMock()
        mock_click_down_event = MagicMock()
        mock_click_up_event = MagicMock()

        # Configure the mouse event creation to return our mock events
        mock_create_mouse_event.side_effect = [
            mock_click_down_event,  # First down
            mock_click_up_event,  # First up
            mock_click_down_event,  # Second down
            mock_click_up_event,  # Second up
        ]

        with (
            patch("src.strands_tools.use_computer.CGEventCreateMouseEvent", mock_create_mouse_event),
            patch("src.strands_tools.use_computer.CGEventSetIntegerValueField", mock_set_integer_value_field),
            patch("src.strands_tools.use_computer.CGEventPost", mock_event_post),
            patch("time.sleep") as mock_sleep,
        ):
            computer._native_mac_double_click(100, 200)

            # Verify CGEventCreateMouseEvent was called 4 times (2 down, 2 up)
            assert mock_create_mouse_event.call_count == 4

            # Verify CGEventPost was called 4 times (post each event)
            assert mock_event_post.call_count == 4

            # Verify CGEventSetIntegerValueField was called 4 times (set state for each event)
            assert mock_set_integer_value_field.call_count == 4

            # Verify sleep was called once (between first and second click)
            mock_sleep.assert_called_once_with(0.05)

    def test_native_mac_double_click_coordinates(self, computer):
        """Test that coordinates are passed correctly to mouse events"""
        mock_create_mouse_event = MagicMock()
        mock_set_integer_value_field = MagicMock()
        mock_event_post = MagicMock()

        with (
            patch("src.strands_tools.use_computer.CGEventCreateMouseEvent", mock_create_mouse_event),
            patch("src.strands_tools.use_computer.CGEventPost", mock_event_post),
            patch("time.sleep"),
        ):
            with patch.dict(
                "sys.modules",
                {"Quartz.CoreGraphics": MagicMock(CGEventSetIntegerValueField=mock_set_integer_value_field)},
            ):
                computer._native_mac_double_click(150, 250)

                # Verify all mouse events were created with correct coordinates
                expected_calls = [
                    # First click down and up
                    unittest.mock.call(None, mock.ANY, (150, 250), mock.ANY),
                    unittest.mock.call(None, mock.ANY, (150, 250), mock.ANY),
                    # Second click down and up
                    unittest.mock.call(None, mock.ANY, (150, 250), mock.ANY),
                    unittest.mock.call(None, mock.ANY, (150, 250), mock.ANY),
                ]

                mock_create_mouse_event.assert_has_calls(expected_calls)

    def test_native_mac_double_click_event_types(self, computer):
        """Test that correct event types are used for mouse down and up"""
        mock_create_mouse_event = MagicMock()
        mock_set_integer_value_field = MagicMock()
        mock_event_post = MagicMock()

        # Import the constants to verify they're used correctly
        from src.strands_tools.use_computer import kCGEventLeftMouseDown, kCGEventLeftMouseUp, kCGMouseButtonLeft

        with (
            patch("src.strands_tools.use_computer.CGEventCreateMouseEvent", mock_create_mouse_event),
            patch("src.strands_tools.use_computer.CGEventPost", mock_event_post),
            patch("time.sleep"),
        ):
            with patch.dict(
                "sys.modules",
                {"Quartz.CoreGraphics": MagicMock(CGEventSetIntegerValueField=mock_set_integer_value_field)},
            ):
                computer._native_mac_double_click(100, 200)

                # Verify the event types and button types are correct
                expected_calls = [
                    # First click: down, up
                    unittest.mock.call(None, kCGEventLeftMouseDown, (100, 200), kCGMouseButtonLeft),
                    unittest.mock.call(None, kCGEventLeftMouseUp, (100, 200), kCGMouseButtonLeft),
                    # Second click: down, up
                    unittest.mock.call(None, kCGEventLeftMouseDown, (100, 200), kCGMouseButtonLeft),
                    unittest.mock.call(None, kCGEventLeftMouseUp, (100, 200), kCGMouseButtonLeft),
                ]

                mock_create_mouse_event.assert_has_calls(expected_calls)

    def test_native_mac_double_click_click_states(self, computer):
        """Test that click states are set correctly (1 for first click, 2 for second)"""
        mock_create_mouse_event = MagicMock()
        mock_set_integer_value_field = MagicMock()
        mock_event_post = MagicMock()
        mock_down_event = MagicMock()
        mock_up_event = MagicMock()

        # Return the same mock events so we can track what state they're set to
        mock_create_mouse_event.return_value = mock_down_event
        mock_create_mouse_event.side_effect = [
            mock_down_event,
            mock_up_event,  # First click pair
            mock_down_event,
            mock_up_event,  # Second click pair
        ]

        from src.strands_tools.use_computer import kCGMouseEventClickState

        with (
            patch("src.strands_tools.use_computer.CGEventCreateMouseEvent", mock_create_mouse_event),
            patch("src.strands_tools.use_computer.CGEventSetIntegerValueField", mock_set_integer_value_field),
            patch("src.strands_tools.use_computer.CGEventPost", mock_event_post),
            patch("time.sleep"),
        ):
            computer._native_mac_double_click(100, 200)

            # Verify click states are set correctly
            expected_calls = [
                # First click: state = 1
                unittest.mock.call(mock_down_event, kCGMouseEventClickState, 1),
                unittest.mock.call(mock_up_event, kCGMouseEventClickState, 1),
                # Second click: state = 2
                unittest.mock.call(mock_down_event, kCGMouseEventClickState, 2),
                unittest.mock.call(mock_up_event, kCGMouseEventClickState, 2),
            ]

            mock_set_integer_value_field.assert_has_calls(expected_calls)

    def test_native_mac_double_click_event_posting(self, computer):
        """Test that events are posted in the correct order"""
        mock_create_mouse_event = MagicMock()
        mock_set_integer_value_field = MagicMock()
        mock_event_post = MagicMock()

        # Create distinct mock objects for each event
        mock_first_down = MagicMock()
        mock_first_up = MagicMock()
        mock_second_down = MagicMock()
        mock_second_up = MagicMock()

        mock_create_mouse_event.side_effect = [mock_first_down, mock_first_up, mock_second_down, mock_second_up]

        from src.strands_tools.use_computer import kCGHIDEventTap

        with (
            patch("src.strands_tools.use_computer.CGEventCreateMouseEvent", mock_create_mouse_event),
            patch("src.strands_tools.use_computer.CGEventPost", mock_event_post),
            patch("time.sleep"),
        ):
            with patch.dict(
                "sys.modules",
                {"Quartz.CoreGraphics": MagicMock(CGEventSetIntegerValueField=mock_set_integer_value_field)},
            ):
                computer._native_mac_double_click(100, 200)

                # Verify events are posted in correct order with correct tap
                expected_calls = [
                    unittest.mock.call(kCGHIDEventTap, mock_first_down),
                    unittest.mock.call(kCGHIDEventTap, mock_first_up),
                    unittest.mock.call(kCGHIDEventTap, mock_second_down),
                    unittest.mock.call(kCGHIDEventTap, mock_second_up),
                ]

                mock_event_post.assert_has_calls(expected_calls)

    def test_native_mac_double_click_sleep_timing(self, computer):
        """Test that sleep is only called once and with correct duration"""
        mock_create_mouse_event = MagicMock()
        mock_set_integer_value_field = MagicMock()
        mock_event_post = MagicMock()

        with (
            patch("src.strands_tools.use_computer.CGEventCreateMouseEvent", mock_create_mouse_event),
            patch("src.strands_tools.use_computer.CGEventPost", mock_event_post),
            patch("time.sleep") as mock_sleep,
        ):
            with patch.dict(
                "sys.modules",
                {"Quartz.CoreGraphics": MagicMock(CGEventSetIntegerValueField=mock_set_integer_value_field)},
            ):
                computer._native_mac_double_click(100, 200)

                # Verify sleep is called exactly once with 0.05 seconds
                mock_sleep.assert_called_once_with(0.05)

    def test_native_mac_double_click_different_coordinates(self, computer):
        """Test with different coordinate values"""
        mock_create_mouse_event = MagicMock()
        mock_set_integer_value_field = MagicMock()
        mock_event_post = MagicMock()

        test_coordinates = [(0, 0), (500, 300), (1920, 1080)]

        with (
            patch("src.strands_tools.use_computer.CGEventCreateMouseEvent", mock_create_mouse_event),
            patch("src.strands_tools.use_computer.CGEventPost", mock_event_post),
            patch("time.sleep"),
        ):
            with patch.dict(
                "sys.modules",
                {"Quartz.CoreGraphics": MagicMock(CGEventSetIntegerValueField=mock_set_integer_value_field)},
            ):
                for x, y in test_coordinates:
                    mock_create_mouse_event.reset_mock()
                    computer._native_mac_double_click(x, y)

                    # Verify coordinates are used in all 4 calls
                    for call in mock_create_mouse_event.call_args_list:
                        args, kwargs = call
                        assert args[2] == (x, y)  # Third argument should be coordinates tuple

    def test_native_mac_double_click_integration_with_click_method(self, computer):
        """Test that _native_mac_double_click is called from click method on macOS"""
        with (
            patch("platform.system", return_value="darwin"),
            patch("pyautogui.moveTo"),
            patch("time.sleep"),
            patch.object(computer, "_native_mac_double_click") as mock_native_click,
        ):
            computer.click(100, 200, click_type="double")

            # Verify _native_mac_double_click was called with correct coordinates
            mock_native_click.assert_called_once_with(100, 200)

    def test_native_mac_double_click_function_call_sequence(self, computer):
        """Test the complete sequence of function calls in correct order"""
        mock_create_mouse_event = MagicMock()
        mock_set_integer_value_field = MagicMock()
        mock_event_post = MagicMock()
        mock_sleep = MagicMock()

        # Track call order
        call_order = []

        def track_create_mouse_event(*args, **kwargs):
            call_order.append("create_mouse_event")
            return MagicMock()

        def track_set_integer_value_field(*args, **kwargs):
            call_order.append("set_integer_value_field")

        def track_event_post(*args, **kwargs):
            call_order.append("event_post")

        def track_sleep(*args, **kwargs):
            call_order.append("sleep")

        mock_create_mouse_event.side_effect = track_create_mouse_event
        mock_set_integer_value_field.side_effect = track_set_integer_value_field
        mock_event_post.side_effect = track_event_post
        mock_sleep.side_effect = track_sleep

        with (
            patch("src.strands_tools.use_computer.CGEventCreateMouseEvent", mock_create_mouse_event),
            patch("src.strands_tools.use_computer.CGEventSetIntegerValueField", mock_set_integer_value_field),
            patch("src.strands_tools.use_computer.CGEventPost", mock_event_post),
            patch("time.sleep", mock_sleep),
        ):
            computer._native_mac_double_click(100, 200)

            # Expected sequence: create_down, create_up, set_field_down, set_field_up,
            # post_down, post_up, sleep, create_down, create_up, set_field_down,
            # set_field_up, post_down, post_up
            expected_sequence = [
                "create_mouse_event",  # First down
                "create_mouse_event",  # First up
                "set_integer_value_field",  # Set first down state
                "set_integer_value_field",  # Set first up state
                "event_post",  # Post first down
                "event_post",  # Post first up
                "sleep",  # Delay between clicks
                "create_mouse_event",  # Second down
                "create_mouse_event",  # Second up
                "set_integer_value_field",  # Set second down state
                "set_integer_value_field",  # Set second up state
                "event_post",  # Post second down
                "event_post",  # Post second up
            ]

            assert call_order == expected_sequence
