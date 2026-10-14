"""
Regression tests for calculator precision handling.

Precision must affect ONLY formatting, not numeric value.
"""

from strands_tools.calculator import calculator


def extract_text(result):
    assert result["status"] == "success"
    return result["content"][0]["text"]


# ------------------------------------------------------------
# Core regression tests (original bug)
# ------------------------------------------------------------


def test_precision_does_not_change_real_value():
    r2 = calculator("221 * 318.11", precision=2)
    r4 = calculator("221 * 318.11", precision=4)

    t2 = extract_text(r2)
    t4 = extract_text(r4)

    assert "70302.31" in t2
    assert "70302.31" in t4  # trailing zeros are stripped by design


def test_precision_does_not_change_division_value():
    r2 = calculator("10 / 3", precision=2)
    r6 = calculator("10 / 3", precision=6)

    t2 = extract_text(r2)
    t6 = extract_text(r6)

    assert "3.33" in t2
    assert "3.333333" in t6


# ------------------------------------------------------------
# Complex number regression tests
# ------------------------------------------------------------


def test_precision_does_not_change_complex_value():
    r2 = calculator("(1 + I) * 318.11", precision=2)
    r4 = calculator("(1 + I) * 318.11", precision=4)

    t2 = extract_text(r2)
    t4 = extract_text(r4)

    assert "318.11" in t2
    assert "318.11" in t4


# ------------------------------------------------------------
# Symbolic guardrails (correct semantics)
# ------------------------------------------------------------


def test_expression_with_symbols_remains_symbolic():
    r = calculator("x + 1", precision=2)
    t = extract_text(r)

    assert "x + 1" in t


def test_force_numeric_applies_numeric_evaluation():
    r = calculator("sqrt(2)", precision=4, force_numeric=True)
    t = extract_text(r)

    assert "1.4142" in t
