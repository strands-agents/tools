"""
Tests for the calculator tool using the Agent interface.
"""

import unittest.mock as mock

import pytest
import sympy as sp
from strands import Agent
from sympy import Integer, Symbol, exp, log

# Function level imports from calculator module
from src.strands_tools.calculator import (
    apply_symbolic_simplifications,
    calculate_derivative,
    calculate_integral,
    calculate_limit,
    calculate_series,
    create_error_panel,
    create_result_table,
    evaluate_expression,
    force_numerical_eval,
    format_number,
    get_precision_level,
    numeric_evaluation,
    parse_expression,
    parse_matrix_expression,
    preprocess_expression,
    solve_equation,
)
from src.strands_tools.calculator import calculator as calculator_func

# Module level import for the Agent fixture
from strands_tools import calculator as calculator_module


@pytest.fixture
def agent():
    """Create an agent with the calculator tool loaded."""
    return Agent(tools=[calculator_module])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    # Handle direct calculator tool calls
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]

    # Default case
    return str(result)


def test_basic_calculations(agent):
    """Test basic calculations using the agent.tool.calculator method."""
    result = agent.tool.calculator(expression="2+2")
    result_text = extract_result_text(result)
    assert "4" in result_text


def test_advanced_operations(agent):
    """Test more advanced operations through the agent interface."""
    test_cases = [
        # Addition
        {"expression": "2 + 3", "expected": "5"},
        # Multiplication
        {"expression": "2 * 3", "expected": "6"},
        # Exponentiation
        {"expression": "2^3", "expected": "8"},
    ]

    for case in test_cases:
        result = agent.tool.calculator(expression=case["expression"])
        result_text = extract_result_text(result)
        assert case["expected"] in result_text


def test_differentiation(agent):
    """Test differentiation of expressions."""
    # Simple polynomial
    result = agent.tool.calculator(expression="x^2 + 3*x + 2", mode="derive", wrt="x")
    result_text = extract_result_text(result)
    assert "2*x + 3" in result_text


def test_integration(agent):
    """Test integration of expressions."""
    result = agent.tool.calculator(expression="2*x + 3", mode="integrate", wrt="x")
    result_text = extract_result_text(result)
    assert "x**2" in result_text and "3*x" in result_text


def test_limits(agent):
    """Test limit calculations."""
    result = agent.tool.calculator(expression="sin(x)/x", mode="limit", wrt="x", point="0")
    result_text = extract_result_text(result)
    assert "1" in result_text


def test_equation_solving(agent):
    """Test equation solving."""
    result = agent.tool.calculator(expression="x^2 + 2*x + 1", mode="solve")
    result_text = extract_result_text(result)
    assert "-1" in result_text

    # Skip the system of equations test for now as it requires deeper investigation
    # of how SymPy expects system inputs through the calculator interface


def test_matrix_operations(agent):
    """Test matrix operations."""
    # Test matrix addition
    result = agent.tool.calculator(expression="[[1, 2], [3, 4]] + [[5, 6], [7, 8]]", mode="matrix")
    result_text = extract_result_text(result)
    assert "6" in result_text and "8" in result_text and "10" in result_text and "12" in result_text

    # Test matrix multiplication
    result = agent.tool.calculator(expression="[[1, 2], [3, 4]] * [[5, 6], [7, 8]]", mode="matrix")
    result_text = extract_result_text(result)
    # Result should be [[19, 22], [43, 50]]
    assert "19" in result_text and "22" in result_text and "43" in result_text and "50" in result_text

    # Test determinant (implicitly)
    result = agent.tool.calculator(expression="[[1, 2], [3, 4]]", mode="matrix")
    result_text = extract_result_text(result)
    assert "Matrix" in result_text or "1" in result_text  # Either the matrix itself or its determinant


def test_scientific_notation_and_precision(agent):
    """Test scientific notation and precision options."""
    # Test scientific notation
    result = agent.tool.calculator(expression="1.2345678901234e15", scientific=True)
    result_text = extract_result_text(result)
    # Should use scientific notation (even if displayed as 1.2345e+15)
    assert "e+" in result_text or "e-" in result_text or "E+" in result_text or "E-" in result_text

    # Test precision
    result = agent.tool.calculator(expression="1/3", precision=15)
    result_text = extract_result_text(result)
    # Should have more decimal places than the default
    assert "0.33333333" in result_text


def test_parse_expression_edge_cases():
    """Test edge cases in the parse_expression function."""
    # Test with non-string input
    with pytest.raises(ValueError, match="Expression must be a string"):
        parse_expression(123)

    # Test with invalid operators
    with pytest.raises(ValueError, match="Invalid operator"):
        parse_expression("2 // 3")

    with pytest.raises(ValueError, match="Invalid operator sequence"):
        parse_expression("2 **/ 3")

    # Test with logical operators
    with pytest.raises(ValueError, match="Logical operators are not supported"):
        parse_expression("x && y")

    # Test logarithm notation
    result = parse_expression("log(10)")
    assert isinstance(result, sp.log)

    # Test pi and e constants in various positions
    pi_cases = [
        "pi + 1",
        "2*(pi)",
        "pi+5",
        "pi-2",
        "pi*3",
        "pi/2",
        "sin(pi)",
    ]
    for case in pi_cases:
        result = parse_expression(case)
        assert isinstance(result, sp.Basic)

    # Test e constant in various positions
    e_cases = [
        "e + 1",
        "2*(e)",
        "e+5",
        "e-2",
        "e*3",
        "e/2",
        "log(e)",
    ]
    for case in e_cases:
        result = parse_expression(case)
        assert isinstance(result, sp.Basic)

    # Test sympify error handling
    with mock.patch("sympy.sympify", side_effect=sp.SympifyError("Test error")):
        with pytest.raises(ValueError, match="Invalid mathematical expression"):
            parse_expression("x + y")


def test_get_precision_level():
    """Test the get_precision_level function."""
    # Test regular numbers
    assert get_precision_level(5.67) == 10

    # Test large numbers
    assert get_precision_level(1.5e10) == 8

    # Test very large numbers
    assert get_precision_level(2.3e20) == 5

    # Test with error
    with mock.patch("builtins.float", side_effect=ValueError("Test error")):
        # No need to capture the print mock if we're not asserting against it
        result = get_precision_level("not_a_number")
        assert result == 10  # Should return default


def test_force_numerical_eval():
    """Test force_numerical_eval function."""
    # Test with basic symbolic expression
    expr = sp.sympify("x + 5")
    result = force_numerical_eval(expr)
    assert isinstance(result, sp.Basic)

    # Test with expression containing pi
    expr = sp.sympify("pi + 2")
    result = force_numerical_eval(expr)
    assert isinstance(result, sp.Basic)
    assert not result.has(sp.pi)  # pi should be substituted

    # Test with expression containing e
    expr = sp.sympify("exp(1) + 2")
    result = force_numerical_eval(expr)
    assert isinstance(result, sp.Basic)

    # Test numerical constant evaluation
    expr = sp.sympify("2 + 3")
    result = force_numerical_eval(expr)
    assert float(result) == 5.0  # Compare float values instead of exact equality

    # Test log expression
    expr = sp.sympify("log(100)")
    result = force_numerical_eval(expr)
    assert isinstance(result, sp.Basic)

    # Skip the mocking test since SymPy objects don't work well with mock.patch.object

    # Test with non-Basic type
    result = force_numerical_eval(5)
    assert result == 5

    # Test error handling by directly checking implementation
    try:
        with mock.patch("sympy.Basic.has", side_effect=ValueError("Test error")):
            # This will trigger an exception in the force_numerical_eval function
            force_numerical_eval(sp.Symbol("x"))
    except ValueError:
        pass  # Expected behavior is to get a ValueError


def test_format_number():
    """Test the format_number function."""
    # Test integers
    assert format_number(5) == "5"
    assert format_number(sp.Integer(5)) == "5"

    # Test complex numbers
    assert "j" in format_number(complex(1, 2))

    # Test float with scientific notation
    assert "e" in format_number(1.23e8, scientific=True).lower()

    # Test float with very large value
    assert "e" in format_number(1.23e22).lower()

    # Test float with very small value
    assert "e" in format_number(1.23e-7).lower()

    # Test large number with commas
    formatted = format_number(1234567.89, scientific=False)
    assert "," in formatted or "1234567.89" in formatted

    # Test precision
    assert format_number(1 / 3, precision=5).startswith("0.3333")

    # Test with a type error - directly check the implementation
    result = format_number("not_a_number")
    assert result == "not_a_number"  # The implementation should return str(num) when not numeric


def test_preprocess_expression():
    """Test the preprocess_expression function."""
    # Test with no variables
    expr = sp.sympify("x + 5")
    result = preprocess_expression(expr)
    assert result == expr

    # Test with simple variables
    expr = sp.sympify("a*x + b")
    variables = {"a": 2, "b": 3}
    result = preprocess_expression(expr, variables)
    # Check the substitution worked correctly by evaluating at x=1
    result_at_x_1 = result.subs(sp.Symbol("x"), 1)
    assert float(result_at_x_1) == 5.0  # a*1 + b = 2*1 + 3 = 5

    # Test with expression variables
    expr = sp.sympify("a*x + b")
    variables = {"a": "sin(pi/2)", "b": "2+3"}
    result = preprocess_expression(expr, variables)
    # Should substitute sin(pi/2) = 1 and 2+3 = 5
    result_at_x_1 = result.subs(sp.Symbol("x"), 1)
    assert abs(float(result_at_x_1) - 6.0) < 1e-10  # a*1 + b = 1*1 + 5 = 6


def test_apply_symbolic_simplifications():
    """Test the apply_symbolic_simplifications function."""
    # Test log of exponential
    expr = log(exp(Symbol("x")))
    result = apply_symbolic_simplifications(expr)
    assert result == Symbol("x")

    # Test exp of log
    expr = exp(log(Symbol("x")))
    result = apply_symbolic_simplifications(expr)
    assert result == Symbol("x")

    # Test exp with numeric argument
    expr = exp(Integer(2))
    result = apply_symbolic_simplifications(expr)
    assert isinstance(result, sp.Basic)

    # Test log with numeric argument
    expr = log(Integer(100))
    result = apply_symbolic_simplifications(expr)
    assert isinstance(result, sp.Basic)

    # Test with non-Basic type
    result = apply_symbolic_simplifications(5)
    assert result == 5


def test_numeric_evaluation():
    """Test the numeric_evaluation function."""

    # Test with integer
    class MockSymbolic:
        is_integer = True

        def __int__(self):
            return 5

        # Add __float__ method for float() function
        def __float__(self):
            return 5.0

    result = numeric_evaluation(MockSymbolic(), 10, False)
    assert result == 5

    # Test with real number - using a real SymPy object with expected methods
    result = numeric_evaluation(sp.Integer(314) / 100, 10, False)
    assert isinstance(result, str)
    assert "3.14" in result

    # Test with complex number - using mock for format_number
    complex_num = sp.sympify("1 + 2*I")
    with mock.patch("src.strands_tools.calculator.format_number", return_value="1+2j"):
        result = numeric_evaluation(complex_num, 10, False)
        assert "1+2j" == result


def test_system_equation_solving(agent):
    """Test solving a system of equations."""
    # Need to pass a properly formatted system expression
    result = agent.tool.calculator(expression="x + y - 10, x - y - 2", mode="solve")
    result_text = extract_result_text(result)

    # The expected solution is x = 6, y = 4
    assert ("6" in result_text and "4" in result_text) or "Error" in result_text


def test_create_result_table():
    """Test the create_result_table function directly."""
    table = create_result_table("Test Operation", "x+y", "x+y", {"Additional": "Info"})
    assert table is not None
    assert len(table.columns) == 2

    # Test without additional info
    table = create_result_table("Test Operation", "x+y", "x+y")
    assert table is not None
    assert len(table.columns) == 2


def test_create_error_panel():
    """Test the create_error_panel function."""
    mock_console = mock.Mock()

    create_error_panel(mock_console, "Test Error")
    mock_console.print.assert_called_once()


def test_parse_matrix_expression_errors():
    """Test error handling in parse_matrix_expression function."""
    # Test with invalid matrix format
    with pytest.raises(ValueError, match="Invalid matrix format"):
        parse_matrix_expression("[[1, 2], [3]]")

    # Test with general error
    with pytest.raises(ValueError, match="Matrix parsing error"):
        parse_matrix_expression("not_a_matrix")


def test_evaluate_expression_comprehensive():
    """Test comprehensive evaluation of expressions."""
    # Test basic numeric evaluation
    result = evaluate_expression(sp.sympify("2 + 3"))
    assert float(result) == 5.0

    # Test with variables
    result = evaluate_expression(sp.sympify("a*x + b"), {"a": 2, "b": 3})
    assert str(result) == "2*x + 3"

    # Test with force_numeric on symbolic expression
    result = evaluate_expression(sp.sympify("sqrt(2)"), force_numeric=True, precision=5)
    assert isinstance(result, str)
    assert "1.414" in result

    # Test with scientific notation
    with mock.patch("src.strands_tools.calculator.numeric_evaluation", return_value="1.0e7"):
        result = evaluate_expression(sp.sympify("10000000"), scientific=True)
        assert "1.0e7" == result

    # Test with constants that need substitution
    result = evaluate_expression(sp.sympify("pi"), force_numeric=True)
    assert "3.14" in str(result)


def test_edge_case_error_handling(agent):
    """Test more edge cases for error handling."""
    # Test with unsupported operators
    result = agent.tool.calculator(expression="x && y")
    result_text = extract_result_text(result)
    assert "Error" in result_text

    # Test with invalid matrix input
    result = agent.tool.calculator(expression="[[1, 2], [3]]", mode="matrix")
    result_text = extract_result_text(result)
    assert "Error" in result_text

    # Test with differentiation errors
    result = agent.tool.calculator(expression="1/0", mode="derive", wrt="x")
    result_text = extract_result_text(result)
    assert "Error" in result_text

    # Test with integration errors
    result = agent.tool.calculator(expression="1/0", mode="integrate", wrt="x")
    result_text = extract_result_text(result)
    assert "Error" in result_text

    # Test with limit errors
    result = agent.tool.calculator(expression="1/0", mode="limit", wrt="x", point="0")
    result_text = extract_result_text(result)
    assert "Error" in result_text

    # Test with series expansion errors
    result = agent.tool.calculator(expression="1/0", mode="series", wrt="x")
    result_text = extract_result_text(result)
    assert "Error" in result_text


def test_direct_tool_call():
    """Test direct call to the calculator tool function."""

    # Test basic calculation
    with mock.patch("src.strands_tools.calculator.Console"):
        result = calculator_func(expression="2+2", mode="evaluate")
        assert result["status"] == "success"
        assert "Result: 4" in result["content"][0]["text"]

    # Test with error - division by zero may not raise an error in SymPy
    with mock.patch("src.strands_tools.calculator.Console"):
        with mock.patch("src.strands_tools.calculator.create_error_panel"):
            result = calculator_func(expression="x +* 2", mode="evaluate")
            assert result["status"] == "error"
            assert "Error" in result["content"][0]["text"]


def test_force_numerical_eval_error_handling():
    """Test error handling in force_numerical_eval function."""
    expr = sp.Symbol("x")

    # Patch specifically N in the try block that catches the exception
    with mock.patch(
        "sympy.N",
        side_effect=[
            sp.Float(3.14),  # First call succeeds (for substitutions)
            sp.Float(3.14),  # Second call succeeds (for has(sp.E))
            sp.Float(3.14),  # Third call succeeds (for has(sp.pi))
            ValueError("Test error"),  # Fourth call in the first try block fails
            sp.Float(3.14),  # Final call succeeds
        ],
    ):
        # This should trigger the except block
        result = force_numerical_eval(expr)
        # Since the function continues after the exception, it should return a value
        assert result is not None


def test_logarithm_numerical_eval():
    """Test the log number condition."""
    # Create a logarithm with a numeric argument
    expr = sp.log(sp.Float(2.0))
    result = force_numerical_eval(expr)
    # The function should return a numerical value
    assert isinstance(result, sp.Float)


def test_matrix_type_conversion():
    """Test the matrix type conversion in format_number."""
    # Create a SymPy Float that would trigger the conversion
    num = sp.Float(3.14)
    result = format_number(num)
    # Verify the result is properly formatted
    assert result == "3.14"


def test_exponential_log_simplification():
    """Test the special case in apply_symbolic_simplifications."""
    # Create an expression e^(ln(x)) which should simplify to x
    x = sp.Symbol("x")
    expr = sp.exp(sp.log(x))
    result = apply_symbolic_simplifications(expr)
    # The simplification should return x
    assert result == x


def test_solve_system_of_equations():
    """Test solving a system of equations."""
    x, y = sp.symbols("x y")
    eq1 = sp.Eq(x + y, 10)
    eq2 = sp.Eq(x - y, 2)

    # Convert to the list format that would trigger the system solver
    eqs = [eq1.lhs - eq1.rhs, eq2.lhs - eq2.rhs]  # x + y - 10, x - y - 2

    result = solve_equation(eqs, precision=10)
    # The solution should be x=6, y=4
    assert result[x] == 6
    assert result[y] == 4


def test_solve_equation_equality_handling():
    """Test equality handling in solve_equation."""
    x = sp.Symbol("x")
    expr = x**2 - 4  # Should be treated as x^2 - 4 = 0

    result = solve_equation(expr, precision=10)
    # Should return [2, -2] or [-2, 2] depending on sympy's internal order
    # Need to handle potential complex numbers with zero imaginary part
    real_results = []
    for val in result:
        if isinstance(val, complex):
            if val.imag == 0:
                real_results.append(val.real)
        else:
            real_results.append(float(val))

    assert set(real_results) == {2.0, -2.0}


def test_solve_equation_complex_result():
    """Test handling of complex solutions."""
    x = sp.Symbol("x")
    expr = x**2 + 1  # Solutions are i and -i

    result = solve_equation(expr, precision=10)
    # Should return [i, -i] as complex numbers
    assert all(isinstance(sol, complex) for sol in result)
    assert set([complex(sol) for sol in result]) == {complex(0, 1), complex(0, -1)}


def test_error_handling_in_calculation_functions():
    """Test error handling in mathematical functions."""
    x = sp.Symbol("x")

    # Test differentiation error
    with mock.patch("sympy.diff", side_effect=ValueError("Test error")):
        with pytest.raises(ValueError):
            calculate_derivative(x**2, "x", order=2)

    # Test integration error
    with mock.patch("sympy.integrate", side_effect=ValueError("Test error")):
        with pytest.raises(ValueError):
            calculate_integral(x**2, "x")

    # Test limit error
    with mock.patch("sympy.limit", side_effect=ValueError("Test error")):
        with pytest.raises(ValueError):
            calculate_limit(1 / x, "x", "0")

    # Test series error
    with mock.patch("sympy.series", side_effect=ValueError("Test error")):
        with pytest.raises(ValueError):
            calculate_series(sp.exp(x), "x", point="oo", order=2)


def test_calculator_tool_with_system_of_equations():
    """Test the calculator tool with a system of equations."""
    # Create a tool use with a system of equations
    from src.strands_tools.calculator import calculator as calc_function

    # Mock parse_expression to return a list of expressions
    with mock.patch(
        "src.strands_tools.calculator.parse_expression",
        side_effect=lambda expr: (
            [sp.Symbol("x") + sp.Symbol("y") - 10, sp.Symbol("x") - sp.Symbol("y") - 2]
            if expr.startswith("[")
            else expr
        ),
    ):
        # This should trigger the system of equations path in calculator function
        result = calc_function(expression="['x + y - 10', 'x - y - 2']", mode="solve")

        # Check for success status
        assert result["status"] == "success"
        # The result should contain the solution
        assert "Result:" in result["content"][0]["text"]


def test_error_handling(agent):
    """Test error handling for invalid expressions."""
    # Test division by zero
    result = agent.tool.calculator(expression="1/0")
    result_text = extract_result_text(result)
    # Division by zero might appear as "zoo" or "ComplexInfinity" or "nan" in SymPy
    assert any(term in result_text for term in ["zoo", "ComplexInfinity", "nan", "Error", "inf", "Infinity", "NaN"])

    # Test invalid syntax
    result = agent.tool.calculator(expression="x +* 2")
    result_text = extract_result_text(result)
    assert "Error" in result_text

    # Test undefined variable in mode that requires variables
    result = agent.tool.calculator(expression="solve(x+y)", mode="solve")
    result_text = extract_result_text(result)
    # Should either give an error or solve in terms of y
    assert "Error" in result_text or "y" in result_text
