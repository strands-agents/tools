"""
Calculator tool powered by SymPy for comprehensive mathematical operations.

This module provides a powerful mathematical calculation engine built on SymPy
that can handle everything from basic arithmetic to advanced calculus, equation solving,
and matrix operations. It's designed to provide formatted, precise results with
proper error handling and robust type conversion.

Key Features:
1. Expression Evaluation:
   • Basic arithmetic operations (addition, multiplication, etc.)
   • Trigonometric functions (sin, cos, tan, etc.)
   • Logarithmic operations and special constants (e, pi)
   • Complex number handling with proper formatting

2. Specialized Mathematical Operations:
   • Equation solving (single equations and systems)
   • Differentiation (single and higher-order derivatives)
   • Integration (indefinite integrals)
   • Limit calculation (at specified points or infinity)
   • Series expansions (Taylor and Laurent series)
   • Matrix operations (determinants, multiplication, etc.)

3. Display and Formatting:
   • Configurable precision for numeric results
   • Scientific notation support for large/small numbers
   • Symbolic results when appropriate
   • Rich formatted output with tables and panels

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import calculator

agent = Agent(tools=[calculator])

# Basic arithmetic evaluation
agent.tool.calculator(expression="2 * sin(pi/4) + log(e**2)")

# Equation solving
agent.tool.calculator(expression="x**2 + 2*x + 1", mode="solve")

# Calculate derivative
agent.tool.calculator(
    expression="sin(x)",
    mode="derive",
    wrt="x",
    order=2
)

# Calculate integral
agent.tool.calculator(
    expression="x**2 + 2*x",
    mode="integrate",
    wrt="x"
)
```

See the calculator function docstring for more details on available modes and parameters.
"""

import ast
import logging
import os
from typing import Any, Dict, Optional, Union

# Required dependencies
import sympy as sp
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from strands import tool

from strands_tools.utils import console_util

logger = logging.getLogger(__name__)


def create_result_table(
    operation: str,
    input_expr: str,
    result: Any,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Table:
    """Create a formatted table with the calculation results."""
    table = Table(show_header=False, box=box.ROUNDED)
    table.add_column("Operation", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Operation", operation)
    table.add_row("Input", str(input_expr))
    table.add_row("Result", str(result))

    if additional_info:
        for key, value in additional_info.items():
            table.add_row(key, str(value))

    return table


def create_error_panel(console: Console, error_message: str) -> None:
    """Create and print an error panel."""
    console.print(
        Panel(
            f"[red]Error: {error_message}[/red]",
            title="[bold red]Calculation Error[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
    )


def parse_expression(expr_str: str) -> Any:
    """Parse a string expression into a SymPy expression."""
    try:
        # Validate expression string
        if not isinstance(expr_str, str):
            raise ValueError("Expression must be a string")

        # Replace common mathematical notations
        expr_str = expr_str.replace("^", "**")

        # Handle logarithm notations
        if "log(" in expr_str:
            expr_str = expr_str.replace("log(", "ln(")  # Convert to natural log

        # Pre-process pi and e for better evaluation - using word boundaries to avoid replacing 'e' in function names
        expr_str = expr_str.replace(" pi ", " " + str(sp.N(sp.pi, 50)) + " ")
        expr_str = expr_str.replace("(pi)", "(" + str(sp.N(sp.pi, 50)) + ")")
        expr_str = expr_str.replace("pi+", str(sp.N(sp.pi, 50)) + "+")
        expr_str = expr_str.replace("pi-", str(sp.N(sp.pi, 50)) + "-")
        expr_str = expr_str.replace("pi*", str(sp.N(sp.pi, 50)) + "*")
        expr_str = expr_str.replace("pi/", str(sp.N(sp.pi, 50)) + "/")
        expr_str = expr_str.replace("pi)", str(sp.N(sp.pi, 50)) + ")")

        # Handle standalone 'e' constant but preserve function names like 'exp'
        expr_str = expr_str.replace(" e ", " " + str(sp.N(sp.E, 50)) + " ")
        expr_str = expr_str.replace("(e)", "(" + str(sp.N(sp.E, 50)) + ")")
        expr_str = expr_str.replace("e+", str(sp.N(sp.E, 50)) + "+")
        expr_str = expr_str.replace("e-", str(sp.N(sp.E, 50)) + "-")
        expr_str = expr_str.replace("e*", str(sp.N(sp.E, 50)) + "*")
        expr_str = expr_str.replace("e/", str(sp.N(sp.E, 50)) + "/")
        expr_str = expr_str.replace("e)", str(sp.N(sp.E, 50)) + ")")

        # Basic validation for common invalid patterns
        if "//" in expr_str:  # Catch integer division which is not supported
            raise ValueError("Invalid operator: //. Use / for division.")

        if "**/" in expr_str:  # Catch power/division confusion
            raise ValueError("Invalid operator sequence: **/")

        if any(op in expr_str for op in ["&&", "||", "&", "|"]):  # Catch logical operators
            raise ValueError("Logical operators are not supported in mathematical expressions")

        try:
            # First try parsing with pre-evaluated constants
            expr = sp.sympify(expr_str, evaluate=True)  # type: ignore

            # If we got any symbolic constants, substitute their values
            if expr.has(sp.pi) or expr.has(sp.E):
                expr = expr.subs({sp.pi: sp.N(sp.pi, 50), sp.E: sp.N(sp.E, 50)})

            return expr

        except sp.SympifyError as e:
            raise ValueError(f"Invalid mathematical expression: {str(e)}") from e

    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}") from e


def get_precision_level(num: Union[float, int, sp.Expr]) -> int:
    """Determine appropriate precision based on number magnitude."""
    try:
        abs_num = abs(float(num))
        if abs_num >= 1e20:
            return 5  # Less precision for very large numbers
        elif abs_num >= 1e10:
            return 8  # Medium precision for large numbers
        else:
            return 10  # Full precision for regular numbers
    except (ValueError, TypeError) as e:
        # Log specific error for debugging
        logger.debug(f"Precision calculation error: {str(e)}")
        return 10  # Default precision for non-numeric or special cases


def force_numerical_eval(expr: Any, precision: int = 50) -> Any:
    """Force numerical evaluation of symbolic expressions."""
    try:
        if isinstance(expr, sp.Basic):
            # First substitute numeric values for constants
            substitutions = {
                sp.pi: sp.N(sp.pi, precision),
                sp.E: sp.N(sp.E, precision),
                sp.exp(1): sp.N(sp.E, precision),
                sp.I: sp.I,  # Keep i symbolic for complex numbers
            }

            # Handle special cases
            if expr.has(sp.E):
                expr = expr.subs(sp.E, sp.N(sp.E, precision))
            if expr.has(sp.pi):
                expr = expr.subs(sp.pi, sp.N(sp.pi, precision))

            # Try direct numerical evaluation
            try:
                result = sp.N(expr, precision)
                if not result.free_symbols:  # If we got a fully numeric result
                    return result
            except (ValueError, TypeError, ZeroDivisionError) as eval_error:
                logger.debug(f"Numerical evaluation error: {str(eval_error)}")
                # Continue to next attempt

            # If direct evaluation didn't work, try step-by-step evaluation
            expr = expr.rewrite(sp.exp)  # Rewrite trig functions in terms of exp
            expr = expr.subs(substitutions)
            if isinstance(expr, sp.log):
                if expr.args[0].is_number:
                    return sp.N(expr, precision)

            # Final attempt at numerical evaluation
            result = sp.N(expr, precision)
            return result
        return expr
    except Exception as e:
        raise ValueError(f"Could not evaluate numerically: {str(e)}") from e


def format_number(
    num: Any,
    scientific: bool = False,
    precision: int = 10,
    force_scientific_threshold: float = 1e21,
) -> str:
    """Format number with control over notation."""

    force_scientific_threshold = float(os.getenv("CALCULATOR_FORCE_SCIENTIFIC_THRESHOLD", "1e21"))

    # If it's not a number, just return its string representation
    if not isinstance(num, (int, float, complex, sp.Basic)):
        return str(num)

    # Handle integers directly
    if isinstance(num, int):
        return str(num)
    if isinstance(num, sp.Basic) and hasattr(num, "is_Integer") and num.is_Integer:
        try:
            return str(int(float(str(num))))
        except Exception:
            return str(num)

    # Handle complex numbers
    if isinstance(num, complex):
        # Format real part
        if num.real == 0:
            real_part = "0"
        elif abs(num.real) >= 1e6 or (0 < abs(num.real) < 1e-6) or scientific:
            # Scientific notation
            adjusted_precision = get_precision_level(num.real)
            real_part = f"{num.real:.{adjusted_precision}e}"
        else:
            # Standard notation
            real_part = f"{num.real:.{precision}f}".rstrip("0").rstrip(".")

        # Format imaginary part
        if num.imag == 0:
            return real_part

        if abs(num.imag) >= 1e6 or (0 < abs(num.imag) < 1e-6) or scientific:
            # Scientific notation
            adjusted_precision = get_precision_level(num.imag)
            imag_part = f"{abs(num.imag):.{adjusted_precision}e}"
        else:
            # Standard notation
            imag_part = f"{abs(num.imag):.{precision}f}".rstrip("0").rstrip(".")

        # Combine parts
        sign = "+" if num.imag > 0 else "-"
        if real_part == "0":
            if sign == "+":
                return f"{imag_part}j"
            else:
                return f"-{imag_part}j"
        return f"{real_part}{sign}{imag_part}j"

    # Try to convert SymPy complex to Python complex
    if hasattr(num, "is_complex") and num.is_complex:
        try:
            # First convert to float to ensure compatibility
            python_complex = complex(float(sp.re(num)), float(sp.im(num)))
            return format_number(python_complex, scientific, precision, force_scientific_threshold)
        except Exception:
            return str(num)

    # Handle SP.Float - convert to Python float
    if isinstance(num, sp.Float):
        try:
            return format_number(float(num), scientific, precision, force_scientific_threshold)
        except Exception:
            return str(num)

    # Handle regular floats
    if isinstance(num, float):
        abs_num = abs(num)

        # Determine if scientific notation should be used
        use_scientific = scientific or (abs_num >= force_scientific_threshold) or (0 < abs_num < 1e-6)

        if use_scientific:
            # Use scientific notation
            adjusted_precision = get_precision_level(num)
            return f"{num:.{adjusted_precision}e}"

        if abs_num >= 1e6:
            # Use commas for large numbers
            return f"{num:,.2f}"

        # Standard notation with proper rounding
        result = f"{num:.{precision}f}"
        if "." in result:
            result = result.rstrip("0").rstrip(".")
        return result

    # Last resort - string representation
    return str(num)


def preprocess_expression(expr: Any, variables: Optional[Dict[str, Any]] = None) -> Any:
    """Preprocess an expression by substituting variables and constants."""
    if variables:
        # Convert variable values to SymPy objects
        sympy_vars = {sp.Symbol(k): parse_expression(str(v)) for k, v in variables.items()}
        result = expr.subs(sympy_vars)
    else:
        result = expr
    return result


def apply_symbolic_simplifications(expr: Any) -> Any:
    """Apply symbolic simplifications to expressions."""
    result = expr

    # Only attempt simplifications on symbolic expressions
    if isinstance(result, sp.Basic):
        # Handle logarithms of exponentials: log(e^x) = x
        if isinstance(result, sp.log) and isinstance(result.args[0], sp.exp):
            result = result.args[0].args[0]

        # Handle exponentials: e^(ln(x)) = x
        elif isinstance(result, sp.exp) and isinstance(result.args[0], sp.log):
            result = result.args[0].args[0]

        # Handle powers of e: e^x
        elif isinstance(result, sp.exp):
            if all(arg.is_number for arg in result.args):
                result = result.evalf()

        # Handle logarithms with numeric arguments
        elif isinstance(result, sp.log):
            if result.args[0].is_number:
                result = result.evalf()

        # Try general simplification for expressions with special constants
        result = sp.simplify(result)

    return result


def numeric_evaluation(result: Any, precision: int, scientific: bool) -> Union[int, float, str, sp.Expr]:
    """Convert symbolic results to numeric form when possible."""
    try:
        # Check if the result is an integer
        if hasattr(result, "is_integer") and result.is_integer:
            return int(result)  # Return as integer to maintain precision

        # For floating point, evaluate numerically
        if isinstance(result, sp.Basic):
            if hasattr(result, "is_real") and result.is_real:
                float_result = float(result.evalf())  # type: ignore
            else:
                # Handle complex numbers
                complex_result = complex(result.evalf())  # type: ignore
                return format_number(complex_result, scientific, precision)
        else:
            float_result = float(result)

        # Format based on scientific notation preference
        return format_number(float_result, scientific, precision)
    except (TypeError, ValueError) as e:
        if isinstance(result, sp.Basic):
            # If we can't convert to float, return the evaluated form
            return result.evalf(precision)  # type: ignore
        raise ValueError(f"Could not evaluate expression numerically: {str(e)}") from e


def evaluate_expression(
    expr: Any,
    variables: Optional[Dict[str, Any]] = None,
    precision: int = 10,
    scientific: bool = False,
    force_numeric: bool = False,
) -> Union[Any, int, float, str]:
    """Evaluate a mathematical expression with optional variables."""
    try:
        # Step 1: Apply variable substitutions
        result = preprocess_expression(expr, variables)

        # Step 2: Apply numerical substitutions for constants if forcing numeric
        if force_numeric and isinstance(result, sp.Basic):
            substitutions = {
                sp.pi: sp.N(sp.pi, precision),
                sp.E: sp.N(sp.E, precision),
                sp.exp(1): sp.N(sp.E, precision),
            }
            result = result.subs(substitutions)

        # Step 3: Apply symbolic simplifications
        result = apply_symbolic_simplifications(result)

        # Step 4: Force numerical evaluation if requested
        if force_numeric and isinstance(result, sp.Basic):
            # Try direct numerical evaluation first
            try:
                numeric_result = sp.N(result, precision)
                if not numeric_result.free_symbols:
                    result = numeric_result
                else:
                    # If that didn't fully evaluate, use force_numerical_eval
                    result = force_numerical_eval(result, precision)
            except Exception as eval_error:
                # Log the specific error for debugging
                logger.debug(f"Numeric evaluation error: {str(eval_error)}")
                result = force_numerical_eval(result, precision)

        # Step 5: If the result still has symbols and we're not forcing numeric, return symbolic
        if hasattr(result, "free_symbols") and result.free_symbols and not force_numeric:
            return result

        # Step 6: Otherwise, perform numeric evaluation and formatting
        return numeric_evaluation(result, precision, scientific)

    except Exception as e:
        raise ValueError(f"Evaluation error: {str(e)}") from e


def solve_equation(expr: Any, precision: int) -> Any:
    """Solve an equation or system of equations."""
    try:
        # Handle system of equations
        if isinstance(expr, list):
            # Get all variables in the system
            variables = set().union(*[eq.free_symbols for eq in expr])
            solution = sp.solve(expr, list(variables))
            return solution

        # Single equation
        if not isinstance(expr, sp.Equality):
            expr = sp.Eq(expr, 0)

        # Get variables from the equation and convert to list
        variables_set = expr.free_symbols
        if not variables_set:
            return None  # No variables to solve for

        variables_list = list(variables_set)
        solution = sp.solve(expr, variables_list[0])

        # Convert to float if possible
        if isinstance(solution, (list, tuple)):
            return [complex(s.evalf(precision)) if isinstance(s, sp.Expr) else s for s in solution]
        return complex(solution.evalf(precision)) if isinstance(solution, sp.Expr) else solution
    except Exception as e:
        raise ValueError(f"Solving error: {str(e)}") from e


def calculate_derivative(expr: Any, var: str, order: int) -> Any:
    """Calculate derivative of expression."""

    try:
        # Check for undefined expressions like 1/0 before attempting differentiation
        try:
            # Try to evaluate the expression to check if it's valid
            test_value = expr.evalf()
            if test_value.has(sp.zoo) or test_value.has(sp.oo) or test_value.has(-sp.oo) or test_value.has(sp.nan):
                raise ValueError(f"Cannot differentiate an undefined expression: {expr}") from None
        except (sp.SympifyError, TypeError, ZeroDivisionError):
            # If evaluation fails, the expression might be undefined
            raise ValueError(f"Cannot differentiate an undefined expression: {expr}") from None

        var_sym = sp.Symbol(var)
        return sp.diff(expr, var_sym, order)
    except Exception as e:
        raise ValueError(f"Differentiation error: {str(e)}") from e


def calculate_integral(expr: Any, var: str) -> Any:
    """Calculate indefinite integral of expression."""
    try:
        # Check for undefined expressions like 1/0 before attempting integration
        try:
            # Try to evaluate the expression to check if it's valid
            test_value = expr.evalf()
            if test_value.has(sp.zoo) or test_value.has(sp.oo) or test_value.has(-sp.oo) or test_value.has(sp.nan):
                raise ValueError(f"Cannot integrate an undefined expression: {expr}") from None
        except (sp.SympifyError, TypeError, ZeroDivisionError):
            # If evaluation fails, the expression might be undefined
            raise ValueError(f"Cannot integrate an undefined expression: {expr}") from None

        var_sym = sp.Symbol(var)
        return sp.integrate(expr, var_sym)
    except Exception as e:
        raise ValueError(f"Integration error: {str(e)}") from e


def calculate_limit(expr: Any, var: str, point: str) -> Any:
    """Calculate limit of expression."""
    try:
        # Check for undefined expressions like 1/0 before attempting to calculate limit
        try:
            # Try to evaluate the expression to check if it's valid
            test_value = expr.evalf()
            if test_value.has(sp.zoo) or test_value.has(sp.oo) or test_value.has(-sp.oo) or test_value.has(sp.nan):
                raise ValueError(f"Cannot calculate limit of an undefined expression: {expr}") from None
        except (sp.SympifyError, TypeError, ZeroDivisionError):
            # If evaluation fails, the expression might be undefined
            raise ValueError(f"Cannot calculate limit of an undefined expression: {expr}") from None

        var_sym = sp.Symbol(var)
        point_val = sp.sympify(point)
        return sp.limit(expr, var_sym, point_val)
    except Exception as e:
        raise ValueError(f"Limit calculation error: {str(e)}") from e


def calculate_series(expr: Any, var: str, point: str, order: int) -> Any:
    """Calculate series expansion of expression."""
    try:
        # Check for undefined expressions like 1/0 before attempting series expansion
        try:
            # Try to evaluate the expression to check if it's valid
            test_value = expr.evalf()
            if test_value.has(sp.zoo) or test_value.has(sp.oo) or test_value.has(-sp.oo) or test_value.has(sp.nan):
                raise ValueError(f"Cannot expand series of an undefined expression: {expr}") from None
        except (sp.SympifyError, TypeError, ZeroDivisionError):
            # If evaluation fails, the expression might be undefined
            raise ValueError(f"Cannot expand series of an undefined expression: {expr}") from None

        var_sym = sp.Symbol(var)
        point_val = sp.sympify(point)
        return sp.series(expr, var_sym, point_val, order)
    except Exception as e:
        raise ValueError(f"Series expansion error: {str(e)}") from e


def parse_matrix_expression(expr_str: str) -> Any:
    """Parse matrix expression and perform operations."""
    try:
        # Function to safely convert string to matrix
        def safe_matrix_from_str(matrix_str: str) -> Any:
            # Use ast.literal_eval for safe evaluation of matrix literals
            try:
                matrix_data = ast.literal_eval(matrix_str.strip())
                return sp.Matrix(matrix_data)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Invalid matrix format: {str(e)}") from e

        # Split into parts for operations
        parts = expr_str.split("*")
        if len(parts) == 2:
            # Handle multiplication
            matrix1 = safe_matrix_from_str(parts[0])
            matrix2 = safe_matrix_from_str(parts[1])
            return matrix1 * matrix2
        elif "+" in expr_str:
            # Handle addition
            parts = expr_str.split("+")
            matrix1 = safe_matrix_from_str(parts[0])
            matrix2 = safe_matrix_from_str(parts[1])
            return matrix1 + matrix2
        else:
            # Single matrix operations
            return safe_matrix_from_str(expr_str)
    except Exception as e:
        raise ValueError(f"Matrix parsing error: {str(e)}") from e


@tool
def calculator(
    expression: str,
    mode: str = None,
    precision: int = None,
    scientific: bool = None,
    force_numeric: bool = None,
    variables: dict = None,
    wrt: str = None,
    point: str = None,
    order: int = None,
) -> dict:
    """
    Calculator powered by SymPy for comprehensive mathematical operations.

    This tool provides advanced mathematical functionality through multiple operation modes,
    including expression evaluation, equation solving, calculus operations (derivatives, integrals),
    limits, series expansions, and matrix operations. Results are formatted with appropriate
    precision and can be displayed in scientific notation when needed.

    How It Works:
    ------------
    1. The function parses the mathematical expression using SymPy's parser
    2. Based on the selected mode, it routes the expression to the appropriate handler
    3. Variables and constants are substituted with their values when provided
    4. The expression is evaluated symbolically and/or numerically as appropriate
    5. Results are formatted based on precision preferences and value magnitude
    6. Rich output is generated with operation details and formatted results

    Operation Modes:
    --------------
    - evaluate: Calculate the value of a mathematical expression
    - solve: Find solutions to an equation or system of equations
    - derive: Calculate derivatives of an expression
    - integrate: Find the indefinite integral of an expression
    - limit: Evaluate the limit of an expression at a point
    - series: Generate series expansion of an expression
    - matrix: Perform matrix operations

    Common Usage Scenarios:
    ---------------------
    - Basic calculations: Evaluating arithmetic expressions
    - Equation solving: Finding roots of polynomials or systems of equations
    - Calculus: Computing derivatives and integrals for analysis
    - Engineering analysis: Working with scientific notations and constants
    - Mathematics education: Visualizing step-by-step solutions
    - Data science: Matrix operations and statistical calculations

    Args:
        expression: The mathematical expression to evaluate, such as "2 + 2 * 3",
            "x**2 + 2*x + 1", or "sin(pi/2)". For matrix operations, use array
            notation like "[[1, 2], [3, 4]]".
        mode: The calculation mode to use. Options are:
            - "evaluate": Compute the value of the expression (default)
            - "solve": Solve an equation or system of equations
            - "derive": Calculate the derivative of an expression
            - "integrate": Find the indefinite integral of an expression
            - "limit": Calculate the limit of an expression at a point
            - "series": Generate a series expansion of an expression
            - "matrix": Perform matrix operations
        precision: Number of decimal places for the result (default: 10).
            Higher values provide more precise output but may impact performance.
        scientific: Whether to use scientific notation for numbers (default: False).
            When True, formats large and small numbers using scientific notation.
        force_numeric: Force numeric evaluation of symbolic expressions (default: False).
            When True, tries to convert symbolic results to numeric values.
        variables: Optional dictionary of variable names and their values to substitute
            in the expression, e.g., {"a": 1, "b": 2}.
        wrt: Variable to differentiate or integrate with respect to (required for
            "derive" and "integrate" modes).
        point: Point at which to evaluate a limit (required for "limit" mode).
            Use "oo" for infinity.
        order: Order of derivative or series expansion (optional for "derive" and
            "series" modes, default is 1 for derivatives and 5 for series).

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [{"text": "Result: <calculated_result>"}]
        }

        Success case: Returns the calculation result with appropriate formatting
        Error case: Returns information about what went wrong during calculation

    Notes:
        - For equation solving, set the expression equal to zero implicitly (x**2 + 1 means x**2 + 1 = 0)
        - Use 'pi' and 'e' for mathematical constants
        - The 'wrt' parameter is required for differentiation and integration
        - Matrix expressions use Python-like syntax: [[1, 2], [3, 4]]
        - Precision control impacts display only, internal calculations use higher precision
        - Symbolic results are returned when possible unless force_numeric=True
    """
    console = console_util.create()

    try:
        # Get environment variables at runtime for all parameters
        mode = os.getenv("CALCULATOR_MODE", "evaluate") if mode is None else mode
        precision = int(os.getenv("CALCULATOR_PRECISION", "10")) if precision is None else precision
        scientific = os.getenv("CALCULATOR_SCIENTIFIC", "False").lower() == "true" if scientific is None else scientific
        force_numeric = (
            os.getenv("CALCULATOR_FORCE_NUMERIC", "False").lower() == "true" if force_numeric is None else force_numeric
        )
        default_order = int(os.getenv("CALCULATOR_DERIVE_ORDER", "1"))
        default_series_point = os.getenv("CALCULATOR_SERIES_POINT", "0")
        default_series_order = int(os.getenv("CALCULATOR_SERIES_ORDER", "5"))

        # Extract parameters
        variables = variables or {}

        # Parse the expression
        if mode == "matrix":
            expr = parse_matrix_expression(expression)
        else:
            expr = parse_expression(expression)

        # Process based on mode
        additional_info = {}

        if mode == "solve":
            if isinstance(expr, list):
                result = solve_equation(expr, precision)
                operation = "Solve System of Equations"
            else:
                result = solve_equation(expr, precision)
                operation = "Solve Equation"

        elif mode == "derive":
            var = wrt or str(list(expr.free_symbols)[0])
            actual_order = order or default_order
            result = calculate_derivative(expr, var, actual_order)
            operation = f"Calculate {actual_order}-th Derivative"
            additional_info = {"With respect to": var}

        elif mode == "integrate":
            var = wrt or str(list(expr.free_symbols)[0])
            result = calculate_integral(expr, var)
            operation = "Calculate Integral"
            additional_info = {"With respect to": var}

        elif mode == "limit":
            var = wrt or str(list(expr.free_symbols)[0])
            point_val = point or default_series_point
            result = calculate_limit(expr, var, point_val)
            operation = "Calculate Limit"
            additional_info = {"Variable": var, "Point": point_val}

        elif mode == "series":
            var = wrt or str(list(expr.free_symbols)[0])
            point_val = point or default_series_point
            actual_order = order or default_series_order
            result = calculate_series(expr, var, point_val, actual_order)
            operation = "Calculate Series Expansion"
            additional_info = {"Variable": var, "Point": point_val, "Order": actual_order}

        elif mode == "matrix":
            result = expr
            operation = "Matrix Operation"

        else:  # evaluate
            result = evaluate_expression(expr, variables, precision, scientific, force_numeric)
            operation = "Evaluate Expression"
            if force_numeric:
                additional_info["Note"] = "Forced numerical evaluation"
            if scientific:
                additional_info["Format"] = "Scientific notation"
                additional_info["Format"] = "Scientific notation"

        # Create and display result table
        table = create_result_table(operation, expression, result, additional_info)
        console.print(
            Panel(
                table,
                title="[bold blue]Calculation Result[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )

        return {
            "status": "success",
            "content": [{"text": f"Result: {result}"}],
        }

    except Exception as e:
        create_error_panel(console, str(e))
        return {
            "status": "error",
            "content": [{"text": f"Error: {str(e)}"}],
        }
