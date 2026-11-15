"""Safe evaluation of the arithmetic expressions allowed in scene files.

Coordinates in a scene file are numbers or strings such as ``"width - 0.6"``
or ``"max(dx * 2, 0.03)"``. Only numbers, the names passed in, + - * /,
parentheses and ``min``/``max`` are accepted.
"""
import ast
import operator
from typing import Dict, Union

Number = Union[int, float]

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCTIONS = {
    "min": min,
    "max": max,
}


def evaluate(value: Union[Number, str], names: Dict[str, float]) -> float:
  """Evaluate a number or an arithmetic expression string"""
  if isinstance(value, bool):
    raise ValueError(f"expected a number or expression, got {value!r}")
  if isinstance(value, (int, float)):
    return float(value)
  if not isinstance(value, str):
    raise ValueError(f"expected a number or expression, got {value!r}")
  try:
    tree = ast.parse(value, mode="eval")
  except SyntaxError as error:
    raise ValueError(f"invalid expression {value!r}: {error.msg}") from None
  return float(_evaluate_node(tree.body, names, value))


def _evaluate_node(node: ast.AST, names: Dict[str, float], source: str) -> Number:
  if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
          and not isinstance(node.value, bool):
    return node.value
  if isinstance(node, ast.Name):
    if node.id not in names:
      raise ValueError(
          f"unknown name '{node.id}' in {source!r}, use one of: {', '.join(names)}")
    return names[node.id]
  if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
    return _BINARY_OPERATORS[type(node.op)](
        _evaluate_node(node.left, names, source),
        _evaluate_node(node.right, names, source))
  if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
    return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand, names, source))
  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
          and node.func.id in _FUNCTIONS and not node.keywords and node.args:
    return _FUNCTIONS[node.func.id](
        *(_evaluate_node(arg, names, source) for arg in node.args))
  raise ValueError(f"unsupported expression {source!r}")
