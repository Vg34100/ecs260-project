"""AST utilities for stability metrics."""
from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional


class _NameNormalizer(ast.NodeTransformer):
    """Normalize identifier names to stable placeholders."""

    def __init__(self) -> None:
        self._name_map: Dict[str, str] = {}
        self._counter = 0

    def _get_name(self, original: str) -> str:
        if original not in self._name_map:
            self._counter += 1
            self._name_map[original] = f"var_{self._counter}"
        return self._name_map[original]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._get_name(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg=self._get_name(node.arg), annotation=node.annotation), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = "func"
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = "func"
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = "Class"
        self.generic_visit(node)
        return node


class _DocstringStripper(ast.NodeTransformer):
    """Remove docstrings from modules, classes, and functions."""

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    @staticmethod
    def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                return body[1:]
        return body


def _wrap_in_function(code: str) -> str:
    indented = "\n".join(("    " + line) if line.strip() else line for line in code.splitlines())
    return "def _wrapped():\n" + (indented if indented else "    pass")


def parse_python(code: str) -> Optional[ast.AST]:
    try:
        return ast.parse(code)
    except SyntaxError:
        pass
    except IndentationError:
        pass
    try:
        return ast.parse(_wrap_in_function(code))
    except Exception:
        return None


def normalize_ast(tree: ast.AST) -> ast.AST:
    tree = _DocstringStripper().visit(tree)
    tree = _NameNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def ast_fingerprint(code: str) -> Optional[str]:
    tree = parse_python(code)
    if not tree:
        return None
    norm = normalize_ast(tree)
    return ast.dump(norm, include_attributes=False)


def ast_tokens(code: str) -> Optional[List[str]]:
    fingerprint = ast_fingerprint(code)
    if fingerprint is None:
        return None
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", fingerprint)


def ast_to_tree(tree: ast.AST) -> "_TreeNode":
    return _TreeNode.from_ast(tree)


class _TreeNode:
    """Simple tree for edit distance."""

    def __init__(self, label: str, children: list["_TreeNode"]) -> None:
        self.label = label
        self.children = children

    @classmethod
    def from_ast(cls, node: ast.AST) -> "_TreeNode":
        label = type(node).__name__
        children = []
        for child in ast.iter_child_nodes(node):
            children.append(cls.from_ast(child))
        return cls(label, children)
