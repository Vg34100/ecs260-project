# scripts/validity.py
from __future__ import annotations
import ast
from typing import Any, Dict, Tuple, Optional

def check_python_syntax(code: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Returns (valid, error_dict).
    error_dict is JSON-serializable and includes SyntaxError details when invalid.
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, {
            "msg": e.msg,
            "lineno": e.lineno,
            "offset": e.offset,
            "text": e.text.strip() if e.text else None,
        }
    except Exception as e:
        # Very rare, but keep it robust.
        return False, {
            "msg": f"{type(e).__name__}: {e}",
            "lineno": None,
            "offset": None,
            "text": None,
        }
