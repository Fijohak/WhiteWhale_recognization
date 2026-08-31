#!/usr/bin/env python3
"""阻止三分钟自动部署执行破坏旧 Release 的数据库迁移。"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


FORBIDDEN_CALLS = {
    "drop_column", "drop_constraint", "drop_index", "drop_table",
    "rename_table", "alter_column", "execute",
}
DESTRUCTIVE_SQL = re.compile(
    r"\b(DROP|TRUNCATE|DELETE\s+FROM|ALTER\s+TABLE|UPDATE\s+)\b",
    re.IGNORECASE,
)


def validate(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    failures: list[str] = []
    upgrade = next((node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "upgrade"), None)
    if upgrade is None:
        return [f"{path}: 缺少 upgrade()"]
    for node in ast.walk(upgrade):
        if not isinstance(node, ast.Call) \
                or not isinstance(node.func, ast.Attribute):
            continue
        if isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "op" \
                and node.func.attr in FORBIDDEN_CALLS:
            failures.append(
                f"{path}:{node.lineno}: op.{node.func.attr} 禁止自动部署")
        for argument in node.args:
            if isinstance(argument, ast.Constant) \
                    and isinstance(argument.value, str) \
                    and DESTRUCTIVE_SQL.search(argument.value):
                failures.append(
                    f"{path}:{node.lineno}: 检测到破坏性 SQL")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    failures = [failure for path in args.files for failure in validate(path)]
    if failures:
        print("\n".join(failures))
        return 1
    print(f"expand-only migration check passed: {len(args.files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
