#!/usr/bin/env python3
"""
Lightweight project_state helpers.

Generates machine-derived artifacts under project_state/_generated:
- repo_inventory.json: file list with sizes and coarse role labels.
- symbol_index.json: AST-derived top-level classes/functions + docstrings.
- import_graph.json: internal import adjacency list.
- make_targets.txt: extracted Makefile targets.

Only stdlib is used. Excludes heavy/data/output directories to stay fast.
AST indexing is limited to src/, experiments/, and tools/ Python files.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = ROOT / "project_state" / "_generated"

# Directories we never descend into
EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "bundles",
    "reports",
    "data",
    "results",
    "agent_runs",
}

# Directory prefixes (relative) to skip entirely
EXCLUDE_PREFIXES = {
    "project_state/_generated",
    "docs/gpt_bundles",
}

# Specific heavy subdirectories to avoid within experiments/equity_panel
EQUITY_OUTPUT_PREFIXES = (
    "experiments/equity_panel/outputs",
    "experiments/equity_panel/outputs_",
    "experiments/equity_panel/figures",
    "experiments/equity_panel/sweeps",
)


def rel_path(path: Path) -> Path:
    return path.relative_to(ROOT)


def should_skip_dir(relative_dir: Path) -> bool:
    # Direct name blocklist
    name = relative_dir.name
    if name in EXCLUDE_DIR_NAMES:
        return True
    if name.startswith(("outputs", "popen", "pytest-", "unit_plots", "outputs_rc-lite")):
        return True
    rel_str = str(relative_dir).replace("\\", "/")
    for prefix in EXCLUDE_PREFIXES:
        if rel_str.startswith(prefix):
            return True
    for eq_prefix in EQUITY_OUTPUT_PREFIXES:
        if rel_str.startswith(eq_prefix):
            return True
    if rel_str.startswith("experiments/outputs") or rel_str.startswith("experiments/rc-lite"):
        return True
    return False


def categorize(path: Path) -> str:
    parts = path.parts
    if not parts:
        return "other"
    head = parts[0]
    if head == "src":
        return "src"
    if head == "tests":
        return "tests"
    if head == "experiments":
        return "experiments"
    if head == "tools":
        return "tools"
    if head == "docs":
        return "docs"
    if head == "project_state":
        return "project_state"
    if head in {"scripts", "ablations"}:
        return head
    return "other"


def collect_files() -> Tuple[List[dict], List[dict]]:
    inventory: List[dict] = []
    skipped: List[dict] = []
    for root, dirs, files in os.walk(ROOT):
        rel_root = rel_path(Path(root))
        # mutate dirs in place to avoid walking excluded subtrees
        dirs[:] = [
            d for d in dirs if not should_skip_dir(rel_root / d)
        ]
        if should_skip_dir(rel_root):
            skipped.append({"path": str(rel_root), "reason": "excluded directory"})
            continue
        for fname in files:
            rel_file = rel_root / fname
            if any(str(rel_file).replace("\\", "/").startswith(prefix) for prefix in EXCLUDE_PREFIXES):
                continue
            file_path = ROOT / rel_file
            try:
                stat = file_path.stat()
                size = stat.st_size
            except OSError:
                size = None
            inventory.append(
                {
                    "path": str(rel_file).replace("\\", "/"),
                    "size_bytes": size,
                    "category": categorize(rel_file),
                }
            )
    inventory.sort(key=lambda x: x["path"])
    return inventory, skipped


def module_name_from_path(py_path: Path) -> str:
    rel = rel_path(py_path)
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return ".".join(parts)


def get_py_files() -> List[Path]:
    py_files: List[Path] = []
    roots = [ROOT / "src", ROOT / "experiments", ROOT / "tools"]
    for base in roots:
        if not base.exists():
            continue
        for root, dirs, files in os.walk(base):
            rel_root = rel_path(Path(root))
            dirs[:] = [d for d in dirs if not should_skip_dir(rel_root / d)]
            if should_skip_dir(rel_root):
                continue
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                rel_file = rel_root / fname
                if any(
                    str(rel_file).replace("\\", "/").startswith(prefix)
                    for prefix in EXCLUDE_PREFIXES
                ):
                    continue
                py_files.append(ROOT / rel_file)
    return py_files


def _format_default(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return "..."


def _format_signature(args: ast.arguments) -> str:
    parts: List[str] = []

    posonly = list(args.posonlyargs)
    pos_args = list(args.args)
    total_pos = posonly + pos_args
    defaults = list(args.defaults)
    default_start = len(total_pos) - len(defaults)

    def fmt_arg(arg: ast.arg, idx: int) -> str:
        default = ""
        if idx >= default_start and defaults:
            default = _format_default(defaults[idx - default_start])
        return f"{arg.arg}={default}" if default else arg.arg

    for idx, arg in enumerate(total_pos):
        parts.append(fmt_arg(arg, idx))

    if posonly:
        parts.insert(len(posonly), "/")

    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")

    for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        default = _format_default(kw_default) if kw_default is not None else ""
        parts.append(f"{kw_arg.arg}={default}" if default else kw_arg.arg)

    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return "(" + ", ".join(parts) + ")"


def parse_symbols(py_files: Iterable[Path]) -> Tuple[Dict[str, dict], Dict[str, List[str]]]:
    symbol_index: Dict[str, dict] = {}
    import_graph: Dict[str, Set[str]] = {}

    # Build set of known module prefixes to recognise internal imports
    module_names = [module_name_from_path(p) for p in py_files]
    top_levels = {name.split(".")[0] for name in module_names}
    module_set = set(module_names)

    for path, mod_name in zip(py_files, module_names):
        rel_str = str(rel_path(path)).replace("\\", "/")
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception as exc:  # noqa: BLE001
            symbol_index[rel_str] = {
                "module": mod_name,
                "error": str(exc),
                "classes": [],
                "functions": [],
            }
            continue

        classes = []
        functions = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                bases = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except Exception:  # noqa: BLE001
                        bases.append("...")
                classes.append(
                    {
                        "name": node.name,
                        "lineno": node.lineno,
                        "doc": (doc.splitlines()[0] if doc else ""),
                        "bases": bases,
                    }
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                signature = _format_signature(node.args)
                functions.append(
                    {
                        "name": node.name,
                        "lineno": node.lineno,
                        "doc": (doc.splitlines()[0] if doc else ""),
                        "signature": signature,
                        "async": isinstance(node, ast.AsyncFunctionDef),
                    }
                )
        symbol_index[rel_str] = {
            "module": mod_name,
            "classes": classes,
            "functions": functions,
        }

        imports: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name.split(".")[0]
                    if target in top_levels:
                        imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    # Resolve relative imports
                    parent_parts = mod_name.split(".")[:-node.level]
                    base_parts = []
                    if base:
                        base_parts = base.split(".")
                    full = ".".join([*parent_parts, *base_parts])
                else:
                    full = base
                if full:
                    target = full.split(".")[0]
                    if target in top_levels:
                        imports.add(full)
        import_graph[mod_name] = sorted(imports)

    return symbol_index, {k: sorted(v) for k, v in import_graph.items()}


def extract_make_targets(makefile: Path) -> List[str]:
    targets: List[str] = []
    pattern = re.compile(r"^([A-Za-z0-9_.\\\\/%:-]+)\s*:")
    with makefile.open() as fh:
        for line in fh:
            stripped = line.strip()
            if (
                line.startswith("\t")
                or stripped.startswith("#")
                or ":=" in stripped
                or "?=" in stripped
            ):
                continue
            m = pattern.match(line)
            if m:
                target = m.group(1)
                targets.append(target)
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for t in targets:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    return deduped


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    inventory, skipped = collect_files()
    (GENERATED_DIR / "repo_inventory.json").write_text(
        json.dumps({"files": inventory, "skipped": skipped}, indent=2)
    )

    py_files = get_py_files()
    symbol_index, import_graph = parse_symbols(py_files)
    (GENERATED_DIR / "symbol_index.json").write_text(
        json.dumps(symbol_index, indent=2)
    )
    (GENERATED_DIR / "import_graph.json").write_text(
        json.dumps(import_graph, indent=2)
    )

    targets = extract_make_targets(ROOT / "Makefile")
    (GENERATED_DIR / "make_targets.txt").write_text("\n".join(targets) + "\n")


if __name__ == "__main__":
    main()
