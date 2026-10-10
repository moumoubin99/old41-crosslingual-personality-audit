#!/usr/bin/env python3
"""Build bounded exact-source excerpts for the pinned vLLM rater closure."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _StripDocstrings(ast.NodeTransformer):
    """Drop non-executable docstrings from excerpts; full-file hashes stay exact."""

    def _visit_body(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        return node

    visit_Module = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body
    visit_ClassDef = _visit_body


def functional_ast(source: str) -> ast.AST:
    tree = _StripDocstrings().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return tree


def ast_sha256(source: str) -> str:
    tree = functional_ast(source)
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return sha256_bytes(canonical.encode("utf-8"))


def canonical_source(source: str) -> str:
    """Keep executable AST semantics while removing packet-only formatting/docs."""
    return ast.unparse(functional_ast(source))


def semantic_fragment(source: str, qualified_name: str) -> tuple[str, str]:
    """Return the exact definition or import binding for one public symbol."""
    tree = ast.parse(source)
    parts = qualified_name.split(".")
    top = parts[0]
    node = next(
        (item for item in tree.body
         if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
         and item.name == top),
        None,
    )
    if node is not None and len(parts) > 1:
        if not isinstance(node, ast.ClassDef):
            raise ValueError(f"{qualified_name}: dotted symbol is not rooted at a class")
        node = next(
            (item for item in node.body
             if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and item.name == parts[1]),
            None,
        )
    if node is not None:
        fragment = ast.get_source_segment(source, node)
        if fragment:
            return "definition", fragment
    for item in tree.body:
        if isinstance(item, ast.ImportFrom):
            for alias in item.names:
                if (alias.asname or alias.name) == top:
                    fragment = ast.get_source_segment(source, item)
                    if fragment:
                        return "import_binding", fragment
        elif isinstance(item, ast.Import):
            for alias in item.names:
                if (alias.asname or alias.name.split(".")[-1]) == top:
                    fragment = ast.get_source_segment(source, item)
                    if fragment:
                        return "import_binding", fragment
    # Some public callables are selected by a module-level feature branch
    # (for example the FlashAttention import/fallback dispatch). Preserve the
    # complete enclosing statement so every alternative binding stays visible.
    for item in tree.body:
        if not isinstance(
            item,
            (ast.Assign, ast.AnnAssign, ast.If, ast.Try, ast.With, ast.For),
        ):
            continue
        binds_name = any(
            (isinstance(child, ast.Name) and child.id == top
             and isinstance(child.ctx, ast.Store))
            or (isinstance(child, ast.alias)
                and (child.asname or child.name.split(".")[-1]) == top)
            for child in ast.walk(item)
        )
        if binds_name:
            fragment = ast.get_source_segment(source, item)
            if fragment:
                return "conditional_binding", fragment
    raise ValueError(f"cannot locate source for {qualified_name}")


def build(project: Path, runtime_manifest: Path) -> dict:
    manifest = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    modules = []
    for entry in manifest["dependency_closure"]:
        snapshot = entry.get("snapshot")
        if not snapshot:
            continue
        path = (project / snapshot).resolve()
        source = path.read_text(encoding="utf-8")
        if sha256_bytes(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"runtime snapshot hash changed: {entry['module']}")
        symbols = []
        declared = entry.get("symbols") or []
        if declared:
            for name in declared:
                kind, fragment = semantic_fragment(source, name)
                canonical = canonical_source(fragment)
                symbols.append({
                    "symbol": name,
                    "kind": kind,
                    "source": canonical,
                    "ast_sha256": ast_sha256(canonical),
                })
        else:
            canonical = canonical_source(source)
            symbols.append({
                "symbol": "<module>",
                "kind": "complete_module",
                "source": canonical,
                "ast_sha256": ast_sha256(canonical),
            })
        modules.append({
            "module": entry["module"],
            "snapshot": snapshot,
            "file_sha256": entry["sha256"],
            "symbols": symbols,
        })
    return {
        "version": 1,
        "runtime_manifest": str(runtime_manifest.relative_to(project)),
        "python_dependency_count": len(modules),
        "modules": modules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    parser.add_argument(
        "--runtime-manifest",
        default="process/consumption/deployment/mistral-rater-runtime-v2-manifest.json",
    )
    parser.add_argument(
        "--output",
        default="process/consumption/deployment/mistral-rater-semantic-closure-v1.json",
    )
    args = parser.parse_args()
    project = Path(args.project).resolve()
    runtime = (project / args.runtime_manifest).resolve()
    output = (project / args.output).resolve()
    if not runtime.is_relative_to(project) or not output.is_relative_to(project):
        raise ValueError("semantic closure paths must remain inside the project")
    payload = build(project, runtime)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
