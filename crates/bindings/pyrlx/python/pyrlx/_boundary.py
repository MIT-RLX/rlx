# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Static findings where an agent's ``updater.py`` reaches past its program:
files and the network (``io``), code built at run time (``dynamic``), the
interpreter's own state (``introspection``: ``sys.modules``, ``__main__``,
``gc``, frames), and the model or loss (``model_mutation``).

Only the source text is read, as OPAL's boundary check reads it: common
import aliases are followed, but indirect access can get past it and a
finding can be harmless. The run records the findings (``boundary`` events,
the report); they never stop a program, and nothing here is a sandbox.
"""

from __future__ import annotations

import ast

_IO_MODULES = {"glob", "shutil", "subprocess", "socket", "urllib", "requests", "http", "httpx", "aiohttp"}
_DYNAMIC = {"__import__", "__builtins__", "eval", "exec", "compile"}
_PATH_METHODS = {
    "open", "read_text", "read_bytes", "write_text", "write_bytes", "iterdir", "glob", "rglob", "mkdir",
    "touch", "unlink", "rename", "rmdir", "tofile",
}
_SERIALIZE = {
    "load", "loads", "loadtxt", "genfromtxt", "save", "saves", "savetxt", "savez", "savez_compressed", "dump",
    "dumps", "memmap", "fromfile", "tofile",
}
_OS_IO = {
    "open", "read", "write", "listdir", "scandir", "walk", "fwalk", "system", "popen", "remove", "unlink",
    "rename", "replace", "mkdir", "makedirs", "rmdir", "removedirs",
}
_FRAMES = {
    "__globals__", "__code__", "__closure__", "__subclasses__", "__mro__", "f_back", "f_code", "f_globals",
    "f_locals", "gi_frame", "cr_frame", "ag_frame", "tb_frame",
}
_INTROSPECTION = {
    "globals", "sys.modules", "sys._getframe", "sys._current_frames", "gc.get_objects", "gc.get_referrers",
    "traceback.walk_stack", "traceback.walk_tb",
}
_MODEL_NAMES = {"model", "module", "trainer", "loss", "loss_fn", "loss_func", "loss_function", "criterion", "graph"}


def _name(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return _name(node.value, aliases) + "." + node.attr
    return ""


def _category(name):
    root, _, tail = name.partition(".")
    if root in _IO_MODULES or name in {"open", "io.open"}:
        return "io"
    if root == "os" and (tail.startswith("path.") or tail in _OS_IO):
        return "io"
    if root in {"numpy", "pickle", "pyrlx"} and name.rsplit(".", 1)[-1] in _SERIALIZE:
        return "io"
    if root in {"importlib", "builtins"} or name in _DYNAMIC:
        return "dynamic"
    if root in {"inspect", "__main__", "gc"} or name in _INTROSPECTION:
        return "introspection"
    return None


def _imported(node, aliases):
    return _name(node, aliases).split(".")[0] in {name.split(".")[0] for name in aliases.values()}


def _attribute(attr, obj, aliases):
    if attr in {"__builtins__", "__import__"}:
        return "dynamic"
    if attr in _FRAMES or (attr == "__dict__" and _imported(obj, aliases)):
        return "introspection"
    if attr in _PATH_METHODS:
        return "io"
    return None


def _model_object(node, aliases):
    name = _name(node, aliases)
    if isinstance(node, ast.Call) and _name(node.func, aliases) == "type" and node.args:
        return _model_object(node.args[0], aliases)
    return bool(_MODEL_NAMES.intersection(name.split(".")))


def _scan(files):
    flags = {}
    for file, source in sorted(files.items()):
        if not file.endswith(".py"):
            continue
        tree = ast.parse(source, filename=file)
        lines, aliases = source.splitlines(), {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for a in node.names:
                    aliases[a.asname or a.name] = node.module + "." + a.name

        def flag(node, category, file=file, lines=lines):
            if category:
                line = node.lineno
                flags[file, line, category] = {
                    "file": file, "line": line, "category": category, "snippet": lines[line - 1].strip()[:200],
                }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    flag(node, _category(a.name))
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                flag(node, _category(node.module))
                for a in node.names:
                    flag(node, _category(node.module + "." + a.name))
            elif isinstance(node, (ast.Name, ast.Attribute)):
                flag(node, _category(_name(node, aliases)))
                if isinstance(node, ast.Attribute):
                    flag(node, _attribute(node.attr, node.value, aliases))
                    if isinstance(node.ctx, ast.Store) and _model_object(node.value, aliases):
                        flag(node, "model_mutation")
            elif isinstance(node, ast.Call):
                func = _name(node.func, aliases)
                if func in {"getattr", "setattr", "delattr"} and len(node.args) >= 2:
                    obj, attr = node.args[:2]
                    if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
                        flag(node, _attribute(attr.value, obj, aliases))
                    if func != "getattr" and _model_object(obj, aliases):
                        flag(node, "model_mutation")
                if func == "vars" and node.args and _imported(node.args[0], aliases):
                    flag(node, "introspection")
    return [flags[k] for k in sorted(flags)]


def check(files):
    """``{file, line, category, snippet}`` findings for the ``.py`` files in
    ``files`` (name → source); ``[]`` when the source does not parse (the
    build reports that) or the check itself fails. Never raises."""
    try:
        return _scan(files)
    except BaseException:  # a diagnostic must not stop training
        return []
