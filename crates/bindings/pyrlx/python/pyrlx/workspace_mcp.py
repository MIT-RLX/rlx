# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""``python -m pyrlx.workspace_mcp --dir <folder> [--write <name>]...``

The workspace MCP server an agent CLI talks to during an ``AgentOptimizer``
decision: it reads every file in the folder and writes only the names given
with ``--write``. The same server as ``rlx-agent workspace-mcp``, built into
this extension so nothing else needs installing.
"""

import sys

from pyrlx import _pyrlx

if __name__ == "__main__":
    _pyrlx._serve_workspace_mcp(sys.argv[1:])
