# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""``python -m pyrlx.agent <command> ...``: the command line over runs of
``pyrlx.AgentOptimizer`` (the same tool as ``rlx-agent-opt``)::

    python -m pyrlx.agent doctor [--live] [--backend claude]
    python -m pyrlx.agent estimate --total-steps N --sec-per-step S
    python -m pyrlx.agent show <run>
    python -m pyrlx.agent watch <run> [--timeout S]
    python -m pyrlx.agent report <run> [--out page.html]
    python -m pyrlx.agent compare <a> <b>
    python -m pyrlx.agent say <run> <text...>
    python -m pyrlx.agent broker <box> [--backend claude]
"""

import sys

from pyrlx import _pyrlx


def main(argv=None):
    return _pyrlx._agent_cli(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
