"""Compatibility shim.

The orchestrator lives in ``d2ql.train`` so the installed console script
(``train`` in pyproject) points at an importable module. ``python main.py
--config ...`` keeps working exactly as before for the Docker entrypoints.
"""

from d2ql.train import main

if __name__ == "__main__":
    main()
