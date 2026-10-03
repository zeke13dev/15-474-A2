"""Run every test_* function in the test_*.py files (pytest also works if installed)."""

import importlib
import sys
import traceback
from pathlib import Path

failed = 0
for path in sorted(Path(__file__).parent.glob("test_*.py")):
    module = importlib.import_module(path.stem)
    for name in sorted(n for n in dir(module) if n.startswith("test_")):
        try:
            getattr(module, name)()
            print(f"ok    {path.stem}.{name}")
        except Exception:
            failed += 1
            print(f"FAIL  {path.stem}.{name}")
            traceback.print_exc()
sys.exit(1 if failed else 0)
