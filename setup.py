# setup.py – retained only for users who still type python setup.py ..."
import sys

sys.stderr.write("""⚠️  PySR uses pyproject.toml instead of setup.py.

Install from a checkout with:
    python -m pip install .       # normal
    python -m pip install -e .    # editable (pip ≥21.3)

Or install from PyPI with:
    pip install pysr
""")
sys.exit(1)
