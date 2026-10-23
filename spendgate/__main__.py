"""`python -m spendgate` — the same CLI as the installed `spendgate` command."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
