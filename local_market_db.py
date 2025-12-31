#!/usr/bin/env python3
from __future__ import annotations

import sys

from tenbagger.tools.local_market_db import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
