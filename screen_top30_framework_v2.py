#!/usr/bin/env python3
from __future__ import annotations

import sys

from tenbagger.screeners.top30_framework_v2 import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

