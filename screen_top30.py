#!/usr/bin/env python3
from __future__ import annotations

import sys

from tenbagger.screeners.top30_fused import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

