from __future__ import annotations

import sys

from train import main


if __name__ == "__main__":
    main(["--model", "vgg16", *sys.argv[1:]])

