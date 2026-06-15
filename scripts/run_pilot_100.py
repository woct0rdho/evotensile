#!/usr/bin/env python3


from evotensile.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["pilot-yaml", "--output-yaml", "out/pilot_100.yaml", "--num-random", "64"]))
