#!/usr/bin/env python3

import subprocess


def main():
    black()


def black():
    files = bytes.decode(subprocess.check_output(["git", "ls-files"])).strip().split()
    py = [f for f in files if f.endswith(".py")]
    subprocess.check_call(["pipenv", "run", "black", "--check"] + py)


main()
