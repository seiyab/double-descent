#!/usr/bin/env python3

import subprocess


def main():
    subprocess.check_call(["pipenv", "run", "check"])


main()
