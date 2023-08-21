#!/usr/bin/env python3

import os
import pathlib
import subprocess


def main():
	for d in dirs():
		hook = pathlib.Path(d, "tools", "pre-commit.py")
		if not hook.exists():
			continue
		subprocess.check_call("./tools/pre-commit.py", cwd=d)


def git_root():
	return pathlib.Path(bytes.decode(subprocess.check_output(["git", "rev-parse", "--show-toplevel"])).strip())

def dirs():
	root = git_root()
	ls = os.listdir(root)
	paths = (pathlib.Path(root, f) for f in ls if not f.startswith('.') and f != 'tools')
	return [p for p in paths if p.is_dir()]


main()
