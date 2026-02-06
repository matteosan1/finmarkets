from pathlib import Path
import sys

testdir = Path(__file__).resolve().parent
srcdir = testdir / '../finmarkets'
sys.path.insert(0, str(srcdir.resolve()))