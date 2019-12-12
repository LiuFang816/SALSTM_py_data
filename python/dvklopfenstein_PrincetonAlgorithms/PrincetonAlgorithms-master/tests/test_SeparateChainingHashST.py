#!/usr/bin/env python

from AlgsSedgewickWayne.SeparateChainingHashST import SeparateChainingHashST
from AlgsSedgewickWayne.testcode.chaining_table import prt_chaining_symtbl
import fileinput
import sys

def test_stdin(prt=sys.stdout):
  """echo "a b c d e f" | test_SeparateChainingHashST.py"""
  # 1. Create and fill SeparateChainingHashST
  M, keys = cli()
  st = SeparateChainingHashST(M) # Initialize empty chaining-hash
  for i, key in enumerate(keys):
    st.put(key, i)

  # 2. Print SeparateChainingHashST
  prt.write("\n")
  for s in st.keys():
    prt.write("{} {}\n".format(s, st.get(s)))

  # 3. Print innards of SeparateChainingHashST
  st.prt_chaining_symtbl(prt)

def cli():
  """Command-line interface: reads data from stdin, stream, or files."""
  items_in = [w.rstrip("\n\r") for t in fileinput.input() for w in t.split(" ")]
  M = None
  keys = []
  for key in items_in:
    if key[:2] == "M=":
      M = int(key[2:])  
    else:
      keys.append(key)
  return M, keys


if __name__ == '__main__':
  test_stdin()
