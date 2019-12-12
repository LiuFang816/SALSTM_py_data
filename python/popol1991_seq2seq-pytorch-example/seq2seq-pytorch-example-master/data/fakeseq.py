import sys
import random

src = open(sys.argv[1], 'w')
dst = open(sys.argv[2], 'w')

for _ in range(int(sys.argv[3])):
    l = random.randint(1, 10)
    # l = 10
    x = [random.randint(0, 9)]
    for _ in range(l-1):
        x.append((x[-1] + 1) % 10)
    src.write(" ".join([str(e) for e in x]))
    src.write("\n")
    dst.write(" ".join([str((e + 1) % 10) for e in x]))
    dst.write("\n")
