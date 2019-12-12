#! /usr/bin/env python

import os
import sys

from barf.barf import BARF
from barf.arch import ARCH_ARM_MODE_ARM

if __name__ == "__main__":
    #
    # Open file
    #
    try:
        filename = os.path.abspath("../../bin/arm/branch4")
        barf = BARF(filename)
    except Exception as err:
        print err

        print "[-] Error opening file : %s" % filename

        sys.exit(1)

    #
    # Translate to REIL
    #
    print("[+] Translating: x86 -> REIL -> SMT...")

    for addr, asm_instr, reil_instrs in barf.translate(ea_start=0x000083c8, ea_end=0x00008404, arch_mode=ARCH_ARM_MODE_ARM):
        print("0x{0:08x} : {1}".format(addr, asm_instr))

        for reil_instr in reil_instrs:
            print("{0:14}{1}".format("", reil_instr))

            try:
                # Some instructions cannot be translate to SMT, i.e,
                # UNKN, UNDEF, JCC. In those cases, an exception is
                # raised.
                smt_exprs = barf.smt_translator.translate(reil_instr)

                for smt_expr in smt_exprs:
                    print("{0:16}{1}".format("", smt_expr))
            except:
                pass
