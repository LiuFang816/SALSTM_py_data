from types import CodeType
from .check import check_stmts
from .interp import shell_interp, Array
from .parser import Parser

__all__ = ('load_source', 'load_file', 'Array')

def code_template():
    shell_interp = 'shell_interp'
    table = 'table'
    shell_interp(table, globals())


def make_code(table):
    co = code_template.func_code
    d = {"shell_interp": shell_interp, "table": table}
    consts = tuple(d.get(x,x)  for x in co.co_consts)

    code = CodeType(
        co.co_argcount,
        co.co_nlocals,
        co.co_stacksize,
        co.co_flags,
        co.co_code,
        consts,
        co.co_names,
        co.co_varnames,
        co.co_filename,
        co.co_name,
        co.co_firstlineno,
        co.co_lnotab
    )
    return code


def load_source(source):
    p = Parser(filename='<string>')
    stmts = p.parse(source)
    table = check_stmts(stmts)
    return make_code(table)


def load_file(*filenames):
    stmts = ()

    for filename in filenames:
        p = Parser(filename=filename)
        with open(filename, "r") as f:
            source = f.read()
        stmts += p.parse(source)

    table = check_stmts(stmts)
    return make_code(table)
