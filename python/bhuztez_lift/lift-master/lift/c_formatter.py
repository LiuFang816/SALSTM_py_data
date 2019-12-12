def format_consts(table, ctx):
    for k,v in ctx.const_values.items():
        shape = table.vars[k].shape[::-1]
        if len(shape) == 0:
            yield "float %s[1]={%s};\n" % (k, v[0])
        else:
            yield "float %s%s={%s};\n" % (
                k,
                "".join("[%d]"%(s,) for s in shape),
                ",".join("%f"%(x,) for x in v))


def format_vars(table,arrays):
    for k in arrays:
        shape = table.vars[k].shape[::-1]
        if len(shape) == 0:
            yield "float %s[1];\n" % (k,)
        else:
            yield "float %s%s;\n"%(k,"".join("[%d]"%(s,) for s in shape))


def format_arguments(table, arrays):
    for v in arrays:
        shape = table.vars[table.symbols[v]].shape[::-1]
        if len(shape) <= 1:
            yield "float *{} = {};\n".format(table.symbols[v], v)
        else:
            yield "float (*{}){} = s->{};\n".format(
                table.symbols[v],
                "".join("[%d]"%(s,) for s in shape[1:]),
                v)



C_TEMPLATE = """#include <math.h>
#include "{name}_c.h"
#define inf (1.0/0.0)
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

void
{name}(struct {name}_state *s){{
{inputs}{outputs}{vars}{body}
}}
"""

def format_c(name, table, ctx, ast):
    assert not ctx.const_arrays
    return C_TEMPLATE.format(
        name = name,
        inputs = "".join(format_arguments(table, table.inputs)),
        outputs = "".join(format_arguments(table, table.outputs)),
        vars = "".join(format_vars(table, ctx.intermediate_arrays.keys())),
        body = format_ast(table, ast)
    )


INT_FUN_TYPE = {
    '<': 'infix',
    '>': 'infix',
    '<=': 'infix',
    '>=': 'infix',
    '==': 'infix',
    '!=': 'infix',
    '&&': 'infix',
    '||': 'infix',
    '+': 'infix',
    '-': 'infix',
    '*': 'infix',
    '/': 'infix',
    '_': "prefix",
    '%': 'infix',
    'max': 'fun2',
    'min': 'fun2',
    'select': 'select'
}

INT_FUN_FMT = {
    '_': '-',
}

def format_int_call(fun, args):
    t = INT_FUN_TYPE[fun]

    if t == 'infix':
        return "({} {} {})".format(format_int_expr(args[0]), INT_FUN_FMT.get(fun,fun), format_int_expr(args[1]))
    elif t == 'prefix':
        return "({} {})".format(INT_FUN_FMT.get(fun,fun), format_int_expr(args[0]))
    elif t == 'fun1':
        return "({}({}))".format(INT_FUN_FMT.get(fun,fun), format_int_expr(args[0]))
    elif t == 'fun2':
        return "({}({},{}))".format(INT_FUN_FMT.get(fun,fun), format_int_expr(args[0]), format_int_expr(args[1]))
    elif t == 'select':
        return "(({})?({}):({}))".format(format_int_expr(args[0]), format_int_expr(args[1]), format_int_expr(args[2]))

    raise NotImplementedError


def format_int_expr(ast):
    if ast[0] == 'var':
        return '{}'.format(ast[1])
    elif ast[0] == 'int':
        return "%d"%(ast[1],)
    elif ast[0] == 'call':
        return format_int_call(ast[1], ast[2])

    raise NotImplementedError


def format_element(v, indices):
    return "%s%s"%(v, "".join("[%s]"%(format_int_expr(i),) for i in indices))


FUN_TYPE = {
    '+': 'infix',
    '-': 'infix',
    '*': 'infix',
    '/': 'infix',
    '_': 'prefix',
    '**': 'fun2',
    'exp': 'fun1',
    'log': 'fun1',
    '==': 'infix',
    'max': 'fun2',
    'min': 'fun2',
}

FUN_FMT = {
    '_': '-',
    '**': 'powf',
    'exp': 'expf',
    'log': 'logf',
    'max': 'fmax',
    'min': 'fmin',
}

def format_call(table, fun, args):
    t = FUN_TYPE[fun]
    if t == 'infix':
        return "({} {} {})".format(format_ast(table, args[0]), FUN_FMT.get(fun,fun), format_ast(table, args[1]))
    elif t == 'prefix':
        return "({} {})".format(FUN_FMT.get(fun,fun), format_ast(table, args[0]))
    elif t == 'fun1':
        return "({}({}))".format(FUN_FMT.get(fun, fun), format_ast(table, args[0]))
    elif t == 'fun2':
        return "({}({},{}))".format(FUN_FMT.get(fun, fun), format_ast(table, args[0]), format_ast(table, args[1]))

    raise NotImplementedError


def format_ast(table, ast, format_kernel=None):
    if isinstance(ast, list):
        return "".join(format_ast(table, node, format_kernel) for node in ast)
    if ast[0] == 'for':
        return "for(int {v} = {init}; {cond}; {v} = {v} + {inc}){{\n{body}}}\n".format(
            v = format_int_expr(ast[1]),
            init = format_int_expr(ast[2]),
            inc = format_int_expr(ast[3]),
            cond = format_int_expr(ast[4]),
            body = format_ast(table, ast[5], format_kernel))
    elif ast[0] == 'if':
        return "if({cond}){{\n{then}}}\n".format(
            cond = format_int_expr(ast[1]),
            then = format_ast(table, ast[2], format_kernel))
    elif ast[0] == 'ifelse':
        return "if({cond}){{\n{then}}}\nelse{{\n{else_}}}".format(
            cond = format_int_expr(ast[1]),
            then = format_ast(table, ast[2], format_kernel),
            else_ = format_ast(table, ast[3], format_kernel))
    elif ast[0] == 'assign':
        assert ast[1][0] == 'element'
        if len(ast[1][2]) == 0:
            assert len(ast[1][2]) == 0
            return "{} = {};\n".format(
                ast[1][1],
                format_ast(table, ast[2], format_kernel))
        return "{} = {};\n".format(
            format_element(ast[1][1], ast[1][2]),
            format_ast(table, ast[2], format_kernel))
    elif ast[0] == 'call':
        return format_call(table, ast[1], ast[2])
    elif ast[0] == 'element':
        if len(ast[2]) == 0:
            assert len(ast[2]) == 0
            return "({}[0])".format(ast[1])
        else:
            return "({})".format(format_element(ast[1], ast[2]))
    elif ast[0] == 'const':
        return "%f"%(ast[1],)
    elif ast[0] == 'var':
        return ast[1]
    elif ast[0] == 'kernel':
        assert format_kernel is not None
        return format_kernel(ast[1])
    elif ast[0] == 'sync':
        return "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
    else:
        raise NotImplementedError


def format_struct_field(table, v):
    shape = table.vars[table.symbols[v]].shape[::-1]
    if len(shape) <= 1:
        return "float *{};\n".format(k)
    return "float (*{}){};\n".format(
        v,
        "".join("[%d]"%(s,) for s in shape[1:]))

HEADER_TEMPLATE = """#pragma once

struct {name}_state {{
{inputs}{outputs}
}};

void {name}(struct {name}_state *s);
"""

def format_c_header(name, table):
    return HEADER_TEMPLATE.format(
        name = name,
        inputs = "".join(format_struct_field(table,v)
                         for v in table.inputs),
        outputs = "".join(format_struct_field(table,v)
                         for v in table.outputs)
    )
