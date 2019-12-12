import json
from .utils import product


def get_shapes(table, ctx):
    vars = (
        ctx.input_arrays.keys()
        + ctx.output_arrays.keys()
        + ctx.intermediate_arrays.keys()
        + ctx.const_arrays.keys())

    return {v:table.vars[v].shape for v in vars}


def get_offsets(ctx, shapes):
    offsets = {}
    next_offset = 0

    for v in ctx.input_arrays:
        offsets[v] = next_offset
        next_offset += product(shapes[v])

    for v in ctx.output_arrays:
        offsets[v] = next_offset
        next_offset += product(shapes[v])

    for v in ctx.intermediate_arrays:
        if len(shapes[v]) == 0:
            continue
        offsets[v] = next_offset
        next_offset += product(shapes[v])

    for v in ctx.const_arrays:
        if len(shapes[v]) == 0:
            continue
        offsets[v] = next_offset
        next_offset += product(shapes[v])

    return offsets, next_offset


def get_iterators(ast):
    if isinstance(ast, list):
        for node in ast:
            for it in get_iterators(node):
                yield it
    if ast[0] == 'for':
        for it in get_iterators(ast[5]):
            yield it
        yield ast[1][1]
    elif ast[0] == 'if':
        for it in get_iterators(ast[2]):
            yield it
    elif ast[0] == 'ifelse':
        for it in get_iterators(ast[2]):
            yield it
        for it in get_iterators(ast[3]):
            yield it
    else:
        return

def format_iterators(iterators):
    for it in iterators:
        yield "var {}= 0;\n".format(it)


def format_offsets(offsets):
    for k, v in offsets.items():
        yield "var off{}={};\n".format(k,v)


def format_consts(offsets, shapes, values):
    for v, a in values.items():
        if len(shapes[v]) == 0:
            continue

        yield "new Float32Array(buffer,{},{}).set({});\n".format(offsets[v]*4,product(shape),json.dumps(list(a)))

def format_scalar_consts(shapes, values):
    for v, a in values.items():
        if len(shapes[v]) != 0:
            continue

        yield "var {}={};\n".format(v, json.dumps(a[0]))

def format_scalar_vars(shapes, vars):
    for v in vars:
        if len(shapes[v]) != 0:
            continue

        yield "var {}=0.0;\n".format(v)

def format_arrays(offsets, shapes, symbols):
    for k, v in symbols.items():
        if not isinstance(v, str):
            continue
        if v not in offsets:
            continue
        yield "{}: new Float32Array(buffer,{},{}),\n".format(
            k,
            offsets[v]*4,
            product(shapes[v]))


ASMJS_TEMPLATE = """function {name}(){{
function asm_module(stdlib, foreign, heap){{
"use asm";
var HEAP = new stdlib.Float32Array(heap);
{offsets}
var Infinity = stdlib.Infinity;
var NaN = stdlib.NaN;
var acos = stdlib.Math.acos;
var asin = stdlib.Math.asin;
var atan = stdlib.Math.atan;
var cos = stdlib.Math.cos;
var sin = stdlib.Math.sin;
var tan = stdlib.Math.tan;
var exp = stdlib.Math.exp;
var log = stdlib.Math.log;
var ceil = stdlib.Math.ceil;
var floor = stdlib.Math.floor;
var sqrt = stdlib.Math.sqrt;
var abs = stdlib.Math.abs;
var min = stdlib.Math.min;
var max = stdlib.Math.max;
var atan2 = stdlib.Math.atan2;
var pow = stdlib.Math.pow;
var imul = stdlib.Math.imul;
var fround = stdlib.Math.fround;
var E = stdlib.Math.E;
var LN10 = stdlib.Math.LN10;
var LN2 = stdlib.Math.LN2;
var LOG2E = stdlib.Math.LOG2E;
var LOG10E = stdlib.Math.LOG10E;
var PI = stdlib.Math.PI;
var SQRT1_2 = stdlib.Math.SQRT1_2;
var SQRT2 = stdlib.Math.SQRT2;

function calc(){{
{scalar_consts}
{scalar_vars}
{iterators}
{body}
}}

return {{calc:calc}};
}}

var buffer = new ArrayBuffer({size});
var mod = asm_module(window, null, buffer);
{consts}

var arrays = {{
{arrays}
}};

return {{arrays: arrays, calc: mod.calc}};
}}
"""

def roundup(size):
    size -= 1
    bits = 0

    while size:
        size >>= 1
        bits += 1

    return max(0x1000, 2**bits)


def format_asmjs(name, table, ctx, ast):
    shapes = get_shapes(table, ctx)
    offsets, size = get_offsets(ctx, shapes)

    iterators = set(get_iterators(ast))

    return ASMJS_TEMPLATE.format(
        name = name,
        offsets = "".join(format_offsets(offsets)),
        consts = "".join(format_consts(offsets, shapes, ctx.const_values)),
        scalar_consts = "".join(format_scalar_consts(shapes, ctx.const_values)),
        scalar_vars = "".join(format_scalar_vars(shapes, ctx.intermediate_arrays)),
        iterators = "".join(format_iterators(iterators)),
        size = roundup(size * 4),
        body = format_ast(shapes, offsets, ast),
        arrays = "".join(format_arrays(offsets, shapes, {k:v for k,v in table.symbols.items() if k in table.inputs or k in table.outputs})))


INT_FUN_TYPE = {
    '<': 'infix',
    '>': 'infix',
    '<=': 'infix',
    '>=': 'infix',
    '==': 'infix',
    '!=': 'infix',
    '&&': '&&',
    '||': '||',
    '+': 'infix',
    '-': 'infix',
    '*': 'fun2',
    '/': 'infix',
    '_': "prefix",
    '%': 'infix',
    'max': 'max',
    'min': 'min',
}

INT_FUN_FMT = {
    '_': '-',
    '*': 'imul',
}

def format_int_call(fun, args):
    t = INT_FUN_TYPE[fun]
    if t == '&&':
        return "({}?{}:0)".format(format_int_expr(args[0]), format_int_expr(args[1]))
    elif t == '||':
        return "({}?1:{})".format(format_int_expr(args[0]), format_int_expr(args[1]))
    elif t == 'max':
        return "((({a}|0)>({b}|0))?{a}:{b})|0".format(a=format_int_expr(args[0]), b=format_int_expr(args[1]))
    elif t == 'min':
        return "((({a}|0)<({b}|0))?{a}:{b})|0".format(a=format_int_expr(args[0]), b=format_int_expr(args[1]))
    elif t == 'infix':
        return "((({}|0){}({}|0))|0)".format(format_int_expr(args[0]), INT_FUN_FMT.get(fun,fun), format_int_expr(args[1]))
    elif t == 'prefix':
        return "(({}({}|0))|0)".format(INT_FUN_FMT.get(fun,fun), format_int_expr(args[0]))
    elif t == 'fun1':
        return "({}({}|0)|0)".format(INT_FUN_FMT.get(fun,fun), format_int_expr(args[0]))
    elif t == 'fun2':
        return "({}({}|0,{}|0)|0)".format(INT_FUN_FMT.get(fun,fun), format_int_expr(args[0]), format_int_expr(args[1]))

    raise NotImplementedError


def format_int_expr(ast):
    if ast[0] == 'var':
        return '{}'.format(ast[1])
    elif ast[0] == 'int':
        return "{}".format(json.dumps(ast[1]))
    elif ast[0] == 'call':
        return format_int_call(ast[1], ast[2])

    raise NotImplementedError


def format_indices(indices, shape):
    index = format_int_expr(indices[0])

    for i, s in zip(indices[1:], shape[:-1][::-1]):
        index = "(imul({}|0,{}|0)|0) + {}".format(index, s, format_int_expr(i))

    return index


def format_element(v, shape, indices):
    if len(shape) == 0:
        assert len(indices) == 0
        return "HEAP[off{} << 2 >> 2]".format(v)

    index = format_indices(indices, shape)

    return "HEAP[(off{} + {}) << 2 >> 2]".format(v, index)


FUN_TYPE = {
    '+': 'infix',
    '-': 'infix',
    '*': 'infix',
    '/': 'infix',
    '_': 'prefix',
    '**': 'fun2',
    'exp': 'fun1',
    'log': 'fun1',
    '==': 'compare',
    'max': 'fun2',
    'min': 'fun2',
}

FUN_FMT = {
    '_': '-',
    '**': 'pow',
}

def format_call(shapes, offsets, fun, args):
    t = FUN_TYPE[fun]
    if t == 'infix':
        return "({} {} {})".format(format_ast(shapes, offsets, args[0]), FUN_FMT.get(fun,fun), format_ast(shapes, offsets, args[1]))
    elif t == 'compare':
        return "(+(({} {} {})|0))".format(format_ast(shapes, offsets, args[0]), FUN_FMT.get(fun,fun), format_ast(shapes, offsets, args[1]))
    elif t == 'prefix':
        return "({} {})".format(FUN_FMT.get(fun,fun), format_ast(shapes, offsets, args[0]))
    elif t == 'fun1':
        return "({}({}))".format(FUN_FMT.get(fun, fun), format_ast(shapes, offsets, args[0]))
    elif t == 'fun2':
        return "({}({},{}))".format(FUN_FMT.get(fun, fun), format_ast(shapes, offsets, args[0]), format_ast(shapes, offsets, args[1]))

    raise NotImplementedError


def format_ast(shapes, offsets, ast):
    if isinstance(ast, list):
        return "".join(format_ast(shapes, offsets, node) for node in ast)
    if ast[0] == 'for':
        return "for({v} = {init}; {cond}; {v} = ({v} + {inc})|0){{\n{body}}}\n".format(
            v = format_int_expr(ast[1]),
            init = format_int_expr(ast[2]),
            inc = format_int_expr(ast[3]),
            cond = format_int_expr(ast[4]),
            body = format_ast(shapes, offsets, ast[5]))
    elif ast[0] == 'if':
        return "if({cond}){{\n{then}}}\n".format(
            cond = format_int_expr(ast[1]),
            then = format_ast(shapes, offsets, ast[2]))
    elif ast[0] == 'ifelse':
        return "if({cond}){{\n{then}}}\nelse{{\n{else_}}}".format(
            cond = format_int_expr(ast[1]),
            then = format_ast(shapes, offsets, ast[2]),
            else_ = format_ast(shapes, offsets, ast[3]))
    elif ast[0] == 'assign':
        assert ast[1][0] == 'element'
        if ast[1][1] not in offsets:
            assert len(ast[1][2]) == 0
            return "{} = {};\n".format(
                ast[1][1],
                format_ast(shapes, offsets, ast[2]))
        return "{} = fround({});\n".format(
            format_element(ast[1][1], shapes[ast[1][1]], ast[1][2]),
            format_ast(shapes, offsets, ast[2]))
    elif ast[0] == 'call':
        return format_call(shapes, offsets, ast[1], ast[2])
    elif ast[0] == 'element':
        if ast[1] not in offsets:
            assert len(ast[2]) == 0
            return ast[1]
        else:
            return "(+{})".format(format_element(ast[1], shapes[ast[1]], ast[2]))
    elif ast[0] == 'const':
        s = json.dumps(ast[1])
        return "{}".format(s if "." in s else s.replace("e", ".e"))
    elif ast[0] == 'var':
        return ast[1]
    else:
        raise NotImplementedError
