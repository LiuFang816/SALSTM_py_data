from itertools import izip_longest
from . import ast, symbol
from .builtins import builtins


def transform_integers(numbers):
    assert all(isinstance(n,ast.Integer) for n in numbers.body)
    return tuple(int(n.n) for n in numbers.body)

def transform_ranks(numbers):
    assert all(isinstance(n,ast.Integer) or isinstance(n,ast.Unknown) for n in numbers.body)
    return tuple((int(n.n) if isinstance(n,ast.Integer) else None) for n in numbers.body)

def split_shape(sy, ry):
    if ry is None:
        ry = len(sy[0])
    assert len(sy[0]) >= ry
    return ((sy[0][:ry],sy[0][ry:]) + sy[1:])

def merge_shape(s):
    return (s[0]+s[1],) + s[2:]

def acc1(l):
    a = 0
    for x in l:
        a += x
        yield a

def recover_monad_rank(y):
    return tuple(acc1(map(len,y[:-1])))

def acc2(l):
    ax, ay = 0, 0
    for x,y in l:
        ax += x
        ay += y
        yield (ax,ay)

def recover_dyad_rank(x, y):
    return tuple(acc2(tuple(izip_longest(map(len,x), map(len,y), fillvalue=0))[:-1]))

def agree(sx, sy, rx, ry):
    sx = split_shape(sx, rx)
    sy = split_shape(sy, ry)
    assert all(a == b for a, b in zip(reversed(sx[1]), reversed(sy[1])))
    sz = tuple((a if len(a) >= len(b) else b) for a, b in izip_longest(sx[1:], sy[1:], fillvalue=()))
    return sx,sy,sz


def check_monad(table, args, op, y):
    if isinstance(op, ast.Exprs) and len(op.body) == 3 and isinstance(op.body[1], ast.Rank):
        rank, = transform_ranks(op.body[2])
        sy = split_shape(y[1], rank)
        z = check_monad(table, args, op.body[0], (y[0],sy))
        assert z[1][1:] == sy[1:]
        return (z[0], merge_shape(z[1]))
    elif isinstance(op, ast.Exprs):
        assert isinstance(op.body[0], ast.Name)
        monad = table.symbols[op.body[0].s](table, op.body[1:])

    elif isinstance(op, ast.Name):
        monad = table.symbols[op.s]
    else:
        raise NotImplementedError

    sy = split_shape(y[1], monad.rank)

    if isinstance(monad, symbol.Monad):
        z = check_expr(table, {'y': (y[0],sy)}, monad.expr)
        return (z[0],merge_shape(z[1]))

    shape = (monad.get_shape(sy[0]),) + sy[1:]
    v = table.vars.add(
        symbol.ApplyMonad(
            op.position,
            shape = sum(shape,()),
            op = monad,
            rank = recover_monad_rank(y[1]),
            y = y[0]))
    return (v,merge_shape(shape))


def check_dyad(table, args, op, x, y):
    if isinstance(op, ast.Exprs) and len(op.body) == 3 and isinstance(op.body[1], ast.Rank):
        rx, ry = transform_ranks(op.body[2])
        sx, sy, sz = agree(x[1], y[1], rx, ry)
        z = check_dyad(table, args, op.body[0], (x[0],sx), (y[0],sy))
        assert z[1][1:] == sz
        return (z[0], merge_shape(z[1]))

    elif isinstance(op, ast.Exprs):
        assert isinstance(op.body[0], ast.Name)
        dyad = table.symbols[op.body[0].s](table, op.body[1:])

    elif isinstance(op, ast.Name):
        dyad = table.symbols[op.s]
    else:
        raise NotImplementedError

    rx, ry = dyad.rank
    sx, sy, sz = agree(x[1],y[1],rx,ry)

    if isinstance(dyad, symbol.Dyad):
        z = check_expr(table, {'x': (x[0],sx),'y': (y[0],sy)}, dyad.expr)
        return (z[0],merge_shape(z[1]))

    shape = (dyad.get_shape(sx[0], sy[0]),) + sz
    v = table.vars.add(
        symbol.ApplyDyad(
            op.position,
            shape = sum(shape,()),
            op = dyad,
            rank = recover_dyad_rank(x[1],y[1]),
            x = x[0],
            y = y[0]))
    return (v,merge_shape(shape))


def check_expr(table, args, expr):
    if isinstance(expr, ast.Name):
        v = table.symbols[expr.s]
        shape = table.vars[v].shape
        return (v, (shape,))
    elif isinstance(expr, ast.Arg):
        return args[expr.s]
    elif isinstance(expr, ast.Numbers):
        shape = () if len(expr.body) == 1 else (len(expr,body),)
        v = table.vars.add(
            symbol.Literal(
                expr.position,
                shape = shape,
                body = expr.body))
        return (v, (shape,))
    elif isinstance(expr, ast.Exprs):
        if len(expr.body) == 1:
            return check_expr(table, args, expr.body[0])
        elif len(expr.body) == 2:
            # f y
            y = check_expr(table, args, expr.body[1])
            return check_monad(table, args, expr.body[0], y)
        elif len(expr.body) == 3:
            # x f y
            x = check_expr(table, args, expr.body[0])
            y = check_expr(table, args, expr.body[2])
            return check_dyad(table, args, expr.body[1], x, y)

    raise NotImplementedError


def check_arg_shape(body):
    if len(body) == 1:
        return ()
    elif len(body) == 2:
        return transform_integers(body[1])[::-1]
    else:
        raise NotImplementedError

def check_arg_in(table, stmt):
    shape = check_arg_shape(stmt.value.body)

    table.symbols[stmt.name] = table.vars.add(
        symbol.Input(stmt.position, shape=shape))

    table.inputs[stmt.name] = table.symbols[stmt.name]

def check_arg_out(table, stmt):
    table.outputs[stmt.name] = symbol.Output(
        stmt.position, shape=check_arg_shape(stmt.value.body))

def check_grad(table, stmt):
    assert len(stmt.value.body) == 3
    assert isinstance(stmt.value.body[1], ast.Name)
    assert isinstance(stmt.value.body[2], ast.Name)

    v, w = stmt.value.body[1].s, stmt.value.body[2].s
    table.symbols[stmt.name] = table.get_grad(
        table.symbols[v], table.symbols[w])


DECLARES = {
    'in': check_arg_in,
    'out': check_arg_out,
    'grad': check_grad,
}


def check_stmt(table, stmt):
    if isinstance(stmt, ast.Declare):
        fun = stmt.value.body[0]
        assert isinstance(fun, ast.Name)
        DECLARES[fun.s](table, stmt)

    elif isinstance(stmt, ast.Assign):
        v, (shape,) = check_expr(table, {}, stmt.value)
        table.symbols[stmt.name] = v

    elif isinstance(stmt, ast.Op):
        rank = transform_ranks(stmt.rank)

        if len(rank) == 1:
            table.symbols[stmt.name] = symbol.Monad(
                stmt.position,
                rank = rank[0],
                expr = stmt.value)
        elif len(rank) == 2:
            table.symbols[stmt.name] = symbol.Dyad(
                stmt.position,
                rank = rank,
                expr = stmt.value)
        else:
            assert False, "rank of op must be either 1 or 2"

    else:
        raise NotImplementedError


def check_stmts(stmts):
    table = symbol.Table(builtins)

    for stmt in stmts:
        check_stmt(table, stmt)

    for k, v in table.outputs.iteritems():
        assert table.vars[table.symbols[k]].shape == v.shape
        assert not isinstance(table.vars[table.symbols[k]], symbol.Input)

    return table
