import islpy as isl
from . import symbol
from .utils import Counter


class Arrays(dict):

    def __init__(self, ctx):
        self.ctx = ctx.isl_context


    def add(self, name, shape):
        assert name not in self
        space = (
            isl.Space.set_alloc(self.ctx, 0, len(shape))
            .set_tuple_name(isl.dim_type.set, name))

        bs = isl.BasicSet.nat_universe(space)

        for i, s in enumerate(shape):
            bs = bs.add_constraint(
                isl.Constraint.alloc_inequality(space)
                .set_coefficient_val(
                    isl.dim_type.set, i, isl.Val.int_from_si(self.ctx, -1))
                .set_constant_val(
                    isl.Val.int_from_si(self.ctx, s-1)))

        self[name] = isl.Set.from_basic_set(bs)


def rename_bm(bm, name):
    return bm.set_tuple_name(isl.dim_type.in_, name)


def rename_expr(expr, name):
    if expr[0] == 'var':
        return ('var', rename_bm(expr[1],name))
    elif expr[0] == 'const':
        return expr
    elif expr[0] == 'call':
        return (expr[0],expr[1], tuple(rename_expr(e,name) for e in expr[2]))
    else:
        raise NotImplementedError


def rename_stmt(stmt, name):
    return (
        rename_bm(stmt[0], name),
        rename_expr(stmt[1], name))


def get_uses(expr):
    if expr[0] == 'var':
        yield expr[1]
    elif expr[0] == 'const':
        return
    elif expr[0] == 'call':
        for e in expr[2]:
            for u in get_uses(e):
                yield u
    else:
        raise NotImplementedError


def subtract_expr(expr, domain):
    if expr[0] == 'var':
        return ('var', expr[1].subtract_domain(domain))
    elif expr[0] == 'const':
        return expr
    elif expr[0] == 'call':
        return (expr[0],expr[1], tuple(subtract_expr(e,domain) for e in expr[2]))
    else:
        raise NotImplementedError


class Statements(dict):

    def __init__(self, ctx):
        self.ctx = ctx

    def add(self, stmt):
        name = self.ctx.stmt_name.next()
        self[name] = rename_stmt(stmt, name)

    def subtract(self, domain):
        name = domain.get_tuple_name()
        stmt = self[name]
        dest = stmt[0].subtract_domain(domain)
        if dest.is_empty():
            del self[name]
        else:
            self[name] = (dest, subtract_expr(stmt[1], domain))


    def get_assign_map(self):
        return self.union_maps(s[0] for s in self.values())

    def get_use_map(self):
        return self.union_maps(sum((tuple(get_uses(s[1])) for s in self.values()), ()))

    def union_maps(self, maps):
        umap = isl.UnionMap("{}", self.ctx.isl_context)
        for m in maps:
            umap = umap.union(isl.UnionMap.from_map(m))
        return umap


class Context(object):


    def __init__(self, isl_context):
        self.isl_context = isl_context
        self.stmt_name = Counter("S%d")
        self.const_values = {}

        self.input_arrays = Arrays(self)
        self.output_arrays = Arrays(self)
        self.intermediate_arrays = Arrays(self)
        self.const_arrays = Arrays(self)
        self.def_stmts = Statements(self)
        self.init_stmts = Statements(self)
        self.update_stmts = Statements(self)
        self.fini_stmts = Statements(self)


    def new_element_access(self, name):
        space = (
            isl.Space.alloc(self.isl_context, 0, 0, 0)
            .set_tuple_name(isl.dim_type.in_, "S")
            .set_tuple_name(isl.dim_type.out, name))

        return isl.Map.nat_universe(space)


    def increase_dim(self, v, type):
        name = v.get_tuple_name(type)
        return v.add_dims(type, 1).set_tuple_name(type, name)


    def append_dim1(self, v, bound):
        n_in_ = v.dim(isl.dim_type.in_)

        v = self.increase_dim(v, isl.dim_type.in_)

        space = v.get_space()

        return (
            v.add_constraint(
                isl.Constraint.alloc_inequality(v.get_space())
                .set_coefficient_val(
                    isl.dim_type.in_, n_in_, isl.Val.int_from_si(self.isl_context, -1))
                .set_constant_val(
                    isl.Val.int_from_si(self.isl_context, bound-1)))
            .add_constraint(
                isl.Constraint.alloc_inequality(v.get_space())
                .set_coefficient_val(
                    isl.dim_type.in_, n_in_, isl.Val.int_from_si(self.isl_context, 1))))


    def append_dim2(self, v, bound):
        n_in_ = v.dim(isl.dim_type.in_)
        n_out = v.dim(isl.dim_type.out)

        v = self.append_dim1(v, bound)
        v = self.increase_dim(v, isl.dim_type.out)

        space = v.get_space()

        return (
            v.add_constraint(
                isl.Constraint.alloc_equality(space)
                .set_coefficient_val(
                    isl.dim_type.in_, n_in_, isl.Val.int_from_si(self.isl_context, -1))
                .set_coefficient_val(
                    isl.dim_type.out, n_out, isl.Val.int_from_si(self.isl_context, 1))))


    def get_use_map(self):
        return self.def_stmts.get_use_map().union(self.update_stmts.get_use_map().subtract(self.update_stmts.get_assign_map()))


def compile_monad(ctx, op, ranks, sy, y, z):
    if not ranks:
        ry = op.rank
        inner = op.compile
    else:
        ry = ranks[-1]
        def inner(ctx, sy, y, z):
            compile_monad(ctx, op, ranks[:-1], sy, y, z)

    if ry is None:
        ry = len(sy)

    for s in sy[ry:][::-1]:
        y = ctx.append_dim2(y, s)
        z = ctx.append_dim2(z, s)

    inner(ctx, sy[:ry], y, z)


def compile_acc_monad(ctx, op, ranks, sy, y, z, acc, g):
    if not ranks:
        ry = op.rank
        inner = op.compile_acc_y
    else:
        ry = ranks[-1]
        def inner(ctx, sy, y, z, acc, g):
            compile_acc_monad(ctx, op, ranks[:-1], sy, y, z, acc, g)

    if ry is None:
        ry = len(sy)

    for s in sy[ry:][::-1]:
        y = ctx.append_dim2(y, s)
        z = ctx.append_dim2(z, s)
        acc = ctx.append_dim2(acc, s)
        g = ctx.append_dim2(g, s)

    inner(ctx, sy[:ry], y, z, acc, g)


def compile_dyad(ctx, op, ranks, sx, x, sy, y, z):
    if not ranks:
        rx, ry = op.rank
        if rx is None:
            rx = len(sx)
        if ry is None:
            ry = len(sy)
        inner = op.compile
    else:
        rx, ry = ranks[-1]
        def inner(ctx, sx, x, sy, y, z):
            compile_dyad(ctx, op, ranks[:-1], sx, x, sy, y, z)

    sxo, syo = sx[rx:], sy[ry:]

    if len(sxo) <= len(syo):
        d = len(syo) - len(sxo)

        for s in sxo[::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)

        for s in syo[:d][::-1]:
            x = ctx.append_dim1(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)
    else:
        d = len(sxo) - len(syo)

        for s in syo[::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)

        for s in sxo[:d][::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim1(y, s)
            z = ctx.append_dim2(z, s)

    inner(ctx, sx[:rx], x, sy[:ry], y, z)


def compile_dyad(ctx, op, ranks, sx, x, sy, y, z):
    if not ranks:
        rx, ry = op.rank
        inner = op.compile
    else:
        rx, ry = ranks[-1]
        def inner(ctx, sx, x, sy, y, z):
            compile_dyad(ctx, op, ranks[:-1], sx, x, sy, y, z)

    if rx is None:
        rx = len(sx)
    if ry is None:
        ry = len(sy)

    sxo, syo = sx[rx:], sy[ry:]

    if len(sxo) <= len(syo):
        d = len(syo) - len(sxo)

        for s in sxo[::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)

        for s in syo[:d][::-1]:
            x = ctx.append_dim1(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)
    else:
        d = len(sxo) - len(syo)

        for s in syo[::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)

        for s in sxo[:d][::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim1(y, s)
            z = ctx.append_dim2(z, s)

    inner(ctx, sx[:rx], x, sy[:ry], y, z)


def compile_acc_dyad_x(ctx, op, ranks, sx, x, sy, y, z, acc, g):
    if not ranks:
        rx, ry = op.rank
        inner = op.compile_acc_x
    else:
        rx, ry = ranks[-1]
        def inner(ctx, sx, x, sy, y, z, acc, g):
            compile_acc_dyad_x(ctx, op, ranks[:-1], sx, x, sy, y, z, acc, g)

    if rx is None:
        rx = len(sx)
    if ry is None:
        ry = len(sy)

    sxo, syo = sx[rx:], sy[ry:]

    if len(sxo) <= len(syo):
        d = len(syo) - len(sxo)

        for s in sxo[::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)
            acc = ctx.append_dim2(acc, s)
            g = ctx.append_dim2(g, s)

        for s in syo[:d][::-1]:
            x = ctx.append_dim1(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)
            acc = ctx.append_dim2(acc, s)
            g = ctx.append_dim1(g, s)

    else:
        d = len(sxo) - len(syo)

        for s in syo[::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)
            acc = ctx.append_dim2(acc, s)
            g = ctx.append_dim2(g, s)

        for s in sxo[:d][::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim1(y, s)
            z = ctx.append_dim2(z, s)
            acc = ctx.append_dim2(acc, s)
            g = ctx.append_dim2(g, s)

    inner(ctx, sx[:rx], x, sy[:ry], y, z, acc, g)


def compile_acc_dyad_y(ctx, op, ranks, sx, x, sy, y, z, acc, g):
    if not ranks:
        rx, ry = op.rank
        inner = op.compile_acc_y
    else:
        rx, ry = ranks[-1]
        def inner(ctx, sx, x, sy, y, z, acc, g):
            compile_acc_dyad_y(ctx, op, ranks[:-1], sx, x, sy, y, z, acc, g)

    if rx is None:
        rx = len(sx)
    if ry is None:
        ry = len(sy)

    sxo, syo = sx[rx:], sy[ry:]

    if len(sxo) <= len(syo):
        d = len(syo) - len(sxo)

        for s in sxo[::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)
            acc = ctx.append_dim2(acc, s)
            g = ctx.append_dim2(g, s)

        for s in syo[:d][::-1]:
            x = ctx.append_dim1(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)
            acc = ctx.append_dim2(acc, s)
            g = ctx.append_dim2(g, s)

    else:
        d = len(sxo) - len(syo)

        for s in syo[::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)
            acc = ctx.append_dim2(acc, s)
            g = ctx.append_dim2(g, s)

        for s in sxo[:d][::-1]:
            x = ctx.append_dim2(x, s)
            y = ctx.append_dim1(y, s)
            z = ctx.append_dim2(z, s)
            acc = ctx.append_dim2(acc, s)
            g = ctx.append_dim1(g, s)

    inner(ctx, sx[:rx], x, sy[:ry], y, z, acc, g)


def add_acc_init(ctx, name, shape):
    v = ctx.new_element_access(name)

    for s in shape[::-1]:
        v = ctx.append_dim2(v, s)

    ctx.init_stmts.add((v, ('const', 0.0)))


def add_acc_fini(ctx, name, shape):
    v = ctx.new_element_access(name)

    for s in shape[::-1]:
        v = ctx.append_dim2(v, s)

    ctx.fini_stmts.add((v, ('var', v)))


def compile(table, isl_context):
    ctx = Context(isl_context)

    for k in table.inputs:
        name = table.symbols[k]
        v = table.vars[name]
        ctx.input_arrays.add(name, v.shape[::-1])

    for k in table.outputs:
        name = table.symbols[k]
        v = table.vars[name]
        ctx.output_arrays.add(name, v.shape[::-1])

    for name in table.vars:
        v = table.vars[name]

        if isinstance(v, symbol.Input):
            continue

        if isinstance(v, symbol.Literal):
            ctx.const_arrays.add(name, v.shape[::-1])
            ctx.const_values[name] = tuple(float(n.n) for n in v.body)
            continue
        elif isinstance(v, symbol.Constant):
            ctx.const_arrays.add(name, v.shape[::-1])
            ctx.const_values[name] = v.value
            continue

        if name not in ctx.output_arrays:
            ctx.intermediate_arrays.add(name, v.shape[::-1])

        if isinstance(v, symbol.ApplyMonad):
            compile_monad(
                ctx,
                v.op,
                v.rank,
                table.vars[v.y].shape,
                ctx.new_element_access(v.y),
                ctx.new_element_access(name))
        elif isinstance(v, symbol.ApplyDyad):
            compile_dyad(
                ctx,
                v.op,
                v.rank,
                table.vars[v.x].shape,
                ctx.new_element_access(v.x),
                table.vars[v.y].shape,
                ctx.new_element_access(v.y),
                ctx.new_element_access(name))
        elif isinstance(v, symbol.AccMonad):
            u = table.vars[v.v]
            add_acc_init(ctx, name, v.shape)
            compile_acc_monad(
                ctx,
                u.op,
                u.rank,
                table.vars[u.y].shape,
                ctx.new_element_access(u.y),
                ctx.new_element_access(v.v),
                ctx.new_element_access(v.acc),
                ctx.new_element_access(name))
            add_acc_fini(ctx, name, v.shape)
        elif isinstance(v, symbol.AccDyad):
            u = table.vars[v.v]
            add_acc_init(ctx, name, v.shape)
            if v.operand == 'x':
                compile_acc_dyad_x(
                    ctx,
                    u.op,
                    u.rank,
                    table.vars[u.x].shape,
                    ctx.new_element_access(u.x),
                    table.vars[u.y].shape,
                    ctx.new_element_access(u.y),
                    ctx.new_element_access(v.v),
                    ctx.new_element_access(v.acc),
                    ctx.new_element_access(name))
            elif v.operand == 'y':
                compile_acc_dyad_y(
                    ctx,
                    u.op,
                    u.rank,
                    table.vars[u.x].shape,
                    ctx.new_element_access(u.x),
                    table.vars[u.y].shape,
                    ctx.new_element_access(u.y),
                    ctx.new_element_access(v.v),
                    ctx.new_element_access(v.acc),
                    ctx.new_element_access(name))
            else:
                raise NotImplementedError
            add_acc_fini(ctx, name, v.shape)
        else:
            raise NotImplementedError

    return ctx
