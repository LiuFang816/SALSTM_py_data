import islpy as isl
from .utils import product


class Plus(object):
    rank = (0,0)
    zero = 0.0

    def get_shape(self, sx, sy):
        return ()

    def interp(self, sx, vx, sy, vy, vz):
        vz[0] = vx[0] + vy[0]

    def interp_acc_x(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += acc[0]

    def interp_acc_y(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += acc[0]

    def compile(self, ctx, sx, x, sy, y, z):
        ctx.def_stmts.add(
            (z,
             ('call', '+',
              (('var',x),
               ('var',y)))))

    def compile_update(self, ctx, sx, x, sy, y, z):
        ctx.update_stmts.add(
            (z,
             ('call', '+',
              (('var',x),
               ('var',y)))))

    def compile_acc_x(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('var', acc)
              ))
            ))

    def compile_acc_y(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('var', acc)
              ))
            ))


class Minus(object):
    rank = (0,0)

    def get_shape(self, sx, sy):
        return ()

    def interp(self, sx, vx, sy, vy, vz):
        vz[0] = vx[0] - vy[0]

    def interp_acc_x(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += acc[0]

    def interp_acc_y(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += -acc[0]

    def compile(self, ctx, sx, x, sy, y, z):
        ctx.def_stmts.add(
            (z,
             ('call', '-',
              (('var',x),
               ('var',y)))))

    def compile_acc_x(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('var', acc)
              ))
            ))

    def compile_acc_y(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('call', '_', (('var', acc),))
              ))
            ))


class Multiply(object):
    rank = (0,0)

    def get_shape(self, sx, sy):
        return ()

    def interp(self, sx, vx, sy, vy, vz):
        vz[0] = vx[0] * vy[0]

    def interp_acc_x(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += acc[0] * vy[0]

    def interp_acc_y(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += acc[0] * vx[0]

    def compile(self, ctx, sx, x, sy, y, z):
        ctx.def_stmts.add(
            (z,
             ('call', '*',
              (('var',x),
               ('var',y)))))

    def compile_acc_x(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('call', '*', (('var', acc), ('var', y)))
              ))
            ))

    def compile_acc_y(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('call', '*', (('var', acc), ('var', x)))
              ))
            ))


class Divide(object):
    rank = (0,0)

    def get_shape(self, sx, sy):
        return ()

    def interp(self, sx, vx, sy, vy, vz):
        vz[0] = vx[0] / vy[0]

    def interp_acc_x(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += acc[0] / vy[0]

    def interp_acc_y(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] -= acc[0] * vx[0] / vy[0] / vy[0]

    def compile(self, ctx, sx, x, sy, y, z):
        ctx.def_stmts.add(
            (z,
             ('call', '/',
              (('var',x),
               ('var',y)))))

    def compile_acc_x(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('call', '/', (('var', acc), ('var', y)))
              ))
            ))

    def compile_acc_y(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('call', '_',
                 (('call', '/',
                   (('call', '*', (('var', acc), ('var', x))),
                    ('call', '*', (('var', y), ('var', y))))),
                 )
                ))
             )))


class Power(object):
    rank = (0,0)

    def get_shape(self, sx, sy):
        return ()

    def interp(self, sx, vx, sy, vy, vz):
        vz[0] = vx[0] ** vy[0]

    def interp_acc_x(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += acc[0] * vz[0] * vy[0] / vx[0]

    def compile(self, ctx, sx, x, sy, y, z):
        ctx.def_stmts.add(
            (z,
             ('call', '**',
              (('var',x),
               ('var',y)))))

    def compile_acc_x(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('call', '/',
                 (('call', '*', (('call', '*', (('var', acc), ('var', z))), ('var', y))),
                  ('var', x))
                )
              ))
            ))


class Exp(object):
    rank = 0

    def get_shape(self, sy):
        return ()

    def interp(self, sy, vy, vz):
        from math import exp
        vz[0] = exp(vy[0])

    def interp_acc_y(self, sy, vy, vz, acc, vg):
        vg[0] += acc[0] * vz[0]

    def compile(self, ctx, sy, y, z):
        ctx.def_stmts.add(
            (z,
             ('call', 'exp',
              (('var',y),))))

    def compile_acc_y(self, ctx, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                (('call', '*', (('var', acc), ('var', z)))))
             )))


class Log(object):
    rank = 0

    def get_shape(self, sy):
        return ()

    def interp(self, sy, vy, vz):
        from math import log
        vz[0] = log(vy[0])

    def interp_acc_y(self, sy, vy, vz, acc, vg):
        vg[0] += acc[0] / vz[0]

    def compile(self, ctx, sy, y, z):
        ctx.def_stmts.add(
            (z,
             ('call', 'log',
              (('var',y),))))

    def compile_acc_y(self, ctx, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                (('call', '/', (('var', acc), ('var', y)))))
             )))


class Reduce(object):
    rank = None

    def __init__(self, table, args):
        n, name = args
        n, = n.body
        self.n = int(n.n)
        assert self.n >= 1
        op = table.symbols[name.s]
        assert op.rank == (0,0)
        assert op.get_shape((),()) == ()
        self.op = op

    def get_shape(self, sy):
        n = self.n
        assert len(sy) >= n
        return sy[:-n]

    def interp(self, sy, vy, vz):
        n = self.n
        p1, p2 = product(sy[:-n]), product(sy[-n:])

        for i in xrange(p1):
            vz[i] = self.op.zero

        for i in xrange(p2):
            v = vy.subview(i, p2)
            for j in xrange(p1):
                self.op.interp((), vz.subview(j,p1), (), v.subview(j,p1), vz.subview(j,p1))

    def interp_acc_y(self, sy, vy, vz, acc, vg):
        n = self.n
        p1, p2 = product(sy[:-n]), product(sy[-n:])

        for i in xrange(p2):
            v = vy.subview(i, p2)
            g = vg.subview(i, p2)

            for j in xrange(p1):
                self.op.interp_acc_y((), vz.subview(j, p1), (), v.subview(j, p1), vz.subview(j, p1), acc.subview(j, p1), g.subview(j, p1))

    def compile(self, ctx, sy, y, z):
        n = self.n

        zi = z
        for s in sy[:-n][::-1]:
            zi = ctx.append_dim2(zi, s)

        ctx.init_stmts.add((zi, ('const', self.op.zero)))

        yu, zu = y, z

        for s in sy[-n:][::-1]:
            yu = ctx.append_dim2(yu, s)
            zu = ctx.append_dim1(zu, s)

        for s in sy[:-n][::-1]:
            yu = ctx.append_dim2(yu, s)
            zu = ctx.append_dim2(zu, s)

        self.op.compile_update(ctx, (), zu, (), yu, zu)
        ctx.fini_stmts.add((zi, ('var', zi)))

    def compile_acc_y(self, ctx, sy, y, z, acc, g):
        n = self.n

        for s in sy[-n:][::-1]:
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim1(z, s)
            acc = ctx.append_dim1(acc, s)
            g = ctx.append_dim2(g, s)

        for s in sy[:-n][::-1]:
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)
            acc = ctx.append_dim2(acc, s)
            g = ctx.append_dim2(g, s)

        self.op.compile_acc_y(ctx, (), z, (), y, z, acc, g)


class Min(object):
    rank = (0,0)
    zero = float("inf")

    def get_shape(self, sx, sy):
        return ()

    def interp(self, sx, vx, sy, vy, vz):
        vz[0] = min(vx[0], vy[0])

    def interp_acc_x(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += (vz[0]==vx[0]) * acc[0]

    def interp_acc_y(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += (vz[0]==vy[0]) * acc[0]

    def compile(self, ctx, sx, x, sy, y, z):
        ctx.def_stmts.add(
            (z,
             ('call', 'min',
              (('var',x),
               ('var',y)))))

    def compile_update(self, ctx, sx, x, sy, y, z):
        ctx.update_stmts.add(
            (z,
             ('call', 'min',
              (('var',x),
               ('var',y)))))

    def compile_acc_x(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('call', '*',
                 (('var', acc),
                  ('call', '==', (('var', z),('var', x)))
                 ))
              ))
            ))

    def compile_acc_y(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('call', '*',
                 (('var', acc),
                  ('call', '==', (('var', z),('var', y)))
                 ))
              ))
            ))


class Max(object):
    rank = (0,0)
    zero = float("-inf")

    def get_shape(self, sx, sy):
        return ()

    def interp(self, sx, vx, sy, vy, vz):
        vz[0] = max(vx[0], vy[0])

    def interp_acc_x(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += (vz[0]==vx[0]) * acc[0]

    def interp_acc_y(self, sx, vx, sy, vy, vz, acc, vg):
        vg[0] += (vz[0]==vy[0]) * acc[0]

    def compile(self, ctx, sx, x, sy, y, z):
        ctx.def_stmts.add(
            (z,
             ('call', 'max',
              (('var',x),
               ('var',y)))))

    def compile_update(self, ctx, sx, x, sy, y, z):
        ctx.update_stmts.add(
            (z,
             ('call', 'max',
              (('var',x),
               ('var',y)))))

    def compile_acc_x(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('call', '*',
                 (('var', acc),
                  ('call', '==', (('var', z),('var', x)))
                 ))
              ))
            ))

    def compile_acc_y(self, ctx, sx, x, sy, y, z, acc, g):
        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('call', '*',
                 (('var', acc),
                  ('call', '==', (('var', z),('var', y)))
                 ))
              ))
            ))


class Oblique(object):
    rank = None

    def __init__(self, table, args):
        n, name, = args
        n, = n.body
        self.n = int(n.n)
        assert self.n >= 1
        op = table.symbols[name.s]
        assert op.rank == (0,0)
        assert op.get_shape((),()) == ()
        self.op = op

    def get_shape(self, sy):
        n = self.n
        assert len(sy) >= 2*n
        return sy[:-2*n] + tuple(a+b-1 for a,b in zip(sy[-2*n:-n],sy[-n:]))

    def compile(self, ctx, sy, y, z):
        n = self.n
        sz = tuple(a+b-1 for a,b in zip(sy[-2*n:-n],sy[-n:]))

        zi = z
        for s in sz[::-1]:
            zi = ctx.append_dim2(zi, s)

        for s in sy[:-2*n][::-1]:
            zi = ctx.append_dim2(zi, s)

        ctx.init_stmts.add((zi, ('const', self.op.zero)))

        n_in_ = z.dim(isl.dim_type.in_)
        n_out = z.dim(isl.dim_type.out)

        yu = y
        zu = z

        for s in sy[-2*n:][::-1]:
            yu = ctx.append_dim2(yu, s)
            zu = ctx.append_dim1(zu, s)

        for i, s in enumerate(sy[-2*n:-n][::-1]):
            zu = ctx.increase_dim(zu, isl.dim_type.out)
            space = zu.get_space()
            zu = (
                zu.add_constraint(
                    isl.Constraint.alloc_equality(space)
                    .set_coefficient_val(
                        isl.dim_type.in_, n_in_+i, isl.Val.int_from_si(ctx.isl_context, -1))
                    .set_coefficient_val(
                        isl.dim_type.in_, n_in_+n+i, isl.Val.int_from_si(ctx.isl_context, 1))
                    .set_constant_val(
                        isl.Val.int_from_si(ctx.isl_context, -s+1))
                    .set_coefficient_val(
                        isl.dim_type.out, n_out+i, isl.Val.int_from_si(ctx.isl_context, 1))
                )
            )

        for s in sy[:-2*n][::-1]:
            yu = ctx.append_dim2(yu, s)
            zu = ctx.append_dim2(zu, s)

        self.op.compile_update(ctx, (), zu, (), yu, zu)
        ctx.fini_stmts.add((zi, ('var', zi)))

    def compile_acc_y(self, ctx, sy, y, z, acc, g):
        n = self.n
        n_in_ = z.dim(isl.dim_type.in_)
        n_out = z.dim(isl.dim_type.out)

        for s in sy[-2*n:][::-1]:
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim1(z, s)
            acc = ctx.append_dim1(acc, s)
            g = ctx.append_dim2(g, s)


        for i, s in enumerate(sy[-2*n:-n][::-1]):
            z = ctx.increase_dim(z, isl.dim_type.out)
            acc = ctx.increase_dim(acc, isl.dim_type.out)
            z = (
                z.add_constraint(
                    isl.Constraint.alloc_equality(z.get_space())
                    .set_coefficient_val(
                        isl.dim_type.in_, n_in_+i, isl.Val.int_from_si(ctx.isl_context, -1))
                    .set_coefficient_val(
                        isl.dim_type.in_, n_in_+n+i, isl.Val.int_from_si(ctx.isl_context, 1))
                    .set_constant_val(
                        isl.Val.int_from_si(ctx.isl_context, -s+1))
                    .set_coefficient_val(
                        isl.dim_type.out, n_out+i, isl.Val.int_from_si(ctx.isl_context, 1))
                )
            )
            acc = (
                acc.add_constraint(
                    isl.Constraint.alloc_equality(acc.get_space())
                    .set_coefficient_val(
                        isl.dim_type.in_, n_in_+i, isl.Val.int_from_si(ctx.isl_context, -1))
                    .set_coefficient_val(
                        isl.dim_type.in_, n_in_+n+i, isl.Val.int_from_si(ctx.isl_context, 1))
                    .set_constant_val(
                        isl.Val.int_from_si(ctx.isl_context, -s+1))
                    .set_coefficient_val(
                        isl.dim_type.out, n_out+i, isl.Val.int_from_si(ctx.isl_context, 1))
                )
            )

        for s in sy[:-2*n][::-1]:
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)
            acc = ctx.append_dim2(acc, s)
            g = ctx.append_dim2(g, s)

        self.op.compile_acc_y(ctx, (), z, (), y, z, acc, g)


class Trim(object):
    rank = None

    def __init__(self, table, args):
        trims, = args
        self.trims = tuple(int(n.n) for n in trims.body)[::-1]
        assert all(t>0 for t in self.trims)

    def get_shape(self, sy):
        n = len(self.trims)
        assert len(sy) >= n
        return sy[:-n] + tuple(s-2*t for s, t in zip(sy[-n:], self.trims))

    def compile(self, ctx, sy, y, z):
        n = len(self.trims)
        for s, t in zip(sy[-n:], self.trims)[::-1]:
            z = ctx.append_dim2(z, s-2*t)
            n_in_ = y.dim(isl.dim_type.in_)
            n_out = y.dim(isl.dim_type.out)
            y = ctx.append_dim1(y, s-2*t)
            y = ctx.increase_dim(y, isl.dim_type.out)
            space = y.get_space()
            y = y.add_constraint(
                isl.Constraint.alloc_equality(space)
                .set_coefficient_val(
                    isl.dim_type.in_, n_in_, isl.Val.int_from_si(ctx.isl_context, 1))
                .set_coefficient_val(
                    isl.dim_type.out, n_out, isl.Val.int_from_si(ctx.isl_context, -1))
                .set_constant_val(
                    isl.Val.int_from_si(ctx.isl_context, t))
            )

        for s in sy[:-n][::-1]:
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)

        ctx.def_stmts.add((z, ('var', y)))

    def compile_acc_y(self, ctx, sy, y, z, acc, g):
        n = len(self.trims)
        for s, t in zip(sy[-n:], self.trims)[::-1]:
            acc = ctx.append_dim2(acc, s-2*t)
            n_in_ = g.dim(isl.dim_type.in_)
            n_out = g.dim(isl.dim_type.out)
            g = ctx.append_dim1(g, s-2*t)
            g = ctx.increase_dim(g, isl.dim_type.out)
            space = g.get_space()
            g = g.add_constraint(
                isl.Constraint.alloc_equality(space)
                .set_coefficient_val(
                    isl.dim_type.in_, n_in_, isl.Val.int_from_si(ctx.isl_context, 1))
                .set_coefficient_val(
                    isl.dim_type.out, n_out, isl.Val.int_from_si(ctx.isl_context, -1))
                .set_constant_val(
                    isl.Val.int_from_si(ctx.isl_context, t))
            )

        for s in sy[:-n][::-1]:
            g = ctx.append_dim2(g, s)
            acc = ctx.append_dim2(acc, s)

        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('var', acc)
              ))
            ))


class Duplicate(object):
    rank = None

    def __init__(self, table, args):
        dups, = args
        self.dups = tuple(int(n.n) for n in dups.body)[::-1]
        assert all(t>0 for t in self.dups)

    def get_shape(self, sy):
        return sy + self.dups

    def compile(self, ctx, sy, y, z):
        for s in self.dups[::-1]:
            y = ctx.append_dim1(y, s)
            z = ctx.append_dim2(z, s)

        for s in sy[::-1]:
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)

        ctx.def_stmts.add((z, ('var', y)))

    def compile_acc_y(self, ctx, sy, y, z, acc, g):
        for s in self.dups[::-1]:
            g = ctx.append_dim1(g, s)
            acc = ctx.append_dim2(acc, s)

        for s in sy[::-1]:
            g = ctx.append_dim2(g, s)
            acc = ctx.append_dim2(acc, s)

        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('var', acc)
              ))
            ))


class Stride(object):
    rank = None

    def __init__(self, table, args):
        strides, = args
        self.strides = tuple(int(n.n) for n in strides.body)[::-1]
        assert all(t>0 for t in self.strides)

    def get_shape(self, sy):
        n = len(self.strides)
        assert len(sy) >= n
        assert all((a%b == 1) for a,b in zip(sy[-n:], self.strides))
        return sy[:-n] + tuple((a/b+1) for a,b in zip(sy[-n:], self.strides))

    def compile(self, ctx, sy, y, z):
        n = len(self.strides)

        for s, t in zip(sy[-n:], self.strides)[::-1]:
            s = s/t + 1
            z = ctx.append_dim2(z, s)
            n_in_ = y.dim(isl.dim_type.in_)
            n_out = y.dim(isl.dim_type.out)
            y = ctx.append_dim1(y, s)
            y = ctx.increase_dim(y, isl.dim_type.out)
            space = y.get_space()
            y = y.add_constraint(
                isl.Constraint.alloc_equality(space)
                .set_coefficient_val(
                    isl.dim_type.in_, n_in_, isl.Val.int_from_si(ctx.isl_context, t))
                .set_coefficient_val(
                    isl.dim_type.out, n_out, isl.Val.int_from_si(ctx.isl_context, -1))
            )

        for s in sy[:-n][::-1]:
            y = ctx.append_dim2(y, s)
            z = ctx.append_dim2(z, s)

        ctx.def_stmts.add((z, ('var', y)))


    def compile_acc_y(self, ctx, sy, y, z, acc, g):
        n = len(self.strides)

        for s, t in zip(sy[-n:], self.strides)[::-1]:
            s = s/t + 1
            acc = ctx.append_dim2(acc, s)
            n_in_ = g.dim(isl.dim_type.in_)
            n_out = g.dim(isl.dim_type.out)
            g = ctx.append_dim1(g, s)
            g = ctx.increase_dim(g, isl.dim_type.out)
            space = g.get_space()
            g = g.add_constraint(
                isl.Constraint.alloc_equality(space)
                .set_coefficient_val(
                    isl.dim_type.in_, n_in_, isl.Val.int_from_si(ctx.isl_context, t))
                .set_coefficient_val(
                    isl.dim_type.out, n_out, isl.Val.int_from_si(ctx.isl_context, -1))
            )

        for s in sy[:-n][::-1]:
            g = ctx.append_dim2(g, s)
            acc = ctx.append_dim2(acc, s)

        ctx.update_stmts.add(
            (g,
             ('call', '+',
              ( ('var', g),
                ('var', acc)
              ))
            ))


builtins = {
    '+': Plus(),
    '-': Minus(),
    '*': Multiply(),
    '/': Divide(),
    '**': Power(),
    'log': Log(),
    'exp': Exp(),
    'reduce': Reduce,
    '<.': Min(),
    '>.': Max(),
    'oblique': Oblique,
    'trim': Trim,
    'duplicate': Duplicate,
    'stride': Stride,
}
