from . import symbol
from .utils import product


class ArrayView(object):

    def __init__(self, data, offset, length):
        self.data = data
        self.offset = offset
        self.length = length

    def subview(self, n, blocks):
        return ArrayView(self.data, self.offset+(self.length/blocks)*n, self.length/blocks)

    def __getitem__(self, index):
        assert 0 <= index < self.length
        return self.data[self.offset+index]

    def __setitem__(self, index, value):
        assert 0 <= index < self.length
        self.data[self.offset+index] = value


class Array(object):

    def __init__(self, shape, data):
        self.shape = shape
        self.data = data

    def view(self):
        return ArrayView(self.data, 0, len(self.data))

    def __eq__(self, other):
        return (self.shape == other.shape) and (self.data == other.data)

    def allclose(self, other):
        return (self.shape == other.shape) and all(round(abs(x-y),7)==0 for x, y in zip(self.data,other.data))

    def __repr__(self):
        return "Array(%s, %s)"%(repr(self.shape),repr(self.data))


def interp_monad(op, ranks, sy, vy, vz):
    if not ranks:
        ry = op.rank
        inner = op.interp
    else:
        ry = ranks[-1]
        def inner(sy, vy, vz):
            interp_monad(op, ranks[:-1], sy, vy, vz)

    if ry is None:
        ry = len(sy)

    p = product(sy[ry:])
    for i in xrange(p):
        inner(sy[:ry], vy.subview(i, p),
              vz.subview(i, p))


def interp_acc_monad(op, ranks, sy, vy, vz, acc, vg):
    if not ranks:
        ry = op.rank
        inner = op.interp_acc_y
    else:
        ry = ranks[-1]
        def inner(sy, vy, vz, acc, vg):
            interp_acc_monad(op, ranks[:-1], sy, vy, vz, acc, vg)

    if ry is None:
        ry = len(sy)

    p = product(sy[ry:])
    for i in xrange(p):
        inner(sy[:ry], vy.subview(i, p),
              vz.subview(i, p),
              acc.subview(i, p),
              vg.subview(i, p))


def interp_dyad(op, ranks, sx, vx, sy, vy, vz):
    if not ranks:
        rx, ry = op.rank
        inner = op.interp
    else:
        rx, ry = ranks[-1]
        def inner(sx, vx, sy, vy, vz):
            interp_dyad(op, ranks[:-1], sx, vx, sy, vy, vz)

    if rx is None:
        rx = len(sx)
    if ry is None:
        ry = len(sy)

    sxo, syo = sx[rx:], sy[ry:]
    px, py = product(sxo), product(syo)

    if len(sxo) <= len(syo):
        common = px
        extra = py/common

        for i in xrange(common):
            for j in xrange(extra):
                inner(sx[:rx], vx.subview(i, px),
                      sy[:ry], vy.subview(i*extra+j, py),
                      vz.subview(i*extra+j, py))
    else:
        common = py
        extra = px/common

        for i in xrange(common):
            for j in xrange(extra):
                inner(sx[:rx], vx.subview(i*extra+j, px),
                      sy[:ry], vy.subview(i, py),
                      vz.subview(i*extra+j, px))


def interp_acc_dyad_x(op, ranks, sx, vx, sy, vy, vz, acc, vg):
    if not ranks:
        rx, ry = op.rank
        inner = op.interp_acc_x
    else:
        rx, ry = ranks[-1]
        def inner(sx, vx, sy, vy, vz, acc, vg):
            interp_acc_dyad_x(op, ranks[:-1], sx, vx, sy, vy, vz, acc, vg)

    if rx is None:
        rx = len(sx)
    if ry is None:
        ry = len(sy)

    sxo, syo = sx[rx:], sy[ry:]
    px, py = product(sxo), product(syo)

    if len(sxo) <= len(syo):
        common = px
        extra = py/common

        for i in xrange(common):
            for j in xrange(extra):
                inner(sx[:rx], vx.subview(i, px),
                      sy[:ry], vy.subview(i*extra+j, py),
                      vz.subview(i*extra+j, py),
                      acc.subview(i*extra+j, py),
                      vg.subview(i, px))
    else:
        common = py
        extra = px/common

        for i in xrange(common):
            for j in xrange(extra):
                inner(sx[:rx], vx.subview(i*extra+j, px),
                      sy[:ry], vy.subview(i, py),
                      vz.subview(i*extra+j, px),
                      acc.subview(i*extra+j, px),
                      vg.subview(i*extra+j, px))


def interp_acc_dyad_y(op, ranks, sx, vx, sy, vy, vz, acc, vg):
    if not ranks:
        rx, ry = op.rank
        inner = op.interp_acc_y
    else:
        rx, ry = ranks[-1]
        def inner(sx, vx, sy, vy, vz, acc, vg):
            interp_acc_dyad_y(op, ranks[:-1], sx, vx, sy, vy, vz, acc, vg)

    if rx is None:
        rx = len(sx)
    if ry is None:
        ry = len(sy)

    sxo, syo = sx[rx:], sy[ry:]
    px, py = product(sxo), product(syo)

    if len(sxo) <= len(syo):
        common = px
        extra = py/common

        for i in xrange(common):
            for j in xrange(extra):
                inner(sx[:rx], vx.subview(i, px),
                      sy[:ry], vy.subview(i*extra+j, py),
                      vz.subview(i*extra+j, py),
                      acc.subview(i*extra+j, py),
                      vg.subview(i*extra+j, py))
    else:
        common = py
        extra = px/common

        for i in xrange(common):
            for j in xrange(extra):
                inner(sx[:rx], vx.subview(i*extra+j, px),
                      sy[:ry], vy.subview(i, py),
                      vz.subview(i*extra+j, px),
                      acc.subview(i*extra+j, px),
                      vg.subview(i, py))


def _interp(table, kwargs):
    values = {}

    for k in table.inputs:
        values[table.symbols[k]] = kwargs[k]

    for name in table.vars:
        if name in values:
            continue

        v = table.vars[name]
        z = Array(v.shape, [0.0 for _ in xrange(product(v.shape))])

        if isinstance(v, symbol.ApplyMonad):
            interp_monad(
                v.op,
                v.rank,
                table.vars[v.y].shape,
                values[v.y].view(),
                z.view())
        elif isinstance(v, symbol.ApplyDyad):
            interp_dyad(
                v.op,
                v.rank,
                table.vars[v.x].shape,
                values[v.x].view(),
                table.vars[v.y].shape,
                values[v.y].view(),
                z.view())
        elif isinstance(v, symbol.AccMonad):
            u = table.vars[v.v]
            interp_acc_monad(
                u.op,
                u.rank,
                table.vars[u.y].shape,
                values[u.y].view(),
                values[v.v].view(),
                values[v.acc].view(),
                z.view())
        elif isinstance(v, symbol.AccDyad):
            u = table.vars[v.v]

            if v.operand == 'x':
                interp_acc_dyad_x(
                    u.op,
                    u.rank,
                    table.vars[u.x].shape,
                    values[u.x].view(),
                    table.vars[u.y].shape,
                    values[u.y].view(),
                    values[v.v].view(),
                    values[v.acc].view(),
                    z.view())
            elif v.operand == 'y':
                interp_acc_dyad_y(
                    u.op,
                    u.rank,
                    table.vars[u.x].shape,
                    values[u.x].view(),
                    table.vars[u.y].shape,
                    values[u.y].view(),
                    values[v.v].view(),
                    values[v.acc].view(),
                    z.view())
            else:
                raise NotImplementedError
        elif isinstance(v, symbol.Literal):
            z.data = [float(n.n) for n in v.body]
        elif isinstance(v, symbol.Constant):
            z.data = [n for n in v.value]
        else:
            raise NotImplementedError

        values[name] = z

    return values


def interp(table, **kwargs):
    return _interp(table, kwargs)


def shell_interp(table, values):
    d = _interp(table, values)

    for k in table.outputs:
        values[k] = d[table.symbols[k]]
