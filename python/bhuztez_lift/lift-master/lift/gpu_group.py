# ported from ppcg

# Copyright 2010-2011 INRIA Saclay
# Copyright 2012-2014 Ecole Normale Superieure
# Copyright 2015      Sven Verdoolaege
#
# Use of this software is governed by the MIT license
#
# Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
# Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
# 91893 Orsay, France
# and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France


from itertools import combinations
import islpy as isl
from .gpu_tree import gpu_tree_move_up_to, gpu_tree_move_down_to


def prefix_with_equalities(node):
    return node.get_prefix_schedule_relation().detect_equalities()


def expand(s, contraction):
    return s.preimage_domain_union_pw_multi_aff(contraction)


def gpu_group_references(arrays, node, core, contraction):
    node = node.child(0) # context
    node = node.child(0) # filter
    node = node.child(0) # sequence
    node = node.child(0)

    copy_sched = prefix_with_equalities(node)

    return {
        name: group_array_references(name, refs, copy_sched)
        for name, refs in arrays.iteritems()}


class ArrayBound(object):

    def __init__(self):
        self.size = None
        self.lb = None
        self.stride = None
        self.shift = None


def extract_stride(c, bound, stride, sign):
    bound.stride = stride

    space = c.get_space()
    space = space.domain()
    nparam = space.dim(isl.dim_type.param)
    nvar = space.dim(isl.dim_type.set)
    v = c.get_constant_val()
    if sign < 0:
        v = v.neg()
    aff = isl.Aff.zero_on_domain(isl.LocalSpace.from_space(space))
    aff = aff.set_constant_val(v)

    for i in xrange(nparam):
        if not c.involves_dims(isl.dim_type.param, i, 1):
            continue
        v = c.get_coefficient_val(isl.dim_type.param, i)
        if sign < 0:
            v = v.neg()
        aff = aff.add_coefficient_val(isl.dim_type.param, i, v)

    for i in xrange(nvar):
        if not c.involves_dims(isl.dim_type.in_, i, 1):
            continue
        v = c.get_coefficient_val(isl.dim_type.in_, i)
        if sign < 0:
            v = v.neg()
        aff = aff.add_coefficient_val(isl.dim_type.in_, i, v)

    bound.shift = aff


def check_stride(bound, bounds):
    bound.stride = None
    hull = bounds.affine_hull()
    ctx = bounds.get_ctx()

    for c in hull.get_constraints():
        n_div = c.dim(isl.dim_type.div)
        v = c.get_coefficient_val(isl.dim_type.out, 0)

        if n_div and (v.is_one() or v.is_negone()):
            s = v.sgn()
            stride = isl.Val.zero(ctx)
            for i in xrange(n_div):
                v = c.get_coefficient_val(isl.dim_type.div, i)
                stride = stride.gcd(v)

            if ((not stride.is_zero())
                and stride.gt(bound.stride)):
                extract_stride(c, bound, stride, s)


    if bound.stride is None:
        return bounds

    shift = isl.BasicMap.from_aff(bound.shift)
    space = bounds.get_space()
    bmap = isl.BasicMap.universe(space).domain_map()
    shift = bmap.apply_range(shift)
    space = bounds.get_space()
    id = isl.BasicMap.universe(space).range_map()
    shift = id.sum(shift)
    space = bounds.get_space()
    id = isl.BasicMap.universe(space).domain_map()
    shift = id.range_product(shift)

    space = bounds.get_space().domain()
    id = space.map_from_set().identity()
    space = bounds.get_space().range()
    aff = isl.Aff.zero_on_domain(isl.LocalSpace.from_space(space))
    aff = aff.add_coefficient_val(isl.dim_type.in_, 0, isl.Val.int_from_si(ctx, 1))
    aff = aff.scale_down_val(bounds.stride)
    scale = isl.BasicMap.from_aff(aff)
    scale = id.product(scale)

    bmap = shift.apply_range(scale)
    bset = bounds.wrap().apply(bmap)
    bounds = bset.unwrap()
    return bounds



def compute_array_dim_size(bound, bounds):
    ctx = bounds.get_ctx()
    bounds = bounds.detect_equalities()
    bounds = check_stride(bound, bounds)

    bound.size = None
    bound.lb = None

    pos = bounds.dim(isl.dim_type.in_)
    bset = bounds.wrap().flatten().compute_divs().simple_hull()

    size = None

    for c in bset.get_constraints():
        nparam = bset.dim(isl.dim_type.param)
        n_div = c.dim(isl.dim_type.div)

        if (c.involves_dims(isl.dim_type.div, 0, n_div)
            or not c.is_lower_bound(isl.dim_type.set, pos)):
            continue
        aff = c.get_bound(isl.dim_type.set, pos)
        aff = aff.ceil()
        lb = aff
        aff = aff.neg()
        aff = aff.add_coefficient_val(isl.dim_type.in_, pos, isl.Val.int_from_si(ctx, 1))
        v = bset.max_val(aff)

        if v.is_int():
            v = v.add(isl.Val.int_from_si(ctx, 1))
            if bound.size is None or v.lt(bound.size):
                bound.size = v
                bound.lb = lb.drop_dims(isl.dim_type.in_, pos, 1)


def can_tile(access):
    bounds = ()

    n = access.dim(isl.dim_type.out)

    for i in xrange(n):
        access_i = access.project_out(isl.dim_type.out, 0, i)
        access_i = access_i.project_out(isl.dim_type.out, 1, n - i - 1)
        access_i = access_i.compute_divs()
        hull = access_i.simple_hull()
        bound = ArrayBound()
        compute_array_dim_size(bound, hull)
        bounds = bounds + (bound,)

    return bounds


class RefGroup(object):

    def __init__(self):
        self.access = None
        self.bounds = None
        self.refs = None

        self.inits = None
        self.updates = None
        self.finis = None
        self.defs = None
        self.uses = None

    def has_read(self):
        return not self.finis.union(self.updates).subtract(self.inits).union(self.uses.subtract(self.finis).subtract(self.defs)).is_empty()

    def has_write(self):
        return not self.inits.union(self.updates).union(self.defs).is_empty()



def populate_array_reference(ref, copy_sched):
    type, access, _ = ref
    access = isl.Map.from_union_map(access.apply_domain(copy_sched))
    access = access.detect_equalities()

    bounds = can_tile(access)

    group = RefGroup()
    group.access = access
    group.bounds = bounds
    group.refs = (ref,)

    space = access.get_space()

    group.inits = isl.Map.empty(space)
    group.updates = isl.Map.empty(space)
    group.finis = isl.Map.empty(space)
    group.defs = isl.Map.empty(space)
    group.uses = isl.Map.empty(space)

    if type == "init":
        group.inits = access
    elif type == "update":
        group.updates = access
    elif type == "fini":
        group.finis = access
    elif type == "def":
        group.defs = access
    elif type == "use":
        group.uses = access
    else:
        raise NotImplementedError

    return group


def join_groups(a, b):
    a.access = a.access.union(b.access)
    a.bounds = can_tile(a.access)
    a.refs = a.refs + b.refs
    a.inits = a.inits.union(b.inits)
    a.updates = a.updates.union(b.updates)
    a.finis = a.finis.union(b.finis)
    a.defs = a.defs.union(b.defs)
    a.uses = a.uses.union(b.uses)


def has_overlapping_writes(a, b):
    if not a.inits.intersect(b.updates).is_empty():
        return True

    if not b.inits.intersect(a.updates).is_empty():
        return True

    if not a.updates.intersect(b.updates).is_empty():
        return True

    if not a.updates.intersect(b.finis).is_empty():
        return True

    if not b.updates.intersect(a.finis).is_empty():
        return True

    if not a.finis.intersect(b.uses).is_empty():
        return True

    if not b.finis.intersect(a.uses).is_empty():
        return True

    if not a.defs.intersect(b.uses).is_empty():
        return True

    if not b.defs.intersect(a.uses).is_empty():
        return True

    return False


def group_array_references(name, refs, copy_sched):
    groups = [
        populate_array_reference(ref, copy_sched)
        for ref in refs]

    group_of = range(len(groups))

    def find(i):
        j = group_of[i]
        if i != j:
            return find(j)
        return i

    def mark_merge(i,j):
        i = find(i)
        j = find(j)
        group_of[j] = i


    for i, j in combinations(range(len(groups)), 2):
        if has_overlapping_writes(groups[i], groups[j]):
            mark_merge(i,j)

    for i in xrange(len(groups)):
        j = find(i)
        if j != i:
            join_groups(groups[j], groups[i])

    groups = [
        groups[i]
        for i in xrange(len(groups))
        if find(i) == i]

    for i, group in enumerate(groups):
        local_name = "shared_" + name + "_" + str(i)
        gpu_array_ref_group_compute_tiling(group, local_name)

    return groups


def gpu_array_ref_group_compute_tiling(group, local_name):
    access = group.access
    depth = access.dim(isl.dim_type.in_)
    space = access.get_space()
    space = isl.Space.from_range(space.range())
    space = space.add_dims(isl.dim_type.in_, depth)
    insert_array = isl.MultiAff.domain_map(space)

    if any(bound.shift is not None for bound in group.bounds):
        assert False
    else:
        tiling = isl.MultiAff.range_map(space)

    lb = isl.MultiAff.zero(space)

    for i, bound in enumerate(group.bounds):
        lb = lb.set_aff(i, bound.lb)

    lb = lb.pullback_multi_aff(insert_array)
    tiling = tiling.sub(lb)

    tiling = tiling.set_tuple_name(isl.dim_type.out, local_name)
    group.tiling = tiling
