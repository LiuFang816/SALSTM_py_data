# ported from ppcg

# Copyright 2010-2011 INRIA Saclay
# Copyright 2012-2013 Ecole Normale Superieure
# Copyright 2015-2016 Sven Verdoolaege
#
# Use of this software is governed by the MIT license
#
# Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
# Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
# 91893 Orsay, France
# and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France


import islpy as isl
from .utils import product, Counter
from .isl_utils import find_kernel_id, get_sizes, list_to_multival, to_sets, to_maps, detect_strides
from .compile import get_uses
from .gpu_tree import gpu_tree_move_up_to, gpu_tree_move_down_to, gpu_tree_move_down_to_depth, gpu_tree_move_left_to_sync, gpu_tree_move_right_to_sync, gpu_tree_ensure_sync_after_core
from .gpu_group import gpu_group_references



class KernelInfo(object):
    required_fields = ('domain', 'grid_sizes', 'block_sizes')

    def __init__(self, **kwargs):
        assert all((k in kwargs) for k in self.required_fields)

        for k,v in kwargs.items():
            self.__dict__[k] = v

    def __repr__(self):
        return repr(self.__dict__)


def n_outer_coincidence(node):
    assert node.get_type() == isl.schedule_node_type.band
    assert node.band_get_permutable()

    n = node.band_n_member()

    for i in xrange(n):
        if not node.band_member_get_coincident(i):
            return i

    return n


def parameter_vector(space, prefix, n):
    for i in xrange(n):
        name = prefix+str(i)
        pos = space.find_dim_by_name(isl.dim_type.param, name)
        if pos >= 0:
            continue
        pos = space.dim(isl.dim_type.param)
        space = space.add_dims(isl.dim_type.param, 1)
        space = space.set_dim_name(isl.dim_type.param, pos, name)

    ma = isl.MultiAff.zero(space)
    ls = isl.LocalSpace.from_space(space.domain())

    for i in xrange(n):
        name = prefix+str(i)
        pos = space.find_dim_by_name(isl.dim_type.param, name)
        aff = isl.Aff.var_on_domain(ls, isl.dim_type.param, pos)
        ma = ma.set_aff(i, aff)

    return ma


def extract_sizes(set):
    n = set.dim(isl.dim_type.param)
    ls = isl.LocalSpace.from_space(set.get_space())
    return [
        set.max_val(
            isl.Aff.var_on_domain(ls, isl.dim_type.param, i))
        .get_num_si() + 1
        for i in xrange(n)]


def make_context(ctx, prefix, sizes):
    n = len(sizes)

    space = isl.Space.set_alloc(ctx, n, 0)

    for i in xrange(n):
        space = space.set_dim_name(isl.dim_type.param, i, prefix+str(i))

    bs = isl.BasicSet.universe(space)

    for i in xrange(n):
        bs = bs.add_constraint(
            isl.Constraint.alloc_inequality(space)
            .set_coefficient_val(
                isl.dim_type.param, i,  isl.Val.int_from_si(ctx, -1)
            )
            .set_constant_val(
                isl.Val.int_from_si(ctx, sizes[i] - 1)
            )
        ).add_constraint(
            isl.Constraint.alloc_inequality(space)
            .set_coefficient_val(
                isl.dim_type.param, i,  isl.Val.int_from_si(ctx, 1)
            )
        )

    return isl.Set.from_basic_set(bs)

def insert_context(node, grid_sizes, block_sizes):
    ctx = node.get_ctx()

    c1 = make_context(ctx, "b", grid_sizes)
    c2 = make_context(ctx, "t", block_sizes)
    return node.insert_context(c2.intersect(c1))


def construct_band_tiles_sizes(node, tile_size):
    space = node.band_get_space()
    return list_to_multival(space, tile_size)


def set_schedule_modulo(node, prefix, sizes):
    n = len(sizes)
    if n == 0:
        return node.get_universe_domain()

    n_zero = n - node.band_n_member()
    mupa = node.band_get_partial_schedule()
    mv = construct_band_tiles_sizes(node, sizes[n_zero:])
    mupa = mupa.mod_multi_val(mv)

    space = mupa.get_space()
    space = space.params()
    space = isl.Space.set_from_params(space)
    space = space.add_dims(isl.dim_type.set, n_zero)
    ma = isl.MultiAff.zero(space)

    domain = node.get_universe_domain()
    mupa2 = isl.MultiUnionPwAff.multi_aff_on_domain(domain, ma)
    mupa = mupa2.range_product(mupa)

    space = mupa.get_space()
    ma = parameter_vector(space, prefix, n)

    mupa2 = isl.MultiUnionPwAff.multi_aff_on_domain(domain, ma)
    mupa = mupa.sub(mupa2)
    return mupa.zero_union_set()


def scale_band(node, sizes):
    mv = list_to_multival(node.band_get_space(), sizes)
    node = node.band_scale(mv)
    return node


def create_local_arrays(ctx, domain):
    arrays = {} # {name: [access]} access: ("init", Stmt->Var)

    for s in to_sets(domain):
        name = s.get_tuple_name()

        if name in ctx.init_stmts:
            stmt = ctx.init_stmts[name]
            access = stmt[0].intersect_domain(s)
            v = access.get_tuple_name(isl.dim_type.out)
            arrays[v] = arrays.get(v, ()) + (("init", access, stmt[0]),)
        elif name in ctx.fini_stmts:
            stmt = ctx.fini_stmts[name]
            access = stmt[0].intersect_domain(s)
            v = access.get_tuple_name(isl.dim_type.out)
            arrays[v] = arrays.get(v, ()) + (("fini", access, stmt[0]),)
        elif name in ctx.update_stmts:
            stmt = ctx.update_stmts[name]
            access = stmt[0].intersect_domain(s)
            v = access.get_tuple_name(isl.dim_type.out)
            arrays[v] = arrays.get(v, ()) + (("update", access, stmt[0]),)

            for u in get_uses(stmt[1]):
                access = u.intersect_domain(s)
                v = access.get_tuple_name(isl.dim_type.out)
                arrays[v] = arrays.get(v, ()) + (("use", access, u),)

        elif name in ctx.def_stmts:
            stmt = ctx.def_stmts[name]
            access = stmt[0].intersect_domain(s)
            v = access.get_tuple_name(isl.dim_type.out)
            arrays[v] = arrays.get(v, ()) + (("def", access, stmt[0]),)

            for u in get_uses(stmt[1]):
                access = u.intersect_domain(s)
                v = access.get_tuple_name(isl.dim_type.out)
                arrays[v] = arrays.get(v, ()) + (("use", access, u),)

    return arrays

def create_from_access(group, read):
    space = group.access.get_space().wrap().map_from_set()
    name = group.tiling.get_tuple_name(isl.dim_type.out)
    id = isl.Id.alloc(space.get_ctx(), ("read_" if read else "write_") + name, None)
    space = space.set_tuple_id(isl.dim_type.in_, id)
    return isl.MultiAff.identity(space)



def add_copies_group(node, group, block_sizes, core, sync_counter, read):
    kernel_depth = node.get_schedule_depth()
    depth = group.access.dim(isl.dim_type.in_)

    node = gpu_tree_move_down_to_depth(node, depth, core)

    from_access = create_from_access(group, read)
    ma = group.tiling.pullback_multi_aff(from_access)
    mpa = isl.MultiPwAff.from_multi_aff(ma)
    mupa = isl.MultiUnionPwAff.from_multi_pw_aff(mpa)

    domain = group.access.wrap()
    # if read:
    #     domain = group_tile(group).wrap()


    domain = domain.preimage_multi_aff(from_access)
    access = domain.wrapped_domain_map().reverse().coalesce()

    graft = isl.ScheduleNode.from_extension(access)

    graft = graft.child(0)
    graft = graft.insert_partial_schedule(mupa)

    filter = set_schedule_modulo(graft, "t", block_sizes)
    graft = graft.insert_filter(filter)

    while graft.has_parent():
        graft = graft.parent()

    if read:
        if kernel_depth < depth:
            node = gpu_tree_ensure_sync_after_core(node, core, sync_counter)

        node = gpu_tree_move_left_to_sync(node, core, sync_counter)
        node = node.graft_before(graft)
    else:
        node = gpu_tree_move_right_to_sync(node, core, sync_counter)
        node = node.graft_after(graft)

        # if kernel_depth < depth:
        #     node = add_group_write_sync(group)


    node = gpu_tree_move_up_to(node, "kernel")
    return node



def add_copies(node, all_groups, block_sizes, core, sync_counter):
    for v, groups in all_groups.iteritems():
        for group in groups:
            if group.has_read():
                node = add_copies_group(node, group, block_sizes, core, sync_counter, read=True)
            if group.has_write():
                node = add_copies_group(node, group, block_sizes, core, sync_counter, read=False)

    return node


def group_statements(node, id):
    return node.group(id)


def create_kernel(ctx, node, grid_sizes, block_sizes):
    domain = node.get_domain()
    arrays = create_local_arrays(ctx, domain)

    core = domain.universe()

    contraction = node.get_subtree_contraction()

    node = node.insert_mark(isl.Id.alloc(node.get_ctx(), "kernel", None))
    node = node.child(0)
    node = node.child(0)

    # node = node.insert_mark(isl.Id.alloc(node.get_ctx(), "shared", None))
    # node = node.child(0)

    node = node.insert_mark(isl.Id.alloc(node.get_ctx(), "thread", None))


    node = gpu_tree_move_up_to(node, "kernel")

    node = node.child(0)

    assert n_outer_coincidence(node) >= len(grid_sizes)

    if len(grid_sizes) < node.band_n_member():
        node = node.band_split(len(grid_sizes))

    block_filter = set_schedule_modulo(node, "b", grid_sizes)
    grid_sizes = extract_sizes(block_filter.intersect(node.get_domain()).params())

    node = scale_band(node, grid_sizes)
    node = node.parent()

    node = gpu_tree_move_down_to(node, core, "thread")
    node = node.child(0)

    assert n_outer_coincidence(node) >= len(block_sizes)

    if len(block_sizes) < node.band_n_member():
        node = node.band_split(len(block_sizes))

    thread_filter = set_schedule_modulo(node, "t", block_sizes)
    block_sizes = extract_sizes(thread_filter.intersect(node.get_domain()).params())

    node = gpu_tree_move_up_to(node, "kernel")
    node = node.child(0)


    node = insert_context(node, grid_sizes, block_sizes)
    node = node.child(0)

    node = node.insert_filter(block_filter)

    node = gpu_tree_move_up_to(node, "kernel")

    groups = gpu_group_references(arrays, node, core, contraction)

    node = node.child(0)

    context = make_context(node.get_ctx(), "b", grid_sizes)


    node = gpu_tree_move_down_to(node, core, "thread")
    node = node.child(0)
    node = node.insert_filter(thread_filter)


    node = gpu_tree_move_up_to(node, "kernel")

    sync_counter = Counter("sync%d")

    node = add_copies(node, groups, block_sizes, core, sync_counter)

    node = gpu_tree_move_down_to(node, core, "thread")
    node = node.delete()

    node = gpu_tree_move_up_to(node, "kernel")
    node = node.delete()

    node = node.insert_mark(node.parent().mark_get_id())

    info = KernelInfo(
        domain=domain,
        grid_sizes=grid_sizes,
        block_sizes=block_sizes,
        groups = groups)

    return node, info


def is_permutable(node):
    node_type = node.get_type()

    if node_type != isl.schedule_node_type.band:
        return False
    if not node.band_get_permutable():
        return False
    if node.band_n_member() < 1:
        return False
    if not node.band_member_get_coincident(0):
        return False
    return True


def create_kernels(ctx, kernels, node, sizes):
    kernel_id = find_kernel_id(node)
    if kernel_id is not None:
        node = node.child(0)
        if is_permutable(node):
            grid_sizes = get_sizes(sizes, "grid", kernel_id)
            block_sizes = get_sizes(sizes, "block", kernel_id)
            if grid_sizes is not None and block_sizes is not None:
                assert len(grid_sizes) == len(block_sizes)
                node, info = create_kernel(ctx, node, grid_sizes, block_sizes)
                kernels[kernel_id] = info

        node = node.parent()
        node = node.delete()
        return node

    n = node.n_children()

    for i in xrange(n):
        node = node.child(i)
        node = create_kernels(ctx, kernels, node, sizes)
        node = node.parent()

    return node


def map_to_device(ctx, schedule, sizes):
    node = schedule.get_root()

    kernels = {}
    node = create_kernels(ctx, kernels, node, sizes)
    return node.get_schedule(), kernels
