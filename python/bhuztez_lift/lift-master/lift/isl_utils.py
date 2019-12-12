import re
import islpy as isl

isl.schedule_node_type = isl._isl.schedule_node_type

def isl_schedule_node_get_type(node):
    return isl._isl.lib.isl_schedule_node_get_type(node.data)

isl.ScheduleNode.get_type = isl_schedule_node_get_type

def isl_constraint_dim(c, type):
    return isl._isl.lib.isl_constraint_dim(c.data, type)

isl.Constraint.dim = isl_constraint_dim


def to_list(l):
    r = []
    l.foreach(lambda e: r.append(e))
    return r


def to_sets(union_set):
    sets = []
    union_set.foreach_set(lambda x: sets.append(x))
    return sets


def to_maps(union_map):
    maps = []
    union_map.foreach_map(lambda x: maps.append(x))
    return maps


def list_to_multival(space, l):
    ctx = space.get_ctx()
    mv = isl.MultiVal.zero(space)
    for i, e in enumerate(l):
        v = isl.Val.int_from_si(ctx, e)
        mv = mv.set_val(i,v)
    return mv


def find_kernel_id(node):
    if node.get_type() == isl.schedule_node_type.mark:
        mark = node.mark_get_id().get_name()
        match = re.match(r'kernel\[(\d+)\]', mark)
        if match is not None:
            return int(match.group(1))


def get_sizes(sizes, name, kernel_id):
    ctx = sizes.get_ctx()
    space = isl.Space.set_alloc(ctx, 0, 1)
    space = space.set_tuple_name(isl.dim_type.set, "kernel")
    uset = (
        isl.Set.universe(space)
        .add_constraint(
            isl.Constraint.alloc_equality(space)
            .set_coefficient_val(
                isl.dim_type.set, 0, isl.Val.int_from_si(ctx, 1))
            .set_constant_val(
                isl.Val.int_from_si(ctx, kernel_id)))
    )

    for set in to_sets(uset.apply(sizes)):
        if set.get_tuple_name() == name:
            n = set.dim(isl.dim_type.set)

            return [
                set.plain_get_val_if_fixed(
                    isl.dim_type.set, i).get_num_si()
                for i in xrange(n)]


def detect_constraint_stride(ctx, c, pos):
    stride = isl.Val.one(ctx)

    if not c.is_equality():
        return stride
    if not c.involves_dims(isl.dim_type.set, pos, 1):
        return stride

    stride = isl.Val.zero(ctx)
    n_div = c.dim(isl.dim_type.div)

    for i in xrange(n_div):
        stride = stride.gcd(c.get_coefficient_val(isl.dim_type.div, i))

    m = stride.gcd(c.get_coefficient_val(isl.dim_type.set, pos))
    stride = stride.div(m)

    if stride.is_zero():
        stride = isl.Val.one(ctx)
    return stride


def detect_stride(ctx, constraints, pos):
    stride = isl.Val.one(ctx)

    for c in constraints:
        s = detect_constraint_stride(ctx, c, pos)
        stride = stride.mul(s).div(stride.gcd(s))

    return stride.get_num_si()


def detect_strides(domain):
    constraints = domain.get_constraints()
    n = domain.dim(isl.dim_type.set)

    ctx = domain.get_ctx()

    return [detect_stride(ctx, constraints, i)
            for i in xrange(n)]


def scale_down_node(node):
    domain = isl.Set.from_union_set(
        node.band_get_partial_schedule_union_map()
        .intersect_domain(node.get_domain())
        .range()).affine_hull()

    strides = detect_strides(domain)

    mv = list_to_multival(node.band_get_space(), strides)
    node = node.band_scale_down(mv)
    return node
