import islpy as isl
from .isl_utils import get_sizes, find_kernel_id, list_to_multival


def tile_band(node, sizes):
    if sizes is None:
        return node

    ctx = node.get_ctx()
    n = len(sizes)
    if n < node.band_n_member():
        node = node.band_split(n)

    mv = list_to_multival(node.band_get_space(), sizes)

    scale_tile = ctx.get_tile_scale_tile_loops()
    shift_point = ctx.get_tile_shift_point_loops()
    ctx.set_tile_scale_tile_loops(0)
    ctx.set_tile_shift_point_loops(0)

    node = node.band_tile(mv)

    ctx.set_tile_scale_tile_loops(scale_tile)
    ctx.set_tile_shift_point_loops(shift_point)
    return node


def tile_kernel(sizes, node):
    kernel_id = find_kernel_id(node)
    if kernel_id is not None:
        node = node.child(0)
        node = tile_band(node, get_sizes(sizes, "tile", kernel_id))
        node = node.parent()
        return node

    n = node.n_children()

    for i in xrange(n):
        node = node.child(i)
        node = tile_kernel(sizes, node)
        node = node.parent()

    return node


def tile_kernels(schedule, sizes):
    return tile_kernel(sizes, schedule.get_root()).get_schedule()
