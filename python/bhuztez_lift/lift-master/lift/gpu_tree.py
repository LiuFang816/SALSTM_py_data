# ported from ppcg

# Copyright 2013      Ecole Normale Superieure
#
# Use of this software is governed by the MIT license
#
# Written by Sven Verdoolaege,
# Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France


import islpy as isl


def is_marked(node, name):
    if node.get_type() != isl.schedule_node_type.mark:
        return False

    mark = node.mark_get_id()
    return mark.get_name() == name


def node_is_core(node, core):
    filter = node.filter_get_filter()
    return not filter.is_disjoint(core)


def core_child(node, core):
    if node.get_type() != isl.schedule_node_type.sequence:
        return node.child(0)

    n = node.n_children()
    for i in xrange(n):
        node = node.child(i)
        if node_is_core(node, core):
            return node.child(0)
        node = node.parent()

    assert False


def gpu_tree_move_down_to(node, core, mark):
    while not is_marked(node, mark):
        node = core_child(node, core)

    return node


def gpu_tree_move_up_to(node, mark):
    while not is_marked(node, mark):
        node = node.parent()
    return node


def gpu_tree_move_down_to_depth(node, depth, core):
    while node.get_schedule_depth() < depth:
        if node.get_type() == isl.schedule_node_type.band:
            node_depth = node.get_schedule_depth()
            node_dim = node.band_n_member()
            if node_depth + node_dim > depth:
                node = node.band_split(depth - node_depth)
        node = core_child(node, core)

    while (not is_marked(node, "shared")
           and not is_marked(node, "thread")
           and node.get_type() != isl.schedule_node_type.band):
        node = core_child(node, core)

    return node

def domain_is_sync(domain):
    if domain.n_set() != 1:
        return False
    name = isl.Set.from_union_set(domain).get_tuple_id().get_name()
    return name.startswith("sync")


def node_is_sync_filter(node):
    if node.get_type() != isl.schedule_node_type.filter:
        return False
    domain = node.filter_get_filter()
    return domain_is_sync(domain)


def has_preceding_sync(node):
    node = node.parent()
    while node.has_previous_sibling():
        node = node.previous_sibling()
        if node_is_sync_filter(node):
            return True
    return False

def has_following_sync(node):
    node = node.parent()
    while node.has_next_sibling():
        node = node.next_sibling()
        if node_is_sync_filter(node):
            return True
    return False


def has_sync_before_core(node, core):
    while not is_marked(node, "thread"):
        node = core_child(node, core)
        if has_preceding_sync(node):
            return True
    return False

def has_sync_after_core(node, core):
    while not is_marked(node, "thread"):
        node = core_child(node, core)
        if has_following_sync(node):
            return True
    return False


def create_sync_domain(ctx, name):
    space = isl.Space.set_alloc(ctx, 0, 0)
    id = isl.Id.alloc(ctx, name, None)
    space = space.set_tuple_id(isl.dim_type.set, id)
    return isl.UnionSet.from_set(isl.Set.universe(space))

def insert_sync_before(node, sync_counter):
    domain = create_sync_domain(node.get_ctx(), sync_counter.next())
    graft = isl.ScheduleNode.from_domain(domain)
    node = node.graft_before(graft)
    return node

def insert_sync_after(node, sync_counter):
    domain = create_sync_domain(node.get_ctx(), sync_counter.next())
    graft = isl.ScheduleNode.from_domain(domain)
    node = node.graft_after(graft)
    return node

def gpu_tree_ensure_preceding_sync(node, sync_counter):
    if has_preceding_sync(node):
        return node

    return insert_sync_before(node, sync_counter)

def gpu_tree_ensure_following_sync(node, sync_counter):
    if has_following_sync(node):
        return node

    return insert_sync_after(node, sync_counter)

def gpu_tree_ensure_sync_after_core(node, core, sync_counter):
    if has_sync_after_core(node, core):
        return node
    if has_following_sync(node):
        return node
    return insert_sync_after(node, sync_counter)


def gpu_tree_move_left_to_sync(node, core, sync_counter):
    if has_sync_before_core(node, core):
        return node

    node = gpu_tree_ensure_preceding_sync(node, sync_counter)
    node = node.parent()
    while not node_is_sync_filter(node):
        node = node.previous_sibling()
    node = node.child(0)
    return node


def gpu_tree_move_right_to_sync(node, core, sync_counter):
    if has_sync_after_core(node, core):
        return node

    node = gpu_tree_ensure_following_sync(node, sync_counter)
    node = node.parent()
    while not node_is_sync_filter(node):
        node = node.next_sibling()
    node = node.child(0)
    return node
