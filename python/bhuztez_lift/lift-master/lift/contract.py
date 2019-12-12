import islpy as isl
from .isl_utils import to_sets, to_maps


def rewrite_stmt(expr, stmt_map):
    if expr[0] == 'var':
        return ('var', stmt_map.apply_range(expr[1]))
    elif expr[0] == 'const':
        return expr
    elif expr[0] == 'call':
        return (expr[0],expr[1],tuple(rewrite_stmt(e,stmt_map) for e in expr[2]))
    else:
        raise NotImplementedError


def subst_stmt(expr, v, stmt_map, stmt, stmt_domain):
    if expr[0] == 'var':
        if expr[1].get_tuple_name(isl.dim_type.out) != v:
            return ('var',expr[1].intersect_domain(stmt_domain))

        return rewrite_stmt(stmt, stmt_map)
    elif expr[0] == 'const':
        return expr
    elif expr[0] == 'call':
        return (expr[0],expr[1],tuple(subst_stmt(e,v,stmt_map,stmt,stmt_domain) for e in expr[2]))
    else:
        raise NotImplementedError


def subst_var(expr, var_map, stmt_domain):
    if expr[0] == 'var':
        var1 = expr[1].intersect_domain(stmt_domain)
        if not var1.range().is_subset(var_map.domain()):
            assert isl.UnionSet.from_set(var1.range()).intersect(var_map.domain()).is_empty()
            return ('var', var1)

        return ('var', var1.apply_range(var_map))
    elif expr[0] == 'const':
        return expr
    elif expr[0] == 'call':
        return (expr[0],expr[1],tuple(subst_var(e,var_map,stmt_domain) for e in expr[2]))
    else:
        raise NotImplementedError


def split_stmt_domains(stmt_maps):
    domains = []

    for stmt_domain in to_sets(stmt_maps.domain()):
        subdomains = [stmt_domain]

        for stmt_map in to_maps(stmt_maps.intersect_domain(stmt_domain)):
            subdomain = stmt_map.domain()
            subdomains = (
                [ d.intersect(subdomain) for d in subdomains ] +
                [ d.subtract(subdomain) for d in subdomains ])
            subdomains = [d for d in subdomains if not d.is_empty()]

        domains += subdomains

    return domains


def subst_const(expr, v, c, stmt_domain):
    if expr[0] == 'var':
        if expr[1].get_tuple_name(isl.dim_type.out) != v:
            return ('var',expr[1].intersect_domain(stmt_domain))

        return ('const', c)
    elif expr[0] == 'const':
        return expr
    elif expr[0] == 'call':
        return (expr[0],expr[1],tuple(subst_const(e,v,c,stmt_domain) for e in expr[2]))
    else:
        raise NotImplementedError


def subst_const_stmt(expr, v, c, stmt_domain):
    if expr[0] == 'var':
        var1 = expr[1].intersect_domain(stmt_domain)
        if not var1.range().is_subset(v):
            assert isl.UnionSet.from_set(var1.range()).intersect(v).is_empty()
            return ('var', var1)
        return ('const', c)
    elif expr[0] == 'const':
        return expr
    elif expr[0] == 'call':
        return (expr[0],expr[1],tuple(subst_const_stmt(e,v,c,stmt_domain) for e in expr[2]))
    else:
        raise NotImplementedError


def contract_single_use_reduction_arrays(ctx):
    def find_contractible():
        def_arrays = ctx.def_stmts.get_assign_map().range()
        update_map = ctx.update_stmts.get_assign_map()

        use_map = ctx.get_use_map()

        for k, v in ctx.intermediate_arrays.items():
            if not v.intersect(def_arrays).is_empty():
                continue

            uses = use_map.intersect_range(v)

            if uses.is_empty():
                continue

            if not uses.is_injective():
                continue

            ops = set(ctx.update_stmts[stmt_map.get_tuple_name(isl.dim_type.in_)][1][1]
                      for stmt_map in to_maps(update_map.intersect_range(uses.range())))

            if len(ops) > 1:
                continue

            op = ops.pop()

            for stmt_map in to_maps(uses):
                assert not stmt_map.is_empty()

                name = stmt_map.get_tuple_name(isl.dim_type.in_)
                stmt = ctx.def_stmts.get(name, ctx.update_stmts.get(name, None))
                assert stmt is not None

                if stmt[1][0] == 'var':
                    continue
                elif stmt[1][0] == 'call':
                    if stmt[1][1] != op:
                        break

                    if stmt[1][2][0][0] == 'var':
                        if stmt_map.is_subset(stmt[1][2][0][1]):
                            continue

                    if stmt[1][2][1][0] == 'var':
                        if stmt_map.is_subset(stmt[1][2][1][1]):
                            continue

                    break
                else:
                    raise NotImplementedError
            else:
                return k, uses

    while True:
        init_map= ctx.init_stmts.get_assign_map()
        fini_map = ctx.fini_stmts.get_assign_map()
        updated_by = ctx.update_stmts.get_assign_map().reverse()
        contractible = find_contractible()
        if contractible is None:
            return

        k, uses = contractible

        for domain in to_sets(uses.domain()):
            m, = to_maps(uses.intersect_domain(domain))

            for stmt_domain in split_stmt_domains(m.apply_range(updated_by)):
                name = stmt_domain.get_tuple_name()
                stmt = ctx.def_stmts.get(name, ctx.update_stmts.get(name, None))
                assert stmt is not None

                stmt_use_map = m.intersect_domain(stmt_domain)
                var_map = stmt_use_map.reverse().apply_range(stmt[0])
                var_domain = var_map.domain()

                if stmt[1][0] == 'var':
                    ctx.def_stmts.subtract(stmt_domain)

                    init_domain = to_maps(init_map.intersect_range(var_domain))[0].domain()
                    init_stmt = ctx.init_stmts[init_domain.get_tuple_name()]
                    ctx.init_stmts.add((init_stmt[0].intersect_domain(init_domain).apply_range(var_map), init_stmt[1]))
                    ctx.init_stmts.subtract(init_domain)

                    for stmt_map in to_maps(stmt_use_map.apply_range(updated_by)):
                        update_domain = stmt_map.range()
                        update_stmt = ctx.update_stmts[update_domain.get_tuple_name()]
                        ctx.update_stmts.add(
                            (update_stmt[0].intersect_domain(update_domain).apply_range(var_map),
                             subst_var(update_stmt[1], var_map, update_domain)))
                        ctx.update_stmts.subtract(update_domain)

                    fini_domain = to_maps(fini_map.intersect_range(var_domain))[0].domain()
                    fini_stmt = ctx.fini_stmts[fini_domain.get_tuple_name()]
                    ctx.fini_stmts.add(
                        (fini_stmt[0].intersect_domain(fini_domain).apply_range(var_map),
                         ('var',fini_stmt[0].intersect_domain(fini_domain).apply_range(var_map))))
                    ctx.fini_stmts.subtract(fini_domain)
                elif name in ctx.def_stmts:
                    ctx.def_stmts.subtract(stmt_domain)

                    init_domain = to_maps(init_map.intersect_range(var_domain))[0].domain()
                    init_stmt = ctx.init_stmts[init_domain.get_tuple_name()]
                    ctx.init_stmts.add((init_stmt[0].intersect_domain(init_domain).apply_range(var_map), init_stmt[1]))
                    ctx.init_stmts.subtract(init_domain)

                    for arg in stmt[1][2]:
                        if arg[0] == 'var' and stmt_use_map.is_subset(arg[1]):
                            for stmt_map in to_maps(stmt_use_map.apply_range(updated_by)):
                                update_domain = stmt_map.range()
                                update_stmt = ctx.update_stmts[update_domain.get_tuple_name()]
                                ctx.update_stmts.add(
                                    (update_stmt[0].intersect_domain(update_domain).apply_range(var_map),
                                     subst_var(update_stmt[1], var_map, update_domain)))
                                ctx.update_stmts.subtract(update_domain)
                        else:
                            ctx.update_stmts.add(
                                (stmt[0].intersect_domain(stmt_domain),
                                 ('call', stmt[1][1],
                                  (('var',stmt[0].intersect_domain(stmt_domain)),
                                   ('var',arg[1].intersect_domain(stmt_domain))))))

                    fini_domain = to_maps(fini_map.intersect_range(var_domain))[0].domain()
                    fini_stmt = ctx.fini_stmts[fini_domain.get_tuple_name()]
                    ctx.fini_stmts.add(
                        (fini_stmt[0].intersect_domain(fini_domain).apply_range(var_map),
                         ('var',fini_stmt[0].intersect_domain(fini_domain).apply_range(var_map))))
                    ctx.fini_stmts.subtract(fini_domain)
                elif name in ctx.update_stmts:
                    ctx.update_stmts.subtract(stmt_domain)

                    for stmt_map in to_maps(stmt_use_map.apply_range(updated_by)):
                        update_domain = stmt_map.range()
                        update_stmt = ctx.update_stmts[update_domain.get_tuple_name()]
                        ctx.update_stmts.add(
                            (update_stmt[0].intersect_domain(update_domain).apply_range(var_map),
                             subst_var(update_stmt[1], var_map, update_domain)))
                        ctx.update_stmts.subtract(update_domain)

                    init_domain = to_maps(init_map.intersect_range(var_domain))[0].domain()
                    ctx.init_stmts.subtract(init_domain)

                    fini_domain = to_maps(fini_map.intersect_range(var_domain))[0].domain()
                    ctx.fini_stmts.subtract(fini_domain)
                else:
                    assert False

        del ctx.intermediate_arrays[k]


def contract_single_use_def_arrays(ctx):
    def find_contractible():
        reduction_arrays = ctx.fini_stmts.get_assign_map().range()
        use_map = ctx.get_use_map()

        for k, v in ctx.intermediate_arrays.items():
            if not v.intersect(reduction_arrays).is_empty():
                continue

            uses = use_map.intersect_range(v)
            if not uses.is_injective():
                continue

            return k, uses

    while True:
        defined_by = ctx.def_stmts.get_assign_map().reverse()
        contractible = find_contractible()
        if contractible is None:
            return

        k, uses = contractible

        for domain in to_sets(uses.domain()):
            m, = to_maps(uses.intersect_domain(domain))
            maps = to_maps(m.apply_range(defined_by))

            for stmt_map in maps:
                assert not stmt_map.is_empty()

                stmt_domain = stmt_map.domain()
                name = stmt_domain.get_tuple_name()
                stmt = ctx.def_stmts.get(name, ctx.update_stmts.get(name, None))

                name2 = stmt_map.get_tuple_name(isl.dim_type.out)

                new_stmt = (
                    stmt[0].intersect_domain(stmt_domain),
                    subst_stmt(stmt[1], k, stmt_map, ctx.def_stmts[name2][1], stmt_domain))

                if name in ctx.def_stmts:
                    ctx.def_stmts.add(new_stmt)
                    ctx.def_stmts.subtract(stmt_domain)
                    ctx.def_stmts.subtract(stmt_map.range())
                elif name in ctx.update_stmts:
                    ctx.update_stmts.add(new_stmt)
                    ctx.update_stmts.subtract(stmt_domain)
                    ctx.def_stmts.subtract(stmt_map.range())
                else:
                    raise NotImplementedError

        del ctx.intermediate_arrays[k]


def remove_dead_stmts(ctx):
    used_arrays = ctx.intermediate_arrays.keys() + ctx.output_arrays.keys()
    for k, v in ctx.def_stmts.items():
        if v[0].get_tuple_name(isl.dim_type.out) not in used_arrays:
            del ctx.def_stmts[k]
    for k, v in ctx.init_stmts.items():
        if v[0].get_tuple_name(isl.dim_type.out) not in used_arrays:
            del ctx.init_stmts[k]
    for k, v in ctx.update_stmts.items():
        if v[0].get_tuple_name(isl.dim_type.out) not in used_arrays:
            del ctx.update_stmts[k]
    for k, v in ctx.fini_stmts.items():
        if v[0].get_tuple_name(isl.dim_type.out) not in used_arrays:
            del ctx.fini_stmts[k]


def convert_single_update_reduction_arrays(ctx):
    init_map = ctx.init_stmts.get_assign_map()
    update_map = ctx.update_stmts.get_assign_map()
    fini_map = ctx.fini_stmts.get_assign_map()

    arrays = {}
    arrays.update(ctx.intermediate_arrays)
    arrays.update(ctx.output_arrays)

    for k, array_range in arrays.items():
        updates = update_map.intersect_range(array_range)
        if not updates.is_injective():
            continue

        for stmt_map in to_maps(updates):
            assert not stmt_map.is_empty()

            stmt_domain = stmt_map.domain()
            name = stmt_domain.get_tuple_name()
            stmt = ctx.update_stmts[name]

            array_range = array_range.subtract(stmt_map.range())
            ctx.def_stmts.add((stmt[0], stmt[1][2][1]))

            init_domain, = to_sets(init_map.intersect_range(stmt_map.range()).domain())
            fini_domain, = to_sets(fini_map.intersect_range(stmt_map.range()).domain())

            ctx.init_stmts.subtract(init_domain)
            ctx.update_stmts.subtract(stmt_domain)
            ctx.fini_stmts.subtract(fini_domain)

        if not array_range.is_empty():
            for stmt_map in to_maps(init_map.intersect_range(array_range)):
                assert not stmt_map.is_empty()

                stmt_domain = stmt_map.domain()
                name = stmt_domain.get_tuple_name()
                stmt = ctx.init_stmts[name]
                ctx.def_stmts.add(stmt)

                fini_domain, = to_sets(fini_map.intersect_range(stmt_map.range()).domain())
                ctx.init_stmts.subtract(stmt_domain)
                ctx.fini_stmts.subtract(fini_domain)


def propagate_const_values(ctx):
    for k, v in ctx.const_values.items():
        if len(v) > 1:
            continue

        c = ctx.const_arrays[k]
        for stmt_domain in to_sets(ctx.get_use_map().intersect_range(c).domain()):
            name = stmt_domain.get_tuple_name()
            stmt = ctx.def_stmts.get(name, ctx.update_stmts.get(name, None))
            assert stmt is not None
            new_stmt = (stmt[0], subst_const(stmt[1], k, v[0], stmt_domain))

            if name in ctx.def_stmts:
                ctx.def_stmts[name] = new_stmt
            elif name in ctx.update_stmts:
                ctx.update_stmts[name] = new_stmt
            else:
                assert False

        del ctx.const_arrays[k]

    ctx.const_values = {k:v for k,v in ctx.const_values.items() if len(v) > 1}


def simplify_stmt(expr):
    if expr[0] == 'var':
        return expr
    elif expr[0] == 'const':
        return expr
    elif expr[0] == 'call':
        if expr[1] == '+':
            x = simplify_stmt(expr[2][0])
            y = simplify_stmt(expr[2][1])
            if x == ('const', 0.0):
                return y
            elif y == ('const', 0.0):
                return x
            elif x[0] == 'const' and y[0] == 'const':
                return ('const', x[1]+y[1])
            else:
                return ('call', '+', (x,y))
        elif expr[1] == '-':
            x = simplify_stmt(expr[2][0])
            y = simplify_stmt(expr[2][1])
            if y == ('const', 0.0):
                return x
            elif x[0] == 'const' and y[0] == 'const':
                return ('const', x[1]-y[1])
            else:
                return ('call', '-', (x,y))
        elif expr[1] == '*':
            x = simplify_stmt(expr[2][0])
            y = simplify_stmt(expr[2][1])
            if x == ('const', 0.0):
                return ('const', 0.0)
            elif y == ('const', 0.0):
                return ('const', 0.0)
            elif x == ('const', 1.0):
                return y
            elif y == ('const', 1.0):
                return x
            elif x[0] == 'const' and y[0] == 'const':
                return ('const', x[1]*y[1])
            else:
                return ('call', '*', (x,y))
        elif expr[1] == '/':
            x = simplify_stmt(expr[2][0])
            y = simplify_stmt(expr[2][1])
            if x == ('const', 0.0):
                return ('const', 0.0)
            elif y == ('const', 1.0):
                return x
            elif x[0] == 'const' and y[0] == 'const':
                return ('const', x[1]/y[1])
            else:
                return ('call', '/', (x,y))
        else:
            return (expr[0],expr[1],tuple(simplify_stmt(e) for e in expr[2]))
    else:
        raise NotImplementedError


def simplify_stmts(ctx):
    ctx.def_stmts.update({k: (v[0],simplify_stmt(v[1])) for k,v in ctx.def_stmts.items()})
    ctx.init_stmts.update({k: (v[0],simplify_stmt(v[1])) for k,v in ctx.init_stmts.items()})
    ctx.update_stmts.update({k: (v[0],simplify_stmt(v[1])) for k,v in ctx.update_stmts.items()})
    ctx.fini_stmts.update({k: (v[0],simplify_stmt(v[1])) for k,v in ctx.fini_stmts.items()})


def propagate_const_stmts(ctx):
    def find_to_propagate():
        for k, v in ctx.def_stmts.items():
            if v[1][0] != 'const':
                continue

            return k, v

    count = 0

    while True:
        to_propagate = find_to_propagate()
        if to_propagate is None:
            return count

        count += 1
        k, v = to_propagate

        for stmt_domain in split_stmt_domains(ctx.get_use_map().intersect_range(v[0].range())):
            name = stmt_domain.get_tuple_name()
            stmt = ctx.def_stmts.get(name, ctx.update_stmts.get(name, None))
            assert stmt is not None

            new_stmt = (
                stmt[0].intersect_domain(stmt_domain),
                subst_const_stmt(stmt[1], v[0].range(), v[1][1], stmt_domain))

            if name in ctx.def_stmts:
                ctx.def_stmts.add(new_stmt)
                ctx.def_stmts.subtract(stmt_domain)
            elif name in ctx.update_stmts:
                ctx.update_stmts.add(new_stmt)
                ctx.update_stmts.subtract(stmt_domain)
            else:
                assert False

        del ctx.def_stmts[k]


def propagate_alias_stmts(ctx):
    def find_to_propagate():
        for k, v in ctx.def_stmts.items():
            if v[1][0] != 'var':
                continue
            name = v[0].get_tuple_name(isl.dim_type.out)
            if name in ctx.output_arrays:
                continue
            return k, v

    while True:
        to_propagate = find_to_propagate()
        if to_propagate is None:
            return

        k, v = to_propagate

        var_map = v[0].reverse().apply_range(v[1][1])

        for stmt_domain in split_stmt_domains(ctx.get_use_map().intersect_range(v[0].range())):
            name = stmt_domain.get_tuple_name()
            stmt = ctx.def_stmts.get(name, ctx.update_stmts.get(name, None))
            assert stmt is not None

            new_stmt = (
                stmt[0].intersect_domain(stmt_domain),
                subst_var(stmt[1], var_map, stmt_domain))

            if name in ctx.def_stmts:
                ctx.def_stmts.add(new_stmt)
                ctx.def_stmts.subtract(stmt_domain)
            elif name in ctx.update_stmts:
                ctx.update_stmts.add(new_stmt)
                ctx.update_stmts.subtract(stmt_domain)
            else:
                assert False

        del ctx.def_stmts[k]


def remove_dummy_update(ctx):
    init_map = ctx.init_stmts.get_assign_map()

    for k, v in ctx.update_stmts.items():
        if v[1][0] == 'var':
            del ctx.update_stmts[k]


def contract_arrays(ctx):
    convert_single_update_reduction_arrays(ctx)
    contract_single_use_reduction_arrays(ctx)
    remove_dead_stmts(ctx)

    convert_single_update_reduction_arrays(ctx)
    contract_single_use_def_arrays(ctx)
    remove_dead_stmts(ctx)

    propagate_const_values(ctx)
    count = 1

    while count > 0:
        simplify_stmts(ctx)
        count = propagate_const_stmts(ctx)

    propagate_alias_stmts(ctx)
    remove_dummy_update(ctx)

    convert_single_update_reduction_arrays(ctx)
    contract_single_use_def_arrays(ctx)
    remove_dead_stmts(ctx)
