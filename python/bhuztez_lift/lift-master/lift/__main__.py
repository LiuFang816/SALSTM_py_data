import islpy as isl
import argparse

from .parser import Parser
from .check import check_stmts
from .compile import compile
from .contract import contract_arrays
from .codegen import get_schedule_constraints, mark_kernels, remove_kernel_marks, codegen
from .tile import tile_kernels
from .gpu import map_to_device
from .c_formatter import format_c, format_c_header
from .opencl_formatter import format_cl_kernel, format_cl_header, format_cl

p = argparse.ArgumentParser()
p.add_argument('--emit', default='schedule', choices=('schedule','c','opencl'), help='type of output')
p.add_argument('--sizes', required=True, help='per-kernel tile/grid/block sizes')
p.add_argument('--schedule', metavar='FILE', help='use schedule from FILE')
p.add_argument('--dump-schedule', action='store_true', help='dump schedule')
p.add_argument('output', help='name of output')
p.add_argument('filename', nargs='+', help='filename of source file')
args = p.parse_args()

isl_context = isl.Context.alloc()

name = args.output
sizes = isl.UnionMap(args.sizes, isl_context)

stmts = ()

for filename in args.filename:
    p = Parser(filename=filename)

    with open(filename, "r") as f:
        source = f.read()

    stmts += p.parse(source)

table = check_stmts(stmts)
context = compile(table, isl_context)
contract_arrays(context)

isl_context.set_ast_build_detect_min_max(1)
isl_context.set_schedule_maximize_band_depth(1)
isl_context.set_schedule_maximize_coincidence(1)
isl_context.set_schedule_whole_component(0)
isl_context.set_schedule_separate_components(1)
isl_context.set_schedule_treat_coalescing(1)
isl_context.set_schedule_outer_coincidence(1)

if args.schedule is None:
    constraints = get_schedule_constraints(context)
    schedule = constraints.compute_schedule()
    schedule = mark_kernels(schedule)
else:
    with open(args.schedule, 'r') as f:
        schedule = isl.Schedule.read_from_str(ctx, f.read())

if args.dump_schedule:
    schedule.dump()

if args.emit == 'schedule':
    with open(name+".yaml", "w") as f:
        f.write(schedule.to_str())
    exit(0)


schedule = tile_kernels(schedule, sizes)

if args.emit == 'c':
    schedule = remove_kernel_marks(schedule)

    with open(name+"_c.h", "w") as f:
        f.write(format_c_header(name, table))

    ast = codegen(context, schedule)

    with open(name+"_c.c", "w") as f:
        f.write(format_c(name, table, context, ast))

    exit(0)


if args.emit == 'opencl':
    schedule, kernels = map_to_device(context, schedule, sizes)
    ast = codegen(context, schedule, kernels)
    arrays = kernels.pop("arrays")

    with open(name+"_cl.h", "w") as f:
        f.write(format_cl_header(name, table))

    with open(name+".cl", "w") as f:
        f.write(format_cl_kernel(table, kernels))

    with open(name+"_cl.c", "w") as f:
        f.write(format_cl(name, table, context, ast, arrays, kernels))

    exit(0)
