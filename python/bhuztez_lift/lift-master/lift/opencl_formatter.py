from .c_formatter import format_ast, format_arguments, format_vars, format_c_header
from .utils import product


OPENCL_KERNEL_TEMPLATE = """__kernel
void
kernel{id}(
{arguments}){{
{local_arrays}{private_arrays}{iterators}
{body}
}}
"""

def format_kernel_iterators(kernel):
    for i in xrange(len(kernel.grid_sizes)):
        yield "int b{0} = get_group_id({0});\n".format(i)

    for i in xrange(len(kernel.block_sizes)):
        yield "int t{0} = get_local_id({0});\n".format(i)


def format_kernel_arguments(table, kernel):
    for v in kernel.arrays:
        yield "__global float %s%s"%(v, "".join("[%d]"%(s,) for s in table.vars[v].shape[::-1]))

def format_kernel_local_arrays(kernel):
    for v, groups in kernel.groups.iteritems():
        for i, group in enumerate(groups):
            yield "__local float shared_%s_%d%s;\n"%(v, i, "".join("[%d]"%(bound.size.get_num_si(),) for bound in group.bounds))


def format_kernel_private_arrays(kernel):
    pass


def format_kernels(table, kernels):
    for kernel_id, kernel in kernels.iteritems():
        yield OPENCL_KERNEL_TEMPLATE.format(
            id = kernel_id,
            arguments = ",\n".join(format_kernel_arguments(table, kernel)),
            iterators = "".join(format_kernel_iterators(kernel)),
            local_arrays = "".join(format_kernel_local_arrays(kernel)),
            private_arrays = "",
            body = format_ast(table, kernel.ast)
        )

OPENCL_TEMPLATE = """#define inf (1.0/0.0)

{kernels}
"""

def format_cl_kernel(table, kernels):
    return OPENCL_TEMPLATE.format(
        kernels="".join(format_kernels(table, kernels))
    )


C_TEMPLATE = """#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define inf (1.0/0.0)
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))
#include "{name}_cl.h"

extern const char _binary_{name}_cl_start[];
extern const char _binary_{name}_cl_end[];

cl_program
{name}_compile(cl_device_id device, cl_context context){{
size_t size = _binary_{name}_cl_end - _binary_{name}_cl_start;
const char *source = _binary_{name}_cl_start;

cl_program program = clCreateProgramWithSource(context, 1, &source, &size, NULL);

cl_int result = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

size_t len;
clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

if(len > 1) {{
char log[len];
clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log, NULL);
fprintf(stderr, "%.*s\\n", (int)len, log);
}}

if (result == CL_SUCCESS)
return program;
exit(1);
}}

void
{name}_run(cl_device_id device, cl_context context, cl_program program, struct {name}_state *s){{
cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
{inputs}{outputs}{vars}{body}
clReleaseCommandQueue(queue);
clReleaseProgram(program);
clReleaseContext(context);
}}

void
{name}(struct {name}_state *s){{
cl_platform_id platform;
cl_device_id device;

if (clGetPlatformIDs(1, &platform, NULL) != CL_SUCCESS) exit(1);
if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL) != CL_SUCCESS) exit(1);

cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

cl_program program = {name}_compile(device, context);
{name}_run(device, context, program, s);
}}
"""

KERNEL_TEMPLATE = """{creates}
{copyins}

{{
cl_kernel kernel = clCreateKernel(program, "kernel{id}", NULL);
{setargs}
size_t global_sizes[] = {{ {global_sizes} }};
size_t block_sizes[] = {{ {block_sizes} }};
clEnqueueNDRangeKernel(queue, kernel, {n}, NULL, global_sizes, block_sizes, 0, NULL, NULL);
clReleaseKernel(kernel);
}}

{copyouts}
{releases}
{finish}
"""

def format_setargs(arrays):
    for i, v in enumerate(arrays):
        yield "clSetKernelArg(kernel, {}, sizeof(cl_mem), (void *)&mem_{});\n".format(i,v)

def format_creates(table, arrays):
    for v in arrays:
        yield "cl_mem mem_{name} = clCreateBuffer(context, CL_MEM_READ_WRITE, {size} * sizeof(float), NULL, NULL);\n".format(
            name = v,
            size = product(table.vars[v].shape))

def format_releases(arrays):
    for v in arrays:
        yield "clReleaseMemObject(mem_{});\n".format(v)

def format_copyins(table, arrays):
    for v in arrays:
        yield "clEnqueueWriteBuffer(queue, mem_{name}, CL_TRUE, 0, {size} * sizeof(float), {name}, 0, NULL, NULL);\n".format(
            name = v,
            size = product(table.vars[v].shape))

def format_copyouts(table, arrays):
    for v in arrays:
        yield "clEnqueueReadBuffer(queue, mem_{name}, CL_TRUE, 0, {size} * sizeof(float), {name}, 0, NULL, NULL);\n".format(
            name = v,
            size = product(table.vars[v].shape))


def format_cl(name, table, ctx, ast, vars, kernels):
    def format_opencl_kernel(kernel_id):
        kernel = kernels[kernel_id]
        return KERNEL_TEMPLATE.format(
            id = kernel_id,
            global_sizes = ", ".join(
                "%d*%d"%(a,b)
                for a,b in zip(kernel.grid_sizes,kernel.block_sizes)),
            block_sizes = ", ".join(str(b) for b in kernel.block_sizes),
            setargs = "".join(format_setargs(kernel.arrays)),
            n = len(kernel.grid_sizes),
            creates = "".join(format_creates(table, kernel.creates)),
            copyins = "".join(format_copyins(table, kernel.copy_ins)),
            copyouts = "".join(format_copyouts(table, kernel.copy_outs)),
            releases = "".join(format_releases(kernel.releases)),
            finish = "clFinish(queue);" if kernel.finish else "",
        )

    return C_TEMPLATE.format(
        name=name,
        inputs = "".join(format_arguments(table, table.inputs)),
        outputs = "".join(format_arguments(table, table.outputs)),
        vars="".join(format_vars(table, vars)),
        body=format_ast(table, ast, format_opencl_kernel)
    )


HEADER_TEMPLATE  = """{header}
#include <CL/opencl.h>

void {name}(struct {name}_state *s);
cl_program {name}_compile(cl_device_id device, cl_context context);
void {name}_run(cl_device_id device, cl_context context, cl_program program, struct {name}_state *s);
"""

def format_cl_header(name, table):
    return HEADER_TEMPLATE.format(
        name=name, header=format_c_header(name, table))
