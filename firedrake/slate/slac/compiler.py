"""This is Slate's Linear Algebra Compiler. This module is
responsible for generating C++ kernel functions representing
symbolic linear algebra expressions written in Slate.

This linear algebra compiler uses both Firedrake's form compiler,
the Two-Stage Form Compiler (TSFC) and COFFEE's kernel abstract
syntax tree (AST) optimizer. TSFC provides this compiler with
appropriate kernel functions (in C) for evaluating integral
expressions (finite element variational forms written in UFL).
COFFEE's AST base helps with the construction of code blocks
throughout the kernel returned by: `compile_expression`.

The Eigen C++ library (http://eigen.tuxfamily.org/) is required, as
all low-level numerical linear algebra operations are performed using
this templated function library.
"""
from coffee import base as ast

from firedrake.constant import Constant
from firedrake.tsfc_interface import SplitKernel, KernelInfo
from firedrake.slate.slac.kernel_builder import LocalKernelBuilder
from firedrake import op2

from itertools import chain

from pyop2.utils import get_petsc_dir
from pyop2.datatypes import as_cstr

from tsfc.parameters import SCALAR_TYPE

import firedrake.slate.slate as slate
import numpy as np


__all__ = ['compile_expression']


PETSC_DIR = get_petsc_dir()

cell_to_facets_dtype = np.dtype(np.int8)


def compile_expression(slate_expr, tsfc_parameters=None):
    """Takes a Slate expression `slate_expr` and returns the appropriate
    :class:`firedrake.op2.Kernel` object representing the Slate expression.

    :arg slate_expr: a :class:'TensorBase' expression.
    :arg tsfc_parameters: an optional `dict` of form compiler parameters to
                          be passed onto TSFC during the compilation of
                          ufl forms.

    Returns: A `tuple` containing a `SplitKernel(idx, kinfo)`
    """
    if not isinstance(slate_expr, slate.TensorBase):
        raise ValueError("Expecting a `TensorBase` object, not %s" % type(slate_expr))

    # TODO: Get PyOP2 to write into mixed dats
    if slate_expr.is_mixed:
        raise NotImplementedError("Compiling mixed slate expressions")

    if len(slate_expr.ufl_domains()) > 1:
        raise NotImplementedError("Multiple domains not implemented.")

    # If the expression has already been symbolically compiled, then
    # simply reuse the produced kernel.
    if slate_expr._metakernel_cache is not None:
        return slate_expr._metakernel_cache

    # Create a builder for the Slate expression
    builder = LocalKernelBuilder(expression=slate_expr,
                                 tsfc_parameters=tsfc_parameters)

    # Keep track of declared temporaries
    declared_temps = {}
    statements = []

    # Declare terminal tensor temporaries
    terminal_declarations = terminal_temporaries(builder, declared_temps)
    statements.extend(terminal_declarations)

    # Generate assembly calls for tensor assembly
    subkernel_calls = tensor_assembly_calls(builder)
    statements.extend(subkernel_calls)

    # Create coefficient temporaries if necessary
    if builder.coefficient_vecs:
        coefficient_temps = coefficient_temporaries(builder, declared_temps)
        statements.extend(coefficient_temps)

    # Create auxiliary temporaries if necessary
    if builder.aux_exprs:
        aux_temps = auxiliary_temporaries(builder, declared_temps)
        statements.extend(aux_temps)

    # Generate the kernel information with complete AST
    kinfo = generate_kernel_ast(builder, statements, declared_temps)

    # Cache the resulting kernel
    idx = tuple([0]*slate_expr.rank)
    kernel = (SplitKernel(idx, kinfo),)
    slate_expr._metakernel_cache = kernel

    return kernel


def generate_kernel_ast(builder, statements, declared_temps):
    """Glues together the complete AST for the Slate expression
    contained in the :class:`LocalKernelBuilder`.

    :arg builder: The :class:`LocalKernelBuilder` containing
                  all relevant expression information.
    :arg statements: A list of COFFEE objects containing all
                     assembly calls and temporary declarations.
    :arg declared_temps: A `dict` containing all previously
                         declared temporaries.

    Return: A `KernelInfo` object describing the complete AST.
    """
    slate_expr = builder.expression
    if slate_expr.rank == 0:
        # Scalars are treated as 1x1 MatrixBase objects
        shape = (1,)
    else:
        shape = slate_expr.shape

    # Now we create the result statement by declaring its eigen type and
    # using Eigen::Map to move between Eigen and C data structs.
    statements.append(ast.FlatBlock("/* Map eigen tensor into C struct */\n"))
    result_sym = ast.Symbol("T%d" % len(declared_temps))
    result_data_sym = ast.Symbol("A%d" % len(declared_temps))
    result_type = "Eigen::Map<%s >" % eigen_matrixbase_type(shape)
    result = ast.Decl(SCALAR_TYPE, ast.Symbol(result_data_sym, shape))
    result_statement = ast.FlatBlock("%s %s((%s *)%s);\n" % (result_type,
                                                             result_sym,
                                                             SCALAR_TYPE,
                                                             result_data_sym))
    statements.append(result_statement)

    # Generate the complete c++ string performing the linear algebra operations
    # on Eigen matrices/vectors
    statements.append(ast.FlatBlock("/* Linear algebra expression */\n"))
    cpp_string = ast.FlatBlock(metaphrase_slate_to_cpp(slate_expr,
                                                       declared_temps))
    statements.append(ast.Incr(result_sym, cpp_string))

    # Generate arguments for the macro kernel
    args = [result, ast.Decl("%s **" % SCALAR_TYPE, builder.coord_sym)]

    # Orientation information
    if builder.oriented:
        args.append(ast.Decl("int **", builder.cell_orientations_sym))

    # Coefficient information
    expr_coeffs = slate_expr.coefficients()
    for c in expr_coeffs:
        if isinstance(c, Constant):
            ctype = "%s *" % SCALAR_TYPE
        else:
            ctype = "%s **" % SCALAR_TYPE
        args.extend([ast.Decl(ctype, csym) for csym in builder.coefficient(c)])

    # Facet information
    if builder.needs_cell_facets:
        args.append(ast.Decl("%s *" % as_cstr(cell_to_facets_dtype),
                             builder.cell_facet_sym))

    # NOTE: We need to be careful about the ordering here. Mesh layers are
    # added as the final argument to the kernel.
    if builder.needs_mesh_layers:
        args.append(ast.Decl("int", builder.mesh_layer_sym))

    # Macro kernel
    macro_kernel_name = "compile_slate"
    stmts = ast.Block(statements)
    macro_kernel = ast.FunDecl("void", macro_kernel_name, args,
                               stmts, pred=["static", "inline"])

    # Construct the final ast
    kernel_ast = ast.Node(builder.templated_subkernels + [macro_kernel])

    # Now we wrap up the kernel ast as a PyOP2 kernel and include the
    # Eigen header files
    include_dirs = builder.include_dirs
    include_dirs.extend(["%s/include/eigen3/" % d for d in PETSC_DIR])
    op2kernel = op2.Kernel(kernel_ast,
                           macro_kernel_name,
                           cpp=True,
                           include_dirs=include_dirs,
                           headers=['#include <Eigen/Dense>',
                                    '#define restrict __restrict'])

    # Send back a "TSFC-like" SplitKernel object with an
    # index and KernelInfo
    kinfo = KernelInfo(kernel=op2kernel,
                       integral_type=builder.integral_type,
                       oriented=builder.oriented,
                       subdomain_id="otherwise",
                       domain_number=0,
                       coefficient_map=tuple(range(len(expr_coeffs))),
                       needs_cell_facets=builder.needs_cell_facets,
                       pass_layer_arg=builder.needs_mesh_layers)

    return kinfo


def auxiliary_temporaries(builder, declared_temps):
    """Generates statements for assigning auxiliary temporaries
    for nodes in an expression with "high" reference count.
    Expressions which require additional temporaries are provided
    by the :class:`LocalKernelBuilder`.

    :arg builder: The :class:`LocalKernelBuilder` containing
                  all relevant expression information.
    :arg declared_temps: A `dict` containing all previously
                         declared temporaries. This dictionary
                         is updated as auxiliary expressions
                         are assigned temporaries.
    """
    statements = [ast.FlatBlock("/* Auxiliary temporaries */\n")]
    results = [ast.FlatBlock("/* Assign auxiliary temps */\n")]
    for exp in builder.aux_exprs:
        if exp not in declared_temps:
            t = ast.Symbol("auxT%d" % len(declared_temps))
            result = metaphrase_slate_to_cpp(exp, declared_temps)
            tensor_type = eigen_matrixbase_type(shape=exp.shape)
            statements.append(ast.Decl(tensor_type, t))
            statements.append(ast.FlatBlock("%s.setZero();\n" % t))
            results.append(ast.Assign(t, result))
            declared_temps[exp] = t

    statements.extend(results)

    return statements


def coefficient_temporaries(builder, declared_temps):
    """Generates coefficient temporary statements for assigning
    coefficients to vector temporaries.

    :arg builder: The :class:`LocalKernelBuilder` containing
                  all relevant expression information.
    :arg declared_temps: A `dict` keeping track of all declared
                         temporaries. This dictionary is updated
                         as coefficients are assigned temporaries.

    'AssembledVector's require creating coefficient temporaries to
    store data. The temporaries are created by inspecting the function
    space of the coefficient to compute node and dof extents. The
    coefficient is then assigned values by looping over both the node
    extent and dof extent (double FOR-loop). A double FOR-loop is needed
    for each function space (if the function space is mixed, then a loop
    will be constructed for each component space). The general structure
    of each coefficient loop will be:

         FOR (i1=0; i1<node_extent; i1++):
             FOR (j1=0; j1<dof_extent; j1++):
                 VT0[offset + (dof_extent * i1) + j1] = w_0_0[i1][j1]
                 VT1[offset + (dof_extent * i1) + j1] = w_1_0[i1][j1]
                 .
                 .
                 .

    where wT0, wT1, ... are temporaries for coefficients sharing the
    same node and dof extents. The offset is computed based on whether
    the function space is mixed. The offset is always 0 for non-mixed
    coefficients. If the coefficient is mixed, then the offset is
    incremented by the total number of nodal unknowns associated with
    the component spaces of the mixed space.
    """
    statements = [ast.FlatBlock("/* Coefficient temporaries */\n")]
    i_sym = ast.Symbol("i1")
    j_sym = ast.Symbol("j1")
    loops = [ast.FlatBlock("/* Loops for coefficient temps */\n")]
    for (nodes, dofs), cinfo_list in builder.coefficient_vecs.items():
        # Collect all coefficients which share the same node/dof extent
        assignments = []
        for cinfo in cinfo_list:
            fs_i = cinfo.space_index
            offset = cinfo.offset_index
            c_shape = cinfo.shape
            vector = cinfo.vector
            function = vector._function

            if vector not in declared_temps:
                # Declare and initialize coefficient temporary
                c_type = eigen_matrixbase_type(shape=c_shape)
                t = ast.Symbol("VT%d" % len(declared_temps))
                statements.append(ast.Decl(c_type, t))
                statements.append(ast.FlatBlock("%s.setZero();\n" % t))
                declared_temps[vector] = t

            # Assigning coefficient values into temporary
            coeff_sym = ast.Symbol(builder.coefficient(function)[fs_i],
                                   rank=(i_sym, j_sym))
            index = ast.Sum(offset,
                            ast.Sum(ast.Prod(dofs, i_sym), j_sym))
            coeff_temp = ast.Symbol(t, rank=(index,))
            assignments.append(ast.Assign(coeff_temp, coeff_sym))

        # Inner-loop running over dof extent
        inner_loop = ast.For(ast.Decl("unsigned int", j_sym, init=0),
                             ast.Less(j_sym, dofs),
                             ast.Incr(j_sym, 1),
                             assignments)

        # Outer-loop running over node extent
        loop = ast.For(ast.Decl("unsigned int", i_sym, init=0),
                       ast.Less(i_sym, nodes),
                       ast.Incr(i_sym, 1),
                       inner_loop)

        loops.append(loop)

    statements.extend(loops)

    return statements


def tensor_assembly_calls(builder):
    """Generates a block of statements for assembling the local
    finite element tensors.

    :arg builder: The :class:`LocalKernelBuilder` containing
                  all relevant expression information and
                  assembly calls.
    """
    statements = [ast.FlatBlock("/* Assemble local tensors */\n")]

    # Cell integrals are straightforward. Just splat them out.
    statements.extend(builder.assembly_calls["cell"])

    if builder.needs_cell_facets:
        # The for-loop will have the general structure:
        #
        #    FOR (facet=0; facet<num_facets; facet++):
        #        IF (facet is interior):
        #            *interior calls
        #        ELSE IF (facet is exterior):
        #            *exterior calls
        #
        # If only interior (exterior) facets are present,
        # then only a single IF-statement checking for interior
        # (exterior) facets will be present within the loop. The
        # cell facets are labelled `1` for interior, and `0` for
        # exterior.

        statements.append(ast.FlatBlock("/* Loop over cell facets */\n"))
        int_calls = list(chain(*[builder.assembly_calls[it_type]
                                 for it_type in ("interior_facet",
                                                 "interior_facet_vert")]))
        ext_calls = list(chain(*[builder.assembly_calls[it_type]
                                 for it_type in ("exterior_facet",
                                                 "exterior_facet_vert")]))

        # Compute the number of facets to loop over
        domain = builder.expression.ufl_domain()
        if domain.cell_set._extruded:
            num_facets = domain.ufl_cell()._cells[0].num_facets()
        else:
            num_facets = domain.ufl_cell().num_facets()

        if_ext = ast.Eq(ast.Symbol(builder.cell_facet_sym,
                                   rank=(builder.it_sym,)), 0)
        if_int = ast.Eq(ast.Symbol(builder.cell_facet_sym,
                                   rank=(builder.it_sym,)), 1)
        body = []
        if ext_calls:
            body.append(ast.If(if_ext, (ast.Block(ext_calls,
                                                  open_scope=True),)))
        if int_calls:
            body.append(ast.If(if_int, (ast.Block(int_calls,
                                                  open_scope=True),)))

        statements.append(ast.For(ast.Decl("unsigned int",
                                           builder.it_sym, init=0),
                                  ast.Less(builder.it_sym, num_facets),
                                  ast.Incr(builder.it_sym, 1), body))

    if builder.needs_mesh_layers:
        # In the presence of interior horizontal facet calls, an
        # IF-ELIF-ELSE block is generated using the mesh levels
        # as conditions for which calls are needed:
        #
        #    IF (layer == bottom_layer):
        #        *bottom calls
        #    ELSE IF (layer == top_layer):
        #        *top calls
        #    ELSE:
        #        *top calls
        #        *bottom calls
        #
        # Any extruded top or bottom calls for extruded facets are
        # included within the appropriate mesh-level IF-blocks. If
        # no interior horizontal facet calls are present, then
        # standard IF-blocks are generated for exterior top/bottom
        # facet calls when appropriate:
        #
        #    IF (layer == bottom_layer):
        #        *bottom calls
        #
        #    IF (layer == top_layer):
        #        *top calls
        #
        # The mesh level is an integer provided as a macro kernel
        # argument.

        # FIXME: No variable layers assumption
        statements.append(ast.FlatBlock("/* Mesh levels: */\n"))
        num_layers = builder.expression.ufl_domain().topological.layers - 1
        int_top = builder.assembly_calls["interior_facet_horiz_top"]
        int_btm = builder.assembly_calls["interior_facet_horiz_bottom"]
        ext_top = builder.assembly_calls["exterior_facet_top"]
        ext_btm = builder.assembly_calls["exterior_facet_bottom"]

        bottom = ast.Block(int_top + ext_btm, open_scope=True)
        top = ast.Block(int_btm + ext_top, open_scope=True)
        rest = ast.Block(int_btm + int_top, open_scope=True)
        statements.append(ast.If(ast.Eq(builder.mesh_layer_sym, 0),
                                 (bottom,
                                  ast.If(ast.Eq(builder.mesh_layer_sym,
                                                num_layers - 1),
                                         (top, rest)))))

    return statements


def terminal_temporaries(builder, declared_temps):
    """Generates statements for assigning auxiliary temporaries
    for nodes in an expression with "high" reference count.
    Expressions which require additional temporaries are provided
    by the :class:`LocalKernelBuilder`.

    :arg builder: The :class:`LocalKernelBuilder` containing
                  all relevant expression information.
    :arg declared_temps: A `dict` keeping track of all declared
                         temporaries. This dictionary is updated
                         as terminal tensors are assigned temporaries.
    """
    statements = [ast.FlatBlock("/* Declare and initialize */\n")]
    for exp in builder.temps:
        t = builder.temps[exp]
        statements.append(ast.Decl(eigen_matrixbase_type(exp.shape), t))
        statements.append(ast.FlatBlock("%s.setZero();\n" % t))
        declared_temps[exp] = t

    return statements


def parenthesize(arg, prec=None, parent=None):
    """Parenthesizes an expression."""
    if prec is None or parent is None or prec >= parent:
        return arg
    return "(%s)" % arg


def metaphrase_slate_to_cpp(expr, temps, prec=None):
    """Translates a Slate expression into its equivalent representation in
    the Eigen C++ syntax.

    :arg expr: a :class:`slate.TensorBase` expression.
    :arg temps: a `dict` of temporaries which map a given expression to its
                corresponding representation as a `coffee.Symbol` object.
    :arg prec: an argument dictating the order of precedence in the linear
               algebra operations. This ensures that parentheticals are placed
               appropriately and the order in which linear algebra operations
               are performed are correct.

    Returns
        This function returns a `string` which represents the C/C++ code
        representation of the `slate.TensorBase` expr.
    """
    # If the tensor is terminal, it has already been declared.
    # Coefficients defined as AssembledVectors will have been declared
    # by now, as well as any other nodes with high reference count.
    if expr in temps:
        return temps[expr].gencode()

    elif isinstance(expr, slate.Transpose):
        tensor, = expr.operands
        return "(%s).transpose()" % metaphrase_slate_to_cpp(tensor, temps)

    elif isinstance(expr, slate.Inverse):
        tensor, = expr.operands
        return "(%s).inverse()" % metaphrase_slate_to_cpp(tensor, temps)

    elif isinstance(expr, slate.Negative):
        tensor, = expr.operands
        result = "-%s" % metaphrase_slate_to_cpp(tensor, temps, expr.prec)
        return parenthesize(result, expr.prec, prec)

    elif isinstance(expr, (slate.Add, slate.Mul)):
        op = {slate.Add: '+',
              slate.Mul: '*'}[type(expr)]
        A, B = expr.operands
        result = "%s %s %s" % (metaphrase_slate_to_cpp(A, temps, expr.prec),
                               op,
                               metaphrase_slate_to_cpp(B, temps, expr.prec))

        return parenthesize(result, expr.prec, prec)

    else:
        raise NotImplementedError("Type %s not supported.", type(expr))


def eigen_matrixbase_type(shape):
    """Returns the Eigen::Matrix declaration of the tensor.

    :arg shape: a tuple of integers the denote the shape of the
                :class:`slate.TensorBase` object.

    Returns: Returns a string indicating the appropriate declaration of the
             `slate.TensorBase` object in the appropriate Eigen C++ template
             library syntax.
    """
    if len(shape) == 0:
        rows = 1
        cols = 1
    elif len(shape) == 1:
        rows = shape[0]
        cols = 1
    else:
        if not len(shape) == 2:
            raise NotImplementedError(
                "%d-rank tensors are not supported." % len(shape)
            )
        rows = shape[0]
        cols = shape[1]
    if cols != 1:
        order = ", Eigen::RowMajor"
    else:
        order = ""

    return "Eigen::Matrix<double, %d, %d%s>" % (rows, cols, order)
