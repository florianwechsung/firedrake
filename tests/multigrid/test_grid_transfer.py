import pytest
import numpy
from firedrake import *
import firedrake
from fractions import Fraction
from firedrake.mg import impl
from firedrake.petsc import *
import numpy

def closure(dm, p):
    return dm.getTransitiveClosure(p, useCone=True)

def bary(cdm):
    rdm = PETSc.DMPlex().create(comm=cdm.comm)

    dim = cdm.getDimension()
    assert dim in [2, 3]
    rdm.setDimension(dim)

    cdm_cells = cdm.getHeightStratum(0)
    cdm_vertices = cdm.getDepthStratum(0)
    cdm_edges = cdm.getDepthStratum(1)

    num_cdm_cells = cdm_cells[1] - cdm_cells[0]
    num_cdm_vertices = cdm_vertices[1] - cdm_vertices[0]
    num_cdm_edges = cdm_edges[1] - cdm_edges[0]

    if dim == 3:
        cdm_facets = cdm.getDepthStratum(2)
        num_cdm_facets = cdm_facets[1] - cdm_facets[0]

    num_rdm_cells = (dim+1)*num_cdm_cells
    num_rdm_vertices = num_cdm_vertices + num_cdm_cells
    num_rdm_edges = num_cdm_edges + (dim+1)*num_cdm_cells
    if dim == 3:
        num_rdm_facets = num_cdm_facets + 6*num_cdm_cells

    rdm_cells = (0, num_rdm_cells)
    rdm_vertices = (rdm_cells[1], rdm_cells[1] + num_rdm_vertices)
    if dim == 3:
        rdm_facets = (rdm_vertices[1], rdm_vertices[1] + num_rdm_facets)
        rdm_edges = (rdm_facets[1], rdm_facets[1] + num_rdm_edges)
    else:
        rdm_edges = (rdm_vertices[1], rdm_vertices[1] + num_rdm_edges)

    rdm.setChart(0, rdm_edges[1])


    for p in range(*rdm_cells):
        rdm.setConeSize(p, dim+1)
    for p in range(*rdm_edges):
        rdm.setConeSize(p, 2)
    if dim == 3:
        for p in range(*rdm_facets):
            rdm.setConeSize(p, dim)

    rdm.setUp()

    for cdm_edge in range(*cdm_edges):
        cone = cdm.getCone(cdm_edge)
        orient = cdm.getConeOrientation(cdm_edge)

        rdm_edge = (cdm_edge - cdm_edges[0]) + rdm_edges[0]
        rdm_cone = [(x - cdm_vertices[0]) + rdm_vertices[0] for x in cone]

        rdm.setCone(rdm_edge, rdm_cone, orientation=orient)
        print("Introducing old edge %d -> %s, orientation=%s" % (rdm_edge, rdm_cone, orient))

    if dim == 3:
        for cdm_facet in range(*cdm_facets):
            cone = cdm.getCone(cdm_facet)
            orient = cdm.getConeOrientation(cdm_facet)

            rdm_facet = (cdm_facet - cdm_facets[0]) + rdm_facets[0]
            rdm_cone = [(x - cdm_edges[0]) + rdm_edges[0] for x in cone]

            rdm.setCone(rdm_facet, rdm_cone, orientation=orient)
            print("Introducing old facet %d -> %s, orientation=%s" % (rdm_facet, rdm_cone, orient))

    for cdm_cell in range(*cdm_cells):
        cdm_cell_entities = closure(cdm, cdm_cell)[0]
        cdm_cell_vertices = [x for x in cdm_cell_entities if cdm_vertices[0] <= x < cdm_vertices[1]]
        cdm_cell_edges = [x for x in cdm_cell_entities if cdm_edges[0] <= x < cdm_edges[1]]
        if dim == 3:
            cdm_cell_facets = [x for x in cdm_cell_entities if cdm_facets[0] <= x < cdm_facets[1]]

        # Step 1.
        # Introduce new vertex at barycentre.
        new_vertex = num_cdm_vertices + cdm_cell + rdm_vertices[0]
        cdm_cell_vertices_in_new_numbering = [old_vertex - cdm_vertices[0] + rdm_vertices[0] for old_vertex in cdm_cell_vertices]
        print("Introducing new vertex %s" % new_vertex)

        # Step 2.
        # Introduce new edges.
        new_edges = list(range(rdm_edges[0] + num_cdm_edges + (dim+1)*cdm_cell,
                               rdm_edges[0] + num_cdm_edges + (dim+1)*(cdm_cell+1)))

        for (old_vertex_in_new_numbering, new_edge) in zip(cdm_cell_vertices_in_new_numbering, new_edges):
            new_edge_cone = [old_vertex_in_new_numbering, new_vertex]
            rdm.setCone(new_edge, new_edge_cone)
            print("Introducing new edge %d -> %s" % (new_edge, new_edge_cone))


        # Step 3.
        # Introduce new facets (in 3D)
        if dim == 3:
            new_facets = list(range(rdm_facets[0] + num_cdm_facets + 6*cdm_cell,
                                    rdm_facets[0] + num_cdm_facets + 6*(cdm_cell+1)))

            for (old_edge, new_facet) in zip(cdm_cell_edges, new_facets):
                # Construct a new facet from the old edge and two new edges connecting
                # those vertices to the barycentre.
                old_edge_in_new_numbering = old_edge - cdm_edges[0] + rdm_edges[0]
                vertices = rdm.getCone(old_edge_in_new_numbering)

                new_facet_cone = [old_edge_in_new_numbering, None, None]
                for new_edge in new_edges:
                    new_vertex = rdm.getCone(new_edge)[0]
                    if new_vertex == vertices[1]:
                        new_facet_cone[1] = new_edge
                    elif new_vertex == vertices[0]:
                        new_facet_cone[2] = new_edge
                new_facet_cone_orientation = [0, 0, -2]

                print("Introducing new facet %d -> %s" % (new_facet, new_facet_cone))
                rdm.setCone(new_facet, new_facet_cone)
                rdm.setConeOrientation(new_facet, new_facet_cone_orientation)

        # Step 4.
        # Introduce new cells.
        new_cells = list(range((dim+1)*cdm_cell, (dim+1)*(cdm_cell+1)))

        # Step 4. Construct new cell cones (and possibly new facet cones, too, in 3D)
        if dim == 2:
            for (old_edge, new_cell) in zip(cdm_cell_edges, new_cells):
                old_edge_in_new_numbering = old_edge - cdm_edges[0] + rdm_edges[0]
                vertices = rdm.getCone(old_edge_in_new_numbering)

                new_cell_cone = [old_edge_in_new_numbering, None, None]
                for new_edge in new_edges:
                    new_vertex = rdm.getCone(new_edge)[0]
                    if new_vertex == vertices[1]:
                        new_cell_cone[1] = new_edge
                    elif new_vertex == vertices[0]:
                        new_cell_cone[2] = new_edge

                new_cell_cone_orientation = [0, 0, -2]
                rdm.setCone(new_cell, new_cell_cone)
                rdm.setConeOrientation(new_cell, new_cell_cone_orientation)
        elif dim == 3:
            for (old_facet, new_cell) in zip(cdm_cell_facets, new_cells):
                old_facet_in_new_numbering = old_facet - cdm_facets[0] + rdm_facets[0]
                old_edges = cdm.getCone(old_facet)

                def get_vertex_walk_for_facet(facet, start):
                    edges = rdm.getCone(facet)
                    edges_orientation = rdm.getConeOrientation(facet)
                    vertex_walk = []
                    if start == 0:
                        ii = [0, 1, 2]
                    elif start == 1:
                        ii = [1, 2, 0]
                    elif start == 2:
                        ii = [2, 0, 1]
                    elif start == -1:
                        ii = [0, 2, 1]
                    elif start == -2:
                        ii = [1, 0, 2]
                    elif start == -3:
                        ii = [2, 1, 0]

                    for i in ii:
                        edge_cone = rdm.getCone(edges[i])
                        if edges_orientation[i] == -2:
                            if start >= 0:
                                forward = False
                            else:
                                forward = True
                        elif edges_orientation[i] == 0: 
                            if start >= 0:
                                forward = True
                            else:
                                forward = False
                        else:
                            raise NotImplementedError
                        if forward:
                            vertex_walk.append(edge_cone[0])
                            vertex_walk.append(edge_cone[1])
                        else:
                            vertex_walk.append(edge_cone[1])
                            vertex_walk.append(edge_cone[0])

                    for i in range(len(edges)-1):
                        assert vertex_walk[2*i+1] == vertex_walk[2*(i+1)]
                    assert vertex_walk[0] == vertex_walk[-1]
                    return [vertex_walk[0], vertex_walk[2], vertex_walk[4]]

                vertex_walk = get_vertex_walk_for_facet(old_facet_in_new_numbering, 0)
                v0 = vertex_walk[0]
                v1 = vertex_walk[1]
                v2 = vertex_walk[2]
                v3 = rdm_vertices[0] + cdm_vertices[1] - cdm_vertices[0] + new_cell//4

                print("v0: ", v0)
                print("v1: ", v1)
                print("v2: ", v2)
                print("v3: ", v3)
                mapdict = {v0: 0, v1: 1, v2: 2, v3: 3}
                new_cell_cone = [old_facet_in_new_numbering, None, None, None]
                new_cell_cone_orientation = [0, None, None, None]
                for old_edge in old_edges:
                    old_edge_in_new_numbering = old_edge - cdm_edges[0] + rdm_edges[0]
                    # the corresponding facet is the one with the same
                    # index in new_facets as old_edge has in cdm_cell_edges
                    idx = cdm_cell_edges.index(old_edge)
                    new_facet = new_facets[idx]
                    vertex_walk = get_vertex_walk_for_facet(new_facet, 0)
                    if v2 not in vertex_walk:
                        new_cell_cone[1] = new_facet
                        iv0 = vertex_walk.index(v0)
                        if iv0 == 2:
                            nextv = vertex_walk[0]
                        else:
                            nextv = vertex_walk[iv0+1]
                        if nextv != v1:
                            new_cell_cone_orientation[1] = iv0
                        else:
                            if iv0>0:
                                new_cell_cone_orientation[1] = -(iv0-1+1)
                            else:
                                new_cell_cone_orientation[1] = -3

                    elif v1 not in vertex_walk:
                        new_cell_cone[2] = new_facet
                        iv0 = vertex_walk.index(v0)
                        if iv0 == 2:
                            nextv = vertex_walk[0]
                        else:
                            nextv = vertex_walk[iv0+1]
                        if nextv == v2:
                            new_cell_cone_orientation[2] = iv0
                        else:
                            if iv0>0:
                                new_cell_cone_orientation[2] = -(iv0-1+1)
                            else:
                                new_cell_cone_orientation[2] = -3
                    elif v0 not in vertex_walk:
                        new_cell_cone[3] = new_facet
                        iv2 = vertex_walk.index(v2)
                        if iv2 == 2:
                            nextv = vertex_walk[0]
                        else:
                            nextv = vertex_walk[iv2+1]
                        if nextv == v1:
                            new_cell_cone_orientation[3] = iv2
                        else:
                            if iv2>0:
                                new_cell_cone_orientation[3] = -(iv2-1+1)
                            else:
                                new_cell_cone_orientation[3] = -3
                    else:
                        raise NotImplementedError




                should_be = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [2, 1, 3]]
                for i in range(4):
                    walk = get_vertex_walk_for_facet(new_cell_cone[i], new_cell_cone_orientation[i])
                    renamed_walk = [mapdict[w] for w in walk]
                    for j in range(3):
                        assert renamed_walk[j] == should_be[i][j]
                    print("Vertex walk for cell %i, edge %i: %s. " % (new_cell, new_cell_cone[i], renamed_walk))
                print("Introducing new cell %d -> %s with orientation %s" % (new_cell, new_cell_cone, new_cell_cone_orientation))
                rdm.setCone(new_cell, new_cell_cone)
                rdm.setConeOrientation(new_cell, new_cell_cone_orientation)

    rdm.symmetrize()
    rdm.stratify()
    # rdm.orient()

    # Now for the coordinates. Cargo-culted from DMPlexBuildCoordinates_Internal.
    # Doesn't work in parallel or deal with coordinates that are not P1.
    space_dim = cdm.getCoordinateDim()
    rdm.setCoordinateDim(space_dim)
    cdm_section = cdm.getCoordinateSection()
    rdm_section = rdm.getCoordinateSection()

    cdm_coorddm = cdm.getCoordinateDM()
    cdm_coordinates = cdm.getCoordinatesLocal()
    def cdm_coords(p):
        return cdm.getVecClosure(cdm_section, cdm_coordinates, p).reshape(-1, space_dim).mean(axis=0)

    rdm_section.setNumFields(1)
    rdm_section.setFieldComponents(0, space_dim)
    rdm_section.setChart(*rdm_vertices)
    for p in range(*rdm_vertices):
        rdm_section.setDof(p, space_dim)
        rdm_section.setFieldDof(p, 0, space_dim)
    rdm_section.setUp()

    rdm_coorddm = rdm.getCoordinateDM()
    rdm_coords = rdm_coorddm.createLocalVector()
    rdm_coords.setBlockSize(space_dim)
    rdm_coords.setName("coordinates")

    rdm_coords_array_ = rdm_coords.getArray()
    rdm_coords_array  = rdm_coords_array_.reshape(-1, space_dim)

    # Now set the damn coordinates. First, retained vertices.
    for old_vertex in range(*cdm_vertices):
        old_coords = cdm_coords(old_vertex)
        old_vertex_in_new_numbering = old_vertex - cdm_vertices[0]
        rdm_coords_array[old_vertex_in_new_numbering] = old_coords

    # Now new vertices at the barycentres.
    for cdm_cell in range(*cdm_cells):
        cdm_cell_entities = closure(cdm, cdm_cell)[0]
        cdm_cell_vertices = [x for x in cdm_cell_entities if cdm_vertices[0] <= x < cdm_vertices[1]]
        new_coords = sum(cdm_coords(p) for p in cdm_cell_vertices)/(dim+1)
        new_vertex = num_cdm_vertices + cdm_cell
        rdm_coords_array[new_vertex] = new_coords

    rdm_coords.setArray(rdm_coords_array_)
    rdm.setCoordinatesLocal(rdm_coords)

    return rdm


def BaryMeshHierarchy(mesh, refinement_levels, distribution_parameters=None, callbacks=None, reorder=None,
                      refinements_per_level=1):
    cdm = mesh._plex
    cdm.setRefinementUniform(True)
    dms = []
    if mesh.comm.size > 1 and mesh._grown_halos:
        raise RuntimeError("Cannot refine parallel overlapped meshes "
                           "(make sure the MeshHierarchy is built immediately after the Mesh)")
    parameters = {}
    if distribution_parameters is not None:
        parameters.update(distribution_parameters)
    else:
        parameters.update(mesh._distribution_parameters)

    parameters["partition"] = False
    distribution_parameters = parameters

    if callbacks is not None:
        before, after = callbacks
    else:
        before = after = lambda dm, i: None

    for i in range(refinement_levels*refinements_per_level):
        if i % refinements_per_level == 0:
            before(cdm, i)
        rdm = cdm.refine()
        if i % refinements_per_level == 0:
            after(rdm, i)
        # Remove interior facet label (re-construct from
        # complement of exterior facets).  Necessary because the
        # refinement just marks points "underneath" the refined
        # facet with the appropriate label.  This works for
        # exterior, but not marked interior facets
        rdm.removeLabel("interior_facets")
        # Remove vertex (and edge) points from labels on exterior
        # facets.  Interior facets will be relabeled in Mesh
        # construction below.
        impl.filter_exterior_facet_labels(rdm)
        rdm.removeLabel("pyop2_core")
        rdm.removeLabel("pyop2_owned")
        rdm.removeLabel("pyop2_ghost")

        dms.append(rdm)
        cdm = rdm
        # Fix up coords if refining embedded circle or sphere
        if hasattr(mesh, '_radius'):
            # FIXME, really we need some CAD-like representation
            # of the boundary we're trying to conform to.  This
            # doesn't DTRT really for cubed sphere meshes (the
            # refined meshes are no longer gnonomic).
            coords = cdm.getCoordinatesLocal().array.reshape(-1, mesh.geometric_dimension())
            scale = mesh._radius / np.linalg.norm(coords, axis=1).reshape(-1, 1)
            coords *= scale

    barydms = (bary(mesh._plex), ) + tuple(bary(dm) for dm in dms)

    barymeshes = [firedrake.Mesh(dm, dim=mesh.ufl_cell().geometric_dimension(),
                                 distribution_parameters=distribution_parameters,
                                 reorder=reorder)
                           for dm in barydms]

    meshes = [mesh] + [firedrake.Mesh(dm, dim=mesh.ufl_cell().geometric_dimension(),
                                      distribution_parameters=distribution_parameters,
                                      reorder=reorder)
                       for dm in dms]

    lgmaps = []
    for i, m in enumerate(meshes):
        no = impl.create_lgmap(m._plex)
        m.init()
        o = impl.create_lgmap(m._plex)
        m._plex.setRefineLevel(i)
        lgmaps.append((no, o))

    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    for (coarse, fine), (clgmaps, flgmaps) in zip(zip(meshes[:-1], meshes[1:]),
                                                  zip(lgmaps[:-1], lgmaps[1:])):
        c2f, f2c = impl.coarse_to_fine_cells(coarse, fine, clgmaps, flgmaps)
        coarse_to_fine_cells.append(c2f)
        fine_to_coarse_cells.append(f2c)

    lgmaps = []
    for i, m in enumerate(barymeshes):
        no = impl.create_lgmap(m._plex)
        m.init()
        o = impl.create_lgmap(m._plex)
        m._plex.setRefineLevel(i)
        lgmaps.append((no, o))

    d = mesh.topological_dimension()
    bary_coarse_to_fine_cells = []
    bary_fine_to_coarse_cells = [None]
    for (coarseu, fineu), (coarse, fine), (clgmaps, flgmaps), uniform_coarse_to_fine \
        in zip(zip(meshes[:-1], meshes[1:]),
               zip(barymeshes[:-1], barymeshes[1:]),
               zip(lgmaps[:-1], lgmaps[1:]),
               coarse_to_fine_cells):

        cdm = coarseu._plex
        fdm = fineu._plex
        _, cn2o = impl.get_entity_renumbering(cdm, coarseu._cell_numbering, "cell")
        _, fn2o = impl.get_entity_renumbering(fdm, fineu._cell_numbering, "cell")
        plex_uniform_coarse_to_fine = numpy.empty_like(uniform_coarse_to_fine)
        for i, cells in enumerate(uniform_coarse_to_fine):
            plexcells = fn2o[cells]
            plex_uniform_coarse_to_fine[cn2o[i], :] = plexcells

        ncoarse, nfine = plex_uniform_coarse_to_fine.shape
        plex_coarse_bary_to_fine_bary = numpy.full((ncoarse*(d+1),
                                                    nfine*(d+1)), -1, dtype=PETSc.IntType)

        for c in range(ncoarse*(d+1)):
            uniform = c // (d+1)
            fine_cells = plex_uniform_coarse_to_fine[uniform]
            bary_cells = []
            for fc in fine_cells:
                bary_cells.extend(list(range(fc*(d+1), (fc+1)*(d+1))))
            plex_coarse_bary_to_fine_bary[c] = bary_cells

        cdm = coarse._plex
        fdm = fine._plex

        co2n, _ = impl.get_entity_renumbering(cdm, coarse._cell_numbering, "cell")
        fo2n, _ = impl.get_entity_renumbering(fdm, fine._cell_numbering, "cell")

        coarse_bary_to_fine_bary = numpy.empty_like(plex_coarse_bary_to_fine_bary)
        # Translate plex numbering to firedrake numbering
        for i, plex_cells in enumerate(plex_coarse_bary_to_fine_bary):
            coarse_bary_to_fine_bary[co2n[i]] = fo2n[plex_cells]

        bary_coarse_to_fine_cells.append(coarse_bary_to_fine_bary)

        # Not fast but seems to work
        fine_bary_to_coarse_bary = [[]]
        for i in range(numpy.max(coarse_bary_to_fine_bary)):
            fine_bary_to_coarse_bary.append([])
        for coarse in range(coarse_bary_to_fine_bary.shape[0]):
            for ifine in range(coarse_bary_to_fine_bary.shape[1]):
                # the coarse cell `coarse` is contained in the fine cell
                # `bary_coarse_to_fine_cells[0][coarse, ifine]` so we
                # should add it to the corresponding list
                fine_bary_to_coarse_bary[coarse_bary_to_fine_bary[coarse, ifine]].append(coarse)
        fine_bary_to_coarse_bary = numpy.asarray(fine_bary_to_coarse_bary, dtype=PETSc.IntType)

        bary_fine_to_coarse_cells.append(fine_bary_to_coarse_bary)

    print(bary_coarse_to_fine_cells)
    print(bary_fine_to_coarse_cells)

    coarse_to_fine_cells = dict((Fraction(i, refinements_per_level), c2f)
                                for i, c2f in enumerate(bary_coarse_to_fine_cells))
    fine_to_coarse_cells = dict((Fraction(i, refinements_per_level), f2c)
                                for i, f2c in enumerate(bary_fine_to_coarse_cells))
    return HierarchyBase(barymeshes, coarse_to_fine_cells, fine_to_coarse_cells,
                         refinements_per_level)


@pytest.fixture(params=["interval", "triangle", "quadrilateral", "tetrahedron",
                        "prism", "hexahedron"], scope="module")
def cell(request):
    return request.param


@pytest.fixture(params=["CG", "DG"])
def space(request, cell):
    if cell in {"quadrilateral", "prism", "hexahedron"} and request.param == "DG":
        return "DQ"
    else:
        return request.param


@pytest.fixture(params=[1], scope="module")
def refinements_per_level(request):
    return request.param


# @pytest.fixture(scope="module", params=["uniform", "barycentric"])
@pytest.fixture(scope="module", params=["barycentric"])
def hierarchy(cell, refinements_per_level, request):
    if cell == "interval":
        mesh = UnitIntervalMesh(3)
        return MeshHierarchy(mesh, 2)
    elif cell in {"triangle", "prism"}:
        mesh = UnitSquareMesh(3, 3, quadrilateral=False)
    elif cell in {"quadrilateral", "hexahedron"}:
        mesh = UnitSquareMesh(3, 3, quadrilateral=True)
    elif cell == "tetrahedron":
        mesh = UnitCubeMesh(2, 2, 2)

    nref = {2: 1, 1: 2}[refinements_per_level]
    if cell == "triangle":
        if request.param == "uniform":
            hierarchy = MeshHierarchy(mesh, nref, refinements_per_level=refinements_per_level)
        else:
            hierarchy = BaryMeshHierarchy(mesh, nref, refinements_per_level=refinements_per_level)
    if cell in {"prism", "hexahedron"}:
        hierarchy = ExtrudedMeshHierarchy(hierarchy, layers=3)

    return hierarchy


@pytest.fixture(params=[False, True],
                ids=["scalar", "vector"])
def vector(request):
    return request.param


@pytest.fixture(params=["injection", "restriction", "prolongation"])
def transfer_type(request):
    return request.param


@pytest.fixture
def degrees(space):
    if space == "CG":
        return (1, 2, 3)
    elif space in {"DG", "DQ"}:
        return (0, 1, 2)


def element(space, cell, degree, vector):
    if vector:
        return VectorElement(space, cell, degree)
    else:
        return FiniteElement(space, cell, degree)


def exact_primal(mesh, vector, degree):
    x = SpatialCoordinate(mesh)
    expr = sum(pow(X, degree) for X in x)
    if vector:
        expr = as_vector([(-1)**i * expr for i in range(len(x))])
    return expr


def run_injection(hierarchy, vector, space, degrees):
    for degree in degrees:
        Ve = element(space, hierarchy[0].ufl_cell(), degree, vector)

        mesh = hierarchy[-1]
        V = FunctionSpace(mesh, Ve)

        actual = interpolate(exact_primal(mesh, vector, degree), V)

        for mesh in reversed(hierarchy[:-1]):
            V = FunctionSpace(mesh, Ve)
            expect = interpolate(exact_primal(mesh, vector, degree), V)
            tmp = Function(V)
            inject(actual, tmp)
            actual = tmp
            assert numpy.allclose(expect.dat.data_ro, actual.dat.data_ro)


def run_prolongation(hierarchy, vector, space, degrees):
    for degree in degrees:
        Ve = element(space, hierarchy[0].ufl_cell(), degree, vector)

        mesh = hierarchy[0]
        V = FunctionSpace(mesh, Ve)

        actual = interpolate(exact_primal(mesh, vector, degree), V)

        for mesh in hierarchy[1:]:
            V = FunctionSpace(mesh, Ve)
            expect = interpolate(exact_primal(mesh, vector, degree), V)
            tmp = Function(V)
            prolong(actual, tmp)
            actual = tmp
            assert numpy.allclose(expect.dat.data_ro, actual.dat.data_ro)


def run_restriction(hierarchy, vector, space, degrees):
    def victim(V):
        return Function(V).assign(1)

    def dual(V):
        f = Function(V).assign(1)
        return assemble(inner(f, TestFunction(V))*dx)

    def functional(victim, dual):
        with victim.dat.vec_ro as v, dual.dat.vec_ro as dv:
            return dv.dot(v)

    for degree in degrees:
        Ve = element(space, hierarchy[0].ufl_cell(), degree, vector)
        for cmesh, fmesh in zip(hierarchy[:-1], hierarchy[1:]):
            Vc = FunctionSpace(cmesh, Ve)
            Vf = FunctionSpace(fmesh, Ve)
            fine_dual = dual(Vf)
            coarse_primal = victim(Vc)

            coarse_dual = Function(Vc)
            fine_primal = Function(Vf)

            wc = Function(Vc)
            wc.assign(1.)
            wf = Function(Vf)
            prolong_count(wc, wf)
            fine_dual_w = fine_dual.copy(deepcopy=True)
            fine_dual_w.vector().set_local(fine_dual.vector().get_local()/wf.vector().get_local())
            restrict(fine_dual_w, coarse_dual)
            prolong(coarse_primal, fine_primal)
            coarse_functional = functional(coarse_primal, coarse_dual)
            fine_functional = functional(fine_primal, fine_dual)

            assert numpy.allclose(fine_functional, coarse_functional)


def test_grid_transfer(hierarchy, vector, space, degrees, transfer_type):
    if transfer_type == "injection":
        run_injection(hierarchy, vector, space, degrees)
    elif transfer_type == "restriction":
        run_restriction(hierarchy, vector, space, degrees)
    elif transfer_type == "prolongation":
        run_prolongation(hierarchy, vector, space, degrees)


@pytest.mark.parallel(nprocs=2)
def test_grid_transfer_parallel(hierarchy, transfer_type):
    space = "CG"
    degs = degrees(space)
    vector = False
    if transfer_type == "injection":
        run_injection(hierarchy, vector, space, degs)
    elif transfer_type == "restriction":
        run_restriction(hierarchy, vector, space, degs)
    elif transfer_type == "prolongation":
        run_prolongation(hierarchy, vector, space, degs)
