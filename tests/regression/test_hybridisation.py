"""Solve a mixed Helmholtz problem

sigma + grad(u) = 0
u + div(sigma) = f

using hybridisation. The corresponding weak (variational problem)

<tau, sigma> - <div(tau), u> + <<[tau.n], lambda>> = 0 for all tau
<v, u> + <v, div(sigma)> = <v, f> for all v
<<gamma, [sigma.n]>> = 0 for all gamma

is solved using broken RT (Raviart-Thomas) elements of degree k for
(sigma, tau), DG (discontinuous Galerkin) elements of degree k - 1
for (u, v), and Trace-RT elements for (lambda, gamma).

No strong boundary conditions are enforced. A weak boundary condition on
u is enforced implicitly, setting <<u, tau.n>> = 0 for all tau.

The forcing function is chosen as

(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)

which reproduces the known analytical solution

sin(x[0]*pi*2)*sin(x[1]*pi*2)
"""

import pytest
from firedrake import *

class APC(object):
    def __init__(self, A_inv):
        self.A_inv = A_inv
        def apply(self, pc, x, y):
            self.A_inv.M.handle.mult(x,y)

@pytest.mark.parametrize('degree', range(1, 3))
def test_hybridisation(degree):
    # Create mesh
    mesh = UnitSquareMesh(8, 8)

    # Define function spaces and mixed (product) space
    RT_elt = FiniteElement("RT", triangle, degree)

    BrokenRT = FunctionSpace(mesh, BrokenElement(RT_elt))
    DG = FunctionSpace(mesh, "DG", degree-1)
    TraceRT = FunctionSpace(mesh, TraceElement(RT_elt))

    W = MixedFunctionSpace([BrokenRT, TraceRT])

    # Define trial and test functions
    sigma, lambdar = TrialFunctions(W)
    tau, gammar = TestFunctions(W)

    # Mesh normal
    n = FacetNormal(mesh)

    # Define source function
    f = Function(DG)
    f.interpolate(Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))

    #Eliminate u from the equation analytically
    #u = -div(sigma) + f

    # Define variational form
    a_dx = (dot(tau, sigma) + div(tau)*div(sigma))*dx
    a_dS = (jump(tau, n=n)*lambdar('+') + gammar('+')*jump(sigma, n=n))*dS
    a = a_dx + a_dS
    L = div(tau)*f*dx

    #build inverse for hybridised solver
    sigma = TrialFunction(BrokenRT)
    tau = TestFunction(BrokenRT)
    DG_inv = assemble(inner(sigma,tau)*dx,inverse=True)

    bcs = DirichletBC(W.sub(1), Constant(0), (1, 2, 3, 4))

    #Build a schur complement S
    aMat = assemble(a, bcs=bcs)
    B = aMat.M[0,1].handle
    C = aMat.M[1, 0].handle
    D = aMat.M[1, 1].handle
    
    tmp = DG_inv.M.handle.matMult(B)
    tmp = C.matMult(tmp)
    S = D.copy()
    S.axpy(-1, tmp)

    # Compute solution
    w = Function(W)
    prob = LinearVariationalProblem(a,L,w,bcs=bcs)
    solver = LinearVariationalSolver(prob,
                                     solver_parameters={'ksp_rtol': 1e-14,
                                                        'ksp_max_it': 300000})
    solver.snes.setUp()
    ksp = solver.snes.ksp
    kspA00 = ksp.pc.getFieldSplitSubKSP()[0]

    # This is the top left block
    kspA00.pc.setType(PETSc.PC.Type.PYTHON)
    kspA00.pc.setPythonContext(APC(DG_inv))

    # Now set the schur complement PC matrix
    ksp.pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER, S)

    solver.solve()
    Hsigma, Hlambdar = w.split()

    #reconstruct Hu
    u = TrialFunction(DG)
    v = TestFunction(DG)
    Hu_rhs = v*(-div(Hsigma) + f)*dx
    a_Hu = u*v*dx
    Hu = Function(DG)
    solve(a_Hu==Hu_rhs, Hu)

    # Compare result to non-hybridised calculation
    RT = FunctionSpace(mesh, "RT", degree)
    W2 = RT * DG
    sigma, u = TrialFunctions(W2)
    tau, v = TestFunctions(W2)
    w2 = Function(W2)
    a = (dot(tau, sigma) - div(tau)*u + v*u + v*div(sigma))*dx
    L = f*v*dx
    solve(a == L, w2, solver_parameters={'ksp_rtol': 1e-14})
    NHsigma, NHu = w2.split()

    # Return L2 norm of error
    # (should be identical, i.e. comparable with solver tol)
    uerr = sqrt(assemble((Hu-NHu)*(Hu-NHu)*dx))
    sigerr = sqrt(assemble(dot(Hsigma-NHsigma, Hsigma-NHsigma)*dx))

    assert uerr < 1e-11
    assert sigerr < 4e-11

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
