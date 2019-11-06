from dolfin import (UnitCubeMesh, VectorFunctionSpace,
                    Function, PETScDMCollection,
                    Expression, DirichletBC, TrialFunction, TestFunction,
                    Constant, MeshFunction, SubDomain, near, Measure,
                    inner, dot, grad, div, dx, Identity,
                    as_backend_type, PETScMatrix, PETScVector,
                    assemble_system, interpolate, File)
from petsc4py import PETSc
from petsc4py.PETSc import Mat


# Use petsc4py to define the smoothers
def direct(Ah, bh):
    '''LU factorisation. Ah is the matrix, bh is the rhs'''
    ksp = PETSc.KSP().create()
    yh = bh.duplicate()
    ksp.setType('preonly')
    ksp.setNormType(PETSc.KSP.NormType.NONE)
    pc = ksp.getPC()
    pc.setType('lu')
    ksp.setOperators(Ah)
    ksp.setFromOptions()
    ksp.solve(bh, yh)
    return yh


def smoother(Ag, bg, Ng, igg, ksptype, pctype):
    '''Smoother for multigrid. Ag, and bg are the LHS and RHS respectively.
    Ng is the number of iterations (usually 1), igg is the initial guess
    for the solution.
    ksptype and pctype can be ('richardson', 'jacobi'), ('richardson', 'sor')
    or ('chebyshev', 'jacobi') for example '''

    ksp = PETSc.KSP().create()
    ksp.setType(ksptype)
    ksp.setNormType(PETSc.KSP.NormType.NONE)
    pc = ksp.getPC()
    pc.setType(pctype)
    ksp.setInitialGuessNonzero(True)
    ksp.setTolerances(max_it=Ng)
    ksp.setOperators(Ag)
    ksp.setFromOptions()
    ksp.solve(bg, igg)


def residual(Ah, bh, xh):
    '''a function to calculate the residual
    Ah is the matrix, bh is the rhs, xh is the approximation'''
    resh = bh - Ah * xh
    normr = PETSc.Vec.norm(resh, 2)
    return normr


def mg(Ahlist, bh, uh, prolongation, restriction, N_cycles, N_levels,
        nu, ksptype, pctype):
    '''multigrid for N level mesh
    Ahlist is the matrix list from fine to coarse
    bh is rhs on the finest grid
    uh is the initial guess for finest grid
    prolongation is a list containing all of operators from fine-to-coarse
    N_cycles is number of cycles and N_levels is number of levels
    nu1, nu2 are the number of pre- and post-smoothers applied
    ksptype, pctype are the smoother used'''
    # r0 = residual(Ahlist[0], bh, uh)

    # make a restriction list and gird operator list and rhs list
    # and initial guess list
    # initialize the first entity
    blist = [None] * N_levels
    uhlist = [None] * N_levels
    blist[0] = bh
    uhlist[0] = uh

    # calculate the restriction, matrix, and initial guess lists
    # except coarsest grid, since transfer operator and initial guess
    # is not defined on that level
    for i_level in range(1, N_levels - 1):
        uhlist[i_level] = restriction[i_level - 1] * uhlist[i_level - 1]

    for num_cycle in range(N_cycles):

        # restriction to coarse grids
        for i in range(N_levels - 1):

            # apply smoother to every level except the coarsest level
            smoother(Ahlist[i], blist[i], nu, uhlist[i], ksptype, pctype)

            # obtain the rhs for next level
            res = blist[i] - Ahlist[i] * uhlist[i]
            blist[i + 1] = restriction[i] * res

        # on the coarsest grid, apply direct lu
        uhlist[N_levels - 1] = direct(Ahlist[N_levels - 1],
                                      blist[N_levels - 1])

        # prolongation back to fine grids
        for j in range(N_levels - 2, -1, -1):

            uhlist[j] += prolongation[j] * uhlist[j + 1]
            smoother(Ahlist[j], blist[j], nu, uhlist[j], ksptype, pctype)

        # calculate the relative residual
        # res4 = residual(Ah, bh, uhlist[0]) / r0

    return uhlist[0]
# ================================================================


def mgcg(Ah, bh, igh, Ncg, Ahlist, prolongation, restriction, Nmg, Nle, nu,
         ksptype, pctype):
    '''multigrid preconditioned conjugate gradient
    Ah is the matrix, bh is the rhs, igh is initial guess
    Ncg is the number of iterations of cg
    Ahlist, prolongation, restriction are multigrid parameters
    Nmg is number of cycles for multigrid, Nle is number of levels
    nu1, nu2 number of smoothers applied
    ksptype and pctype are to define the smoother'''

    # initialize the problem
    r0 = residual(Ah, bh, igh)
    rk = bh - Ah * igh
    wk = igh.copy()
    zk = mg(Ahlist, rk, wk, prolongation, restriction, Nmg, Nle, nu,
            ksptype, pctype)
    pk = zk.copy()
    xk = igh.copy()

    # conjugate gradient
    for ite in range(Ncg):
        alpha = (zk.dot(rk)) / ((Ah * pk).dot(pk))
        w1 = alpha * pk
        xk = (xk+w1).copy()
        rtest = Ah * pk
        rs = alpha * rtest
        rk2 = (rk - rs).copy()
        rt = igh.copy()
        zk2 = mg(Ahlist, rk2, rt, prolongation, restriction, Nmg, Nle, nu,
                 ksptype, pctype)
        beta = (rk2.dot(zk2)) / (rk.dot(zk))
        y1 = beta * pk
        pk = (zk2 + y1).copy()
        rk = rk2.copy()
        zk = zk2.copy()

        res4 = residual(Ah, bh, xk)
        print('residual after', ite + 1, 'iterations of CG is:',
              res4)
    return xk


# ===============================================================
# Read the meshes.
nl = 4
Vspace = []

mesh = UnitCubeMesh(64, 64, 64)
V = VectorFunctionSpace(mesh, 'P', 1)
Vspace.append(V)

mesh1 = UnitCubeMesh(32, 32, 32)
V1 = VectorFunctionSpace(mesh1, 'P', 1)
Vspace.append(V1)

mesh2 = UnitCubeMesh(16, 16, 16)
V2 = VectorFunctionSpace(mesh2, 'P', 1)
Vspace.append(V2)

mesh3 = UnitCubeMesh(8, 8, 8)
V3 = VectorFunctionSpace(mesh3, 'P', 1)
Vspace.append(V3)

# Find the transfer operators, puse is the prolongation operator list
# note the order is from fine to coarse
puse = []
for il in range(nl-1):
    pmat = PETScDMCollection.create_transfer_matrix(Vspace[il+1], Vspace[il])
    pmat = pmat.mat()
    puse.append(pmat)
# ===================================================================

# lame parameters
E = 69e9
nu = 0.33
mu = Constant(E/(2.0*(1.0 + nu)))
lambda_ = Constant(E*nu/((1.0 + nu)*(1.0 - 2.0*nu)))

# Dirichlet boundary condition
tol = 1E-14


def clamped_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0, tol)


bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)


# Define strain and stress
def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)


def sigma(u):
    return lambda_*div(u)*Identity(d) + 2*mu*epsilon(u)


# Mark facets of the mesh and Neumann boundary condition
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)


class NeumanBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1, tol)


NeumanBoundary().mark(boundaries, 1)


# Define outer surface measure aware of Dirichlet and Neumann boundaries
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)


# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0, 0, 0))
T = Constant((10**3, 0, 0))
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx + dot(T, v) * ds(1)

A = PETScMatrix()
b = PETScVector()
assemble_system(a, L, bc, A_tensor=A, b_tensor=b)

A = A.mat()
b = b.vec()
print('problem size: ', b.getSize())
# =========================================================================

# Construct the alist for systems on levels from fine to coarse
# construct the transfer operators first
ruse = [None] * (3)
Alist = [None] * (4)

ruse[0] = Mat()
puse[0].transpose(ruse[0])
Alist[0] = A

for il in range(1, nl-1):
    ruse[il] = Mat()
    puse[il].transpose(ruse[il])
    Alist[il] = Mat()
    Alist[il - 1].PtAP(puse[il - 1], Alist[il])

# find the coarsest grid matrix
Alist[3] = Mat()
Alist[2].PtAP(puse[2], Alist[3])
# =========================================================
igh = as_backend_type(Function(V).vector())
igh[:] = 0.0
fph = igh.vec()
# Multigrid conjugate gradient
print('Initial residual is:', residual(A, b, fph))
wh = mgcg(A, b, fph, 10, Alist, puse, ruse, 1, nl, 2, 'chebyshev', 'jacobi')
quit()
# ==============================================================================
rei = Function(V)
reh = b - A * wh
rei.vector()[:] = reh.getArray()

file_p = File('./finalresidual.pvd')
file_p << rei
