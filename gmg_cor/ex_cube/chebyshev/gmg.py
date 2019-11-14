from dolfin import (Mesh, FunctionSpace, PETScDMCollection,
                    Expression, near, DirichletBC, TrialFunction, TestFunction,
                    Constant, dot, grad, dx, ds, PETScMatrix, PETScVector,
                    assemble_system, interpolate)
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
    Ng is the number of iterations (usually 2), igg is the initial guess
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
        res4 = residual(Ahlist[0], bh, uhlist[0])
        print('the residual after', num_cycle + 1, 'cycles: ', res4)

    return uhlist[0]


# =================================================================
# Read the meshes.
nl = 4
Vspace = []

mesh = Mesh("../level_1.xml")
V = FunctionSpace(mesh, 'P', 1)
Vspace.append(V)

mesh1 = Mesh("../level_2.xml")
V1 = FunctionSpace(mesh1, 'P', 1)
Vspace.append(V1)

mesh2 = Mesh("../level_3.xml")
V2 = FunctionSpace(mesh2, 'P', 1)
Vspace.append(V2)

mesh3 = Mesh("../level_4.xml")
V3 = FunctionSpace(mesh3, 'P', 1)
Vspace.append(V3)

# Find the transfer operators, puse is the prolongation operator list
# note the order is from fine to coarse
puse = []
for il in range(nl-1):
    pmat = PETScDMCollection.create_transfer_matrix(Vspace[il+1], Vspace[il])
    pmat = pmat.mat()
    puse.append(pmat)
# ==========================================================================

# Use FEniCS to formulate the FEM problem. A is the matrix, b is the rhs.
u_D = Constant(0.0)

tol = 1E-14


def boundary_D(x, on_boundary):

    return on_boundary and (near(x[0], 0, tol) or near(x[0], 1.0, tol))


bc = DirichletBC(V, u_D, boundary_D)
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10 * exp( - (pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2) \
                 + pow(x[2] - 0.5, 2)) / 1)", degree=6)
g = Expression("sin(5.0*x[0])*sin(5.0*x[1])", degree=6)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx + g * v * ds
A = PETScMatrix()
b = PETScVector()
assemble_system(a, L, bc, A_tensor=A, b_tensor=b)

A = A.mat()
b = b.vec()

# =========================================================================

# Construct the alist for systems on levels from fine to coarse
# construct the transfer operators first
ruse = [None] * (nl - 1)
Alist = [None] * (nl)

ruse[0] = Mat()
puse[0].transpose(ruse[0])
Alist[0] = A

for il in range(1, nl-1):
    ruse[il] = Mat()
    puse[il].transpose(ruse[il])
    Alist[il] = Mat()
    Alist[il - 1].PtAP(puse[il - 1], Alist[il])

# find the coarsest grid matrix
Alist[nl-1] = Mat()
Alist[nl-2].PtAP(puse[nl-2], Alist[nl-1])
# =========================================================

# Set initial guess
fe = Constant(0.0)
fp = interpolate(fe, V)
fph = fp.vector().vec()

# Multigrid
print('Initial residual is:', residual(A, b, fph))
wh = mg(Alist, b, fph, puse, ruse, 20, nl, 2, 'chebyshev', 'jacobi')
print('Final residual is:', residual(A, b, wh))
