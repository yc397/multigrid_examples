from dolfin import (UnitCubeMesh, FunctionSpace, PETScDMCollection,
                    Expression, DirichletBC, TrialFunction, TestFunction,
                    Constant, dot, grad, dx, PETScMatrix, PETScVector,
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


def mg(Ah, bh, uh, prolongation, N_cycles, N_levels, ksptype, pctype):
    '''multigrid for N level mesh
    Ah is the matrix, bh is rhs on the finest grid,
    uh is the initial guess for finest grid
    prolongation is a list containing all of operators from fine-to-coarse
    N_cycles is number of cycles and N_levels is number of levels'''

    r0 = residual(Ah, bh, uh)

    # make a restriction list and gird operator list and rhs list
    # and initial guess list
    # initialize the first entity
    restriction = [None] * (N_levels - 1)
    Alist = [None] * N_levels
    blist = [None] * N_levels
    restriction[0] = Mat()
    prolongation[0].transpose(restriction[0])
    uhlist = [None] * N_levels
    Alist[0] = Ah
    blist[0] = bh
    uhlist[0] = uh

    # calculate the restriction, matrix, and initial guess lists
    # except coarsest grid, since transfer operator and initial guess
    # is not defined on that level
    for i_level in range(1, N_levels - 1):
        restriction[i_level] = Mat()
        prolongation[i_level].transpose(restriction[i_level])
        Alist[i_level] = Mat()
        Alist[i_level - 1].PtAP(prolongation[i_level - 1], Alist[i_level])
        uhlist[i_level] = restriction[i_level - 1] * uhlist[i_level - 1]

    # find the coarsest grid matrix
    Alist[N_levels - 1] = Mat()
    Alist[N_levels - 2].PtAP(prolongation[N_levels - 2], Alist[N_levels - 1])

    for num_cycle in range(N_cycles):

        # restriction to coarse grids
        for i in range(N_levels - 1):

            # apply smoother to every level except the coarsest level
            smoother(Alist[i], blist[i], 2, uhlist[i], ksptype, pctype)

            # obtain the rhs for next level
            res = blist[i] - Alist[i] * uhlist[i]
            blist[i + 1] = restriction[i] * res

        # on the coarsest grid, apply direct lu
        uhlist[N_levels - 1] = direct(Alist[N_levels - 1], blist[N_levels - 1])

        # prolongation back to fine grids
        for j in range(N_levels - 2, -1, -1):

            uhlist[j] += prolongation[j] * uhlist[j + 1]
            smoother(Alist[j], blist[j], 2, uhlist[j], ksptype, pctype)

        # calculate the relative residual
        res4 = residual(Ah, bh, uhlist[0]) / r0
        print('relative residual after', num_cycle + 1, 'cycles is:', res4)

    return uhlist[0]


# =================================================================
# Read the meshes. mesh1 is the coarse mesh and mesh2 is the fine mesh

# mesh1=Mesh("./coarse.xml")
mesh1 = UnitCubeMesh(10, 10, 10)
V1 = FunctionSpace(mesh1, 'P', 1)
n1 = mesh1.num_vertices()

# mesh2=Mesh("./fine.xml")
mesh2 = UnitCubeMesh(20, 20, 20)
V2 = FunctionSpace(mesh2, 'P', 1)
n2 = mesh2.num_vertices()

mesh3 = UnitCubeMesh(40, 40, 40)
V3 = FunctionSpace(mesh3, 'P', 1)
n3 = mesh3.num_vertices()

mesh4 = UnitCubeMesh(80, 80, 80)
V4 = FunctionSpace(mesh4, 'P', 1)
n4 = mesh4.num_vertices()

# Find the transfer operators, puse is the prolongation operator list
# note the order is from fine to coarse
puse0 = PETScDMCollection.create_transfer_matrix(V3, V4)
puse0 = puse0.mat()
puse1 = PETScDMCollection.create_transfer_matrix(V2, V3)
puse1 = puse1.mat()
puse2 = PETScDMCollection.create_transfer_matrix(V1, V2)
puse2 = puse2.mat()
puse = [puse0, puse1, puse2]

# Use FEniCS to formulate the FEM problem. A is the matrix, b is the rhs.
u_D = Expression('0.0', degree=0)


# Define boundary for DirichletBC
def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V4, u_D, boundary)
u = TrialFunction(V4)
v = TestFunction(V4)
# f = Expression('2*pi*pi*sin(pi*x[0])*sin(pi*x[1])',degree=6)
f = Constant(0.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx
A = PETScMatrix()
b = PETScVector()
assemble_system(a, L, bc, A_tensor=A, b_tensor=b)

A = A.mat()
b = b.vec()

# Set initial guess
fe = Expression('sin(pi*k*x[0])*sin(pi*k*x[1])', degree=6, k=10.0)
# fe=Expression('sin(pi*k*x[0])*sin(pi*k*x[1])+sin(2.0*pi*x[0])*sin(2.0*pi*x[1])',degree=6,k=10.0)
fp = interpolate(fe, V4)
fph = fp.vector().vec()

# Multigrid
print('Initial residual is:', residual(A, b, fph))
wh = mg(A, b, fph, puse, 10, 4, 'richardson', 'sor')
print('Final residual is:', residual(A, b, fph))
