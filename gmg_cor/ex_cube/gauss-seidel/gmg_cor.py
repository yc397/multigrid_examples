from dolfin import (Mesh, FunctionSpace, PETScDMCollection,
                    Expression, near, DirichletBC, TrialFunction, TestFunction,
                    Constant, dot, grad, dx, ds, PETScMatrix, PETScVector,
                    assemble_system, interpolate)
from petsc4py import PETSc
from petsc4py.PETSc import (Mat, Vec)
import numpy as np
import find_low


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


def smoother_global(Ag, bg, Ng, igg, ksptype, pctype):
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


def makecorrection(Ahlist, poorlist):

    Acorlist = [None] * (len(Ahlist) - 1)

    for k in range(len(Ahlist) - 1):
        Ah = Ahlist[k]
        badpet = PETSc.IS()
        badpet.createGeneral(poorlist[k])
        Abad = Mat()
        Ah.createSubMatrix(badpet, badpet, Abad)
        Abad.assemblyBegin()
        Abad.assemblyEnd()
        Acorlist[k] = Abad.copy()

    return Acorlist


def smoother_local(Ah, bh, ugh, Acor, poordof):
    '''
    Local correction smoother. Ah is the whole matrix,
    bh is the whole rhs, ugh is the initial guess or input.
    Also need a vector B which contains the indices of all bad nodes.
    fixa is the submatrix, fixr is the corresponding residual part and
    fixu is the error obtained.
    '''
    rh = bh - Ah * ugh
    badpet = PETSc.IS()
    badpet.createGeneral(poordof)
    rcor = Vec()
    rh.getSubVector(badpet, rcor)
    nb = rcor.getSize()
    ecor = direct(Acor, rcor)
    for i in range(nb):
        row = poordof[i]
        ugh[row] += ecor.getValue(i)

    return ugh


def smoother_cor(Ag, bg, Ng, igg, ksptype, pctype, Acor, poordof):

    for i in range(Ng):

        smoother_global(Ag, bg, 1, igg, ksptype, pctype)
        smoother_local(Ag, bg, igg, Acor, poordof)

    return igg


def mg(Ahlist, bh, uh, prolongation, restriction, N_cycles, N_levels,
        nu, ksptype, pctype, Acorlist, poorlist):
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
            smoother_cor(Ahlist[i], blist[i], nu, uhlist[i], ksptype, pctype,
                         Acorlist[i], poorlist[i])

            # obtain the rhs for next level
            res = blist[i] - Ahlist[i] * uhlist[i]
            blist[i + 1] = restriction[i] * res

        # on the coarsest grid, apply direct lu
        uhlist[N_levels - 1] = direct(Ahlist[N_levels - 1],
                                      blist[N_levels - 1])

        # prolongation back to fine grids
        for j in range(N_levels - 2, -1, -1):

            uhlist[j] += prolongation[j] * uhlist[j + 1]
            smoother_cor(Ahlist[j], blist[j], nu, uhlist[j], ksptype, pctype,
                         Acorlist[j], poorlist[j])

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
# ==========================================================================

# perturb several nodes to make low quality meshes
# level one
coord = mesh.coordinates()
coord[33870] = np.array([0.84422225, 0.12003844, 0.85444908])
coord[40535] = np.array([0.84422365, 0.12005959, 0.85447413])
coord[18836] = np.array([0.4531466, 0.99313058, 0.4044442])
coord[62510] = np.array([0.4532232, 0.99307068, 0.40441478])
coord[70888] = np.array([0.76749874, 0.43333037, 0.15925556])
coord[83559] = np.array([0.76743825, 0.43337698, 0.15928119])
coord[27904] = np.array([0.85599083, 0.89423123, 0.54497891])
coord[43516] = np.array([0.85603036, 0.89426995, 0.54499929])

# level two
coord1 = mesh1.coordinates()
coord1[9475] = np.array([0.90587922, 0.01510134, 0.14508207])
coord1[5122] = np.array([0.52960123, 0.98707025, 0.69302149])
coord1[9700] = np.array([0.53421175, 0.9826492, 0.69542906])
coord1[6539] = np.array([0.2653881, 0.11901515, 0.66318406])
coord1[12445] = np.array([0.26543822, 0.11894697, 0.66315402])
coord1[12108] = np.array([0.85761841, 0.90535035, 0.01650838])

# level three
coord2 = mesh2.coordinates()
coord2[495] = np.array([0.60887783, 0.03752392, 0.49570812])
coord2[1660] = np.array([0.60925493, 0.03829348, 0.49679833])
coord2[1038] = np.array([0.54908314, 0.96332498, 0.47490629])
coord2[1339] = np.array([0.54848358, 0.96211321, 0.47332813])
coord2[1492] = np.array([0.27133558, 0.27171539, 0.50072606])
coord2[1142] = np.array([0.27136816, 0.27167537, 0.50056124])
coord2[1448] = np.array([0.80807753, 0.90899875, 0.77327936])
coord2[1654] = np.array([0.80824022, 0.90899337, 0.77317984])


# level four
coord3 = mesh3.coordinates()
coord3[235] = np.array([0.73388744, 0.42498976, 0.60173443])

# ==========================================================================

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
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2) \
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

# generate the local correction systems
bad_level1 = find_low.find_low(mesh, V, Alist[0])
bad_level2 = find_low.find_low(mesh1, V1, Alist[1])
bad_level3 = find_low.find_low(mesh2, V2, Alist[2])
Blist = [bad_level1, bad_level2, bad_level3]


corlist = makecorrection(Alist, Blist)

# =========================================================


# Set initial guess
fe = Constant(0.0)
fp = interpolate(fe, V)
fph = fp.vector().vec()

# Multigrid
print('Initial residual is:', residual(A, b, fph))
wh = mg(Alist, b, fph, puse, ruse, 20, nl, 2,
        'richardson', 'sor', corlist, Blist)
print('Final residual is:', residual(A, b, wh))
