from dolfin import (Mesh, FunctionSpace, PETScDMCollection,
                    Expression, DirichletBC, TrialFunction, TestFunction,
                    Constant, dot, grad, dx, PETScMatrix, PETScVector,
                    assemble_system, interpolate)
from petsc4py import PETSc
from petsc4py.PETSc import (Mat, Vec)
import numpy as np


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

mesh = Mesh("./level_1.xml")
V = FunctionSpace(mesh, 'P', 1)
Vspace.append(V)

mesh1 = Mesh("./level_2.xml")
V1 = FunctionSpace(mesh1, 'P', 1)
Vspace.append(V1)

mesh2 = Mesh("./level_3.xml")
V2 = FunctionSpace(mesh2, 'P', 1)
Vspace.append(V2)

mesh3 = Mesh("./level_4.xml")
V3 = FunctionSpace(mesh3, 'P', 1)
Vspace.append(V3)
# ==========================================================================

# perturb several nodes to make low quality meshes
# level one
coord = mesh.coordinates()
coord[233] = np.array([0.45286306, 0.55238262])
coord[1083] = np.array([0.45308537, 0.55225124])
coord[1000] = np.array([0.00788574, 0.68094166])
coord[1098] = np.array([0.00788488, 0.69369237])
coord[526] = np.array([0.94154541, 0.89955308])
coord[1052] = np.array([0.94127031, 0.8994045])

# level two
coord1 = mesh1.coordinates()
coord1[52] = np.array([0.45749346, 0.96374366])
coord1[281] = np.array([0.52153044, 0.57187736])
coord1[475] = np.array([0.52212573, 0.5722054])
coord1[419] = np.array([0.13211501, 0.37105498])
coord1[450] = np.array([0.13114422, 0.36911634])

# level three
coord2 = mesh2.coordinates()
coord2[48] = np.array([0.14844939, 0.48768435])
coord2[93] = np.array([0.14419585, 0.48552866])
coord2[54] = np.array([0.61874016, 0.20549839])
coord2[82] = np.array([0.61747261, 0.20908676])
coord2[120] = np.array([0.99948903, 0.65001777])
coord2[94] = np.array([0.99935963, 0.5501653])

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
u_D = Expression('0.0', degree=0)


# Define boundary for DirichletBC
def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V, u_D, boundary)
u = TrialFunction(V)
v = TestFunction(V)
# f = Expression('2*pi*pi*sin(pi*x[0])*sin(pi*x[1])',degree=6)
f = Constant(0.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx
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
bad_level1 = [1792, 14, 15, 16, 784, 21, 22, 23, 920, 921, 922, 27, 919, 29, 30, 31, 28, 37, 38, 40, 41, 45, 1845, 1846, 1847, 1848, 53, 1850, 55, 54, 1849, 1726, 959, 960, 56, 840, 841, 842, 843, 844, 1899, 1900, 1901, 1902, 1013, 1787, 1788, 1789, 1790, 1791]
bad_level2 = [129, 130, 131, 133, 134, 135, 141, 142, 143, 144, 157, 158, 159, 420, 422, 423, 166, 169, 170, 437, 183, 184, 199, 200, 459, 460, 474, 475, 96, 97, 226, 227, 486, 487, 488, 506, 507]
bad_level3 = [132, 133, 134, 135, 9, 15, 16, 23, 24, 25, 26, 33, 34, 61, 62, 63, 64, 75, 76, 77, 78, 79, 82, 83, 84, 92, 94, 95, 96, 97, 99, 100, 101, 102, 103, 108, 112, 113, 114, 118, 119, 120, 121]
# bad_level1 = []
# bad_level2 = []
# bad_level3 = []
Blist = [bad_level1, bad_level2, bad_level3]


corlist = makecorrection(Alist, Blist)

# =========================================================


# Set initial guess
fe = Expression('sin(pi*k*x[0])*sin(pi*k*x[1])', degree=6, k=10.0)
fp = interpolate(fe, V)
fph = fp.vector().vec()

# Multigrid
print('Initial residual is:', residual(A, b, fph))
wh = mg(Alist, b, fph, puse, ruse, 10, nl, 2, 'richardson', 'sor', corlist, Blist)
print('Final residual is:', residual(A, b, wh))