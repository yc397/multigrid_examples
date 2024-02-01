from dolfin import (Mesh, VectorFunctionSpace, PETScDMCollection,
                    DirichletBC, near, TrialFunction, TestFunction,
                    div, Identity, MeshFunction, SubDomain, Measure,
                    Constant, dot, grad, dx, inner, PETScMatrix, PETScVector,
                    assemble_system, as_backend_type, Function)
import numpy as np
from petsc4py import PETSc
from petsc4py.PETSc import (Mat, Vec)
import find_low

# set petsc options at beginning
petsc_options = PETSc.Options()

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

def smoother_l1(Ag, bg, Ng, igg, ksptype, pctype):
    '''Smoother for multigrid. Ag, and bg are the LHS and RHS respectively.
    Ng is the number of iterations (usually 1), igg is the initial guess
    for the solution.
    ksptype and pctype can be ('richardson', 'jacobi'), ('richardson', 'sor')
    or ('chebyshev', 'jacobi') for example '''

    ksp = PETSc.KSP().create()
    ksp.setOptionsPrefix('smoother1_')
    petsc_options['smoother1_ksp_type'] = ksptype
    ksp.setNormType(PETSc.KSP.NormType.NONE)
    # ksp.setComputeEigenvalues(1)
    pc = ksp.getPC()
    pc.setOptionsPrefix('smpc1_')
    petsc_options['smpc1_pc_type'] = pctype
    petsc_options['smoother1_ksp_initial_guess_nonzero'] = True
    petsc_options['smoother1_ksp_chebyshev_eigenvalues'] = '0.2993908,2.993908'
    ksp.setTolerances(max_it=Ng)
    ksp.setOperators(Ag)
    ksp.setFromOptions()
    ksp.solve(bg, igg)
    pc.destroy()
    ksp.destroy()


def smoother_l2(Ag, bg, Ng, igg, ksptype, pctype):
    '''Smoother for multigrid. Ag, and bg are the LHS and RHS respectively.
    Ng is the number of iterations (usually 1), igg is the initial guess
    for the solution.
    ksptype and pctype can be ('richardson', 'jacobi'), ('richardson', 'sor')
    or ('chebyshev', 'jacobi') for example '''

    ksp = PETSc.KSP().create()
    ksp.setOptionsPrefix('smoother2_')
    petsc_options['smoother2_ksp_type'] = ksptype
    ksp.setNormType(PETSc.KSP.NormType.NONE)
    pc = ksp.getPC()
    pc.setOptionsPrefix('smpc2_')
    petsc_options['smpc2_pc_type'] = pctype
    petsc_options['smoother2_ksp_initial_guess_nonzero'] = True
    petsc_options['smoother2_ksp_chebyshev_eigenvalues'] = '0.2469,\
                                                             2.469006'
    ksp.setTolerances(max_it=Ng)
    ksp.setOperators(Ag)
    ksp.setFromOptions()
    ksp.solve(bg, igg)
    pc.destroy()
    ksp.destroy()


def smoother_l3(Ag, bg, Ng, igg, ksptype, pctype):
    '''Smoother for multigrid. Ag, and bg are the LHS and RHS respectively.
    Ng is the number of iterations (usually 1), igg is the initial guess
    for the solution.
    ksptype and pctype can be ('richardson', 'jacobi'), ('richardson', 'sor')
    or ('chebyshev', 'jacobi') for example '''

    ksp = PETSc.KSP().create()
    ksp.setOptionsPrefix('smoother3_')
    petsc_options['smoother3_ksp_type'] = ksptype
    ksp.setNormType(PETSc.KSP.NormType.NONE)
    pc = ksp.getPC()
    pc.setOptionsPrefix('smpc3_')
    petsc_options['smpc3_pc_type'] = pctype
    petsc_options['smoother3_ksp_initial_guess_nonzero'] = True
    petsc_options['smoother3_ksp_chebyshev_eigenvalues'] = '0.2690010, \
                                                            2.689671'
    ksp.setTolerances(max_it=Ng)
    ksp.setOperators(Ag)
    ksp.setFromOptions()
    ksp.solve(bg, igg)
    pc.destroy()
    ksp.destroy()


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


def smoother_cor(Ag, bg, Ng, igg, lev, ksptype, pctype, Acor, poordof):

    for i in range(Ng):

        if lev == 0:
            smoother_local(Ag, bg, igg, Acor, poordof)
            smoother_l1(Ag, bg, 1, igg, ksptype, pctype)
            smoother_local(Ag, bg, igg, Acor, poordof)
        if lev == 1:
            smoother_local(Ag, bg, igg, Acor, poordof)
            smoother_l2(Ag, bg, 1, igg, ksptype, pctype)
            smoother_local(Ag, bg, igg, Acor, poordof)
        if lev == 2:
            smoother_local(Ag, bg, igg, Acor, poordof)
            smoother_l3(Ag, bg, 1, igg, ksptype, pctype)
            smoother_local(Ag, bg, igg, Acor, poordof)

    return igg


def residual(Ah, bh, xh):
    '''a function to calculate the residual
    Ah is the matrix, bh is the rhs, xh is the approximation'''
    resh = bh - Ah * xh
    normr = PETSc.Vec.norm(resh, 2)
    return normr


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
            smoother_cor(Ahlist[i], blist[i], nu, uhlist[i], i, ksptype, pctype,
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
            smoother_cor(Ahlist[j], blist[j], nu, uhlist[j], j, ksptype, pctype,
                         Acorlist[j], poorlist[j])

        # calculate the relative residual
        # res4 = residual(Ah, bh, uhlist[0]) / r0

    return uhlist[0]

# ================================================================


def mgcg(Ah, bh, igh, Ncg, Ahlist, prolongation, restriction, Nmg, Nle, nu,
         ksptype, pctype, Acorlist, poorlist):
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
            ksptype, pctype, Acorlist, poorlist)
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
                 ksptype, pctype, Acorlist, poorlist)
        beta = (rk2.dot(zk2)) / (rk.dot(zk))
        y1 = beta * pk
        pk = (zk2 + y1).copy()
        rk = rk2.copy()
        zk = zk2.copy()

        res4 = residual(Ah, bh, xk)
        print('residual after', ite + 1, 'iterations of CG is:',
              res4/r0)
    return xk


# ===============================================================

# Read the meshes.
Vspace = []

mesh = Mesh("./meshes/level1_bad.xml")
V = VectorFunctionSpace(mesh, 'P', 1)
Vspace.append(V)

mesh1 = Mesh("./meshes/level2_bad.xml")
V1 = VectorFunctionSpace(mesh1, 'P', 1)
Vspace.append(V1)

mesh2 = Mesh("./meshes/level3_bad.xml")
V2 = VectorFunctionSpace(mesh2, 'P', 1)
Vspace.append(V2)

mesh3 = Mesh("./meshes/level4_bad.xml")
V3 = VectorFunctionSpace(mesh3, 'P', 1)
Vspace.append(V3)


nl = len(Vspace)
# ==========================================================================

# Find the transfer operators, puse is the prolongation operator list
# note the order is from fine to coarse
puse = []
for il in range(nl-1):
    pmat = PETScDMCollection.create_transfer_matrix(Vspace[il+1], Vspace[il])
    pmat = pmat.mat()
    puse.append(pmat)
# ===================================================================
# Scaled variables
E = 69e9
nu = 0.33
mu = E/(2.0*(1.0 + nu))
lambda_ = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))


# Dirichlet boundary condition
tol = 1E-14


def clamped_boundary(x, on_boundary):
    return on_boundary and near(x[2], -1, tol)


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
        return on_boundary and near(x[2], 6, tol)


NeumanBoundary().mark(boundaries, 1)


# Define outer surface measure aware of Dirichlet and Neumann boundaries
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)


# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0, 0, -10**3))
T = Constant((0, 0, 10**3))
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
bad_level1 = find_low.find_low(mesh, V)
bad_level2 = find_low.find_low(mesh1, V1)
bad_level3 = find_low.find_low(mesh2, V2)
Blist = [bad_level1, bad_level2, bad_level3]


corlist = makecorrection(Alist, Blist)

# =========================================================

igh = as_backend_type(Function(V).vector())
igh[:] = 0.0
fph = igh.vec()
# Multigrid conjugate gradient
print('Initial residual is:', residual(A, b, fph))
wh = mgcg(A, b, fph, 30, Alist, puse, ruse, 1, nl, 2,
          'chebyshev', 'jacobi', corlist, Blist)

