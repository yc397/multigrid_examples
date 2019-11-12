from dolfin import (Mesh, VectorFunctionSpace, PETScDMCollection,
                    DirichletBC, near, TrialFunction, TestFunction,
                    div, Identity, MeshFunction, SubDomain, Measure,
                    Constant, dot, grad, dx, inner, PETScMatrix, PETScVector,
                    assemble_system, as_backend_type, Function)
import numpy as np
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
              res4/r0)
    return xk


# ===============================================================
# Read the meshes.
Vspace = []

mesh = Mesh("./level_1.xml")
V = VectorFunctionSpace(mesh, 'P', 1)
Vspace.append(V)

mesh1 = Mesh("./level_2.xml")
V1 = VectorFunctionSpace(mesh1, 'P', 1)
Vspace.append(V1)

mesh2 = Mesh("./level_3.xml")
V2 = VectorFunctionSpace(mesh2, 'P', 1)
Vspace.append(V2)

mesh3 = Mesh("./level_4.xml")
V3 = VectorFunctionSpace(mesh3, 'P', 1)
Vspace.append(V3)

nl = len(Vspace)
# ==========================================================================

# perturb several nodes to make low quality meshes
# level one
coord = mesh.coordinates()
coord[95550] = np.array([0.47591777, 0.24052105, 3.31886418])
coord[106700] = np.array([0.47591629, 0.24050852, 3.31887405])
coord[106894] = np.array([5.41804831, 1.64820845, 5.64692289])
coord[115739] = np.array([5.41803166, 1.64821801, 5.64692542])
coord[73535] = np.array([5.50960973, 5.8918863, 1.9815105])
coord[121748] = np.array([5.50960565, 5.89189095, 1.98152539])
coord[68916] = np.array([0.13966908, 5.26853919, 5.63403176])
coord[75803] = np.array([0.13966732, 5.26853227, 5.63401703])

# level two
coord1 = mesh1.coordinates()
coord1[3546] = np.array([5.86806422, 3.17114982, 5.68377472])
coord1[18310] = np.array([5.86802687, 3.17108845, 5.68375266])
coord1[4793] = np.array([3.80139399, 5.67744588, 0.14650002])
coord1[14340] = np.array([3.80142903, 5.67745504, 0.14655154])
coord1[18635] = np.array([1.90427582, 5.46539432, 5.85642511])
coord1[18713] = np.array([1.90427047, 5.46542908, 5.85642393])
coord1[10542] = np.array([5.05823635, 0.28695481, 3.41446808])
coord1[16748] = np.array([5.06124513, 0.28518137, 3.41349033])

# level three
coord2 = mesh2.coordinates()
coord2[2590] = np.array([5.23436637, 0.27726406, 4.21149236])
coord2[2076] = np.array([5.23430188, 0.27723736, 4.21156968])
coord2[1263] = np.array([0.16958664, 4.01511841, 5.56267808])
coord2[1324] = np.array([0.16958664, 4.0151591,  5.56273487])
coord2[1574] = np.array([5.07777822, 0.07400174, 3.67485144])
coord2[2060] = np.array([5.0774665, 0.07430691, 3.67452278])
coord2[2750] = np.array([4.35516738, 5.89244318, 5.49917535])

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
E = 1.0e9
nu = 0.3
mu = E/(2.0*(1.0 + nu))
lambda_ = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))


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
        return on_boundary and near(x[0], 6, tol)


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
igh = as_backend_type(Function(V).vector())
igh[:] = 0.0
fph = igh.vec()
# Multigrid conjugate gradient
print('Initial residual is:', residual(A, b, fph))
wh = mgcg(A, b, fph, 15, Alist, puse, ruse, 1, nl, 2, 'richardson', 'sor')
quit()
