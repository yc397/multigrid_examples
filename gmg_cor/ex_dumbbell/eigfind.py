from petsc4py import PETSc
from petsc4py.PETSc import Vec
from slepc4py import SLEPc


def eigfind(mac):
    n = mac.getSize()
    Dt = Vec()
    mac.getDiagonal(Dt)
    Dt.reciprocal()
    Dpt = PETSc.Mat().createAIJ([n, n])
    Dpt.setUp()
    Dpt.setDiagonal(Dt)
    Dpt.assemblyBegin()
    Dpt.assemblyEnd()
    tarmat = Dpt*mac

    def solve_eigensystem(Ac, problem_type=SLEPc.EPS.ProblemType.HEP):

        # Create the result vectors
        xr, xi = Ac.createVecs()

        # Setup the eigensolver
        E = SLEPc.EPS().create()
        E.setOperators(Ac, None)
        E.setDimensions(2, PETSc.DECIDE)
        E.setProblemType(problem_type)
        E.setFromOptions()

        # Solve the eigensystem
        E.solve()

        print("")
        its = E.getIterationNumber()
        print("Number of iterations of the method: %i" % its)
        sol_type = E.getType()
        print("Solution method: %s" % sol_type)
        nev, ncv, mpd = E.getDimensions()
        print("Number of requested eigenvalues: %i" % nev)
        tol, maxit = E.getTolerances()
        print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
        nconv = E.getConverged()
        print("Number of converged eigenpairs: %d" % nconv)
        if nconv > 0:
            print("")
            print("        k          ||Ax-kx||/||kx|| ")
            print("----------------- ------------------")
            for i in range(nconv):
                k = E.getEigenpair(i, xr, xi)
                error = E.computeError(i)
                if k.imag != 0.0:
                    print(" %9f%+9f j  %12g" % (k.real, k.imag, error))
                else:
                    print(" %12f       %12g" % (k.real, error))
            print("")

    solve_eigensystem(tarmat)
