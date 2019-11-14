from petsc4py import PETSc
from petsc4py.PETSc import Mat
import eigfind


def matrix_split(Aww, badset):
    goodset = []
    nw = Aww.getSize()[0]
    for ki in range(nw):
        if ki not in badset:
            goodset.append(ki)
    goodset.sort()
    gtt = PETSc.IS()
    btt = PETSc.IS()
    gtt.createGeneral(goodset)
    btt.createGeneral(badset)
    A1 = Mat()
    A2 = Mat()
    Aww.createSubMatrix(gtt, gtt, A1)
    Aww.createSubMatrix(btt, btt, A2)
    eigfind.eigfind(A1)
    eigfind.eigfind(A2)
    return A1, A2
