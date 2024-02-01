from dolfin import (MeshQuality, Cell, vertices, cells)
import numpy as np


def find_low(mesh, V):

    V2dm = V.dofmap()
    cq2 = MeshQuality.radius_ratios(mesh)
    cq = cq2.array()

    indices = np.where(cq < 0.1)[0]

    dof_set = []
    cell_set = []
    for i in indices:
        cell = Cell(mesh, i)
        for v in vertices(cell):
            for c in cells(v):
                cell_set += [c.index()]
                dof_set.extend(V2dm.cell_dofs(c.index()))

    bad_dofs = list(set(dof_set))
    bad_cells = list(set(cell_set))

    # print('BAD CELLS=', bad_cells)
    # print(len(bad_cells))
    # print('BAD DOFS=', bad_dofs)
    # print(len(bad_dofs))

    return bad_dofs
