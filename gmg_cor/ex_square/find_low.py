from dolfin import (Mesh, FunctionSpace, MeshQuality, Cell, vertices, cells)
import numpy as np

mesh = Mesh("./level_1.xml")
coord = mesh.coordinates()
coord[233] = np.array([0.45286306, 0.55238262])
coord[1083] = np.array([0.45308537, 0.55225124])
coord[1000] = np.array([0.00788574, 0.68094166])
coord[1098] = np.array([0.00788488, 0.69369237])
coord[526] = np.array([0.94154541, 0.89955308])
coord[1052] = np.array([0.94127031, 0.8994045])

V = FunctionSpace(mesh, 'CG', 1)
n = mesh.num_vertices()
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

print('BAD CELLS=', bad_cells)
print(len(bad_cells))
print('BAD DOFS=', bad_dofs)
print(len(bad_dofs))
