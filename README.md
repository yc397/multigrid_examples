
Various implementations of multigrid using FEniCS and PETSc
-----------------------------------------------------------
The repository contains the codes of a straightforward and clear 
implementation of multigrid methods.

A standard code includes two parts:

(i) The finite element analysis by FEniCS

(ii) The multigrid procedure by PETSc4py

In terms of multigrid, the transfer operators are constructed
by FEniCS, and the smoothers are implemented by PETSc.
The multigrid cycles are then constructed by our own codes.

These codes requires the FEniCS version 2019.1.0, and PETSc version 3.12.
