from jVMC import operator

def make_TFI_hamiltonian( L: int, g: float) -> operator.Operator:
    hamiltonian = operator.BranchFreeOperator()
    for l in range(L):
        hamiltonian.add(operator.scal_opstr(-1.0, (operator.Sz(l), operator.Sz((l + 1)%L))))
        hamiltonian.add(operator.scal_opstr(-g, (operator.Sx(l), )))
    return hamiltonian