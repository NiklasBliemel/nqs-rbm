from jVMC import operator

def make_TFI_hamiltonian( L: int, J: float=1.0, g: float=1.0) -> operator.Operator:
    hamiltonian = operator.BranchFreeOperator()
    for l in range(L):
        hamiltonian.add(operator.scal_opstr(-J, (operator.Sz(l), operator.Sz((l + 1)%L))))
        hamiltonian.add(operator.scal_opstr(-g, (operator.Sx(l), )))
    return hamiltonian