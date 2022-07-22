from dataclasses import dataclass, field

from rdkit import Chem

hartree2kcalmol = 627.5094740631


@dataclass
class Individual:
    rdkit_mol: Chem.rdchem.Mol = field(repr=False, compare=False)
    idx: str = field(default=None, compare=True, repr=True)
    smiles: str = field(init=False, compare=True, repr=True)
    score: float = field(default=None, repr=False, compare=False)
    energy: float = field(default=None, repr=False, compare=False)
    sa_score: float = field(default=None, repr=False, compare=False)
    structure: tuple = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        self.smiles = Chem.MolToSmiles(
            Chem.MolFromSmiles(Chem.MolToSmiles(self.rdkit_mol))
        )


def read_xyz(content, element_symbols=False):
    content = content.strip().split("\n")
    del content[:2]

    atomic_symbols = []
    atomic_positions = []

    for line in content:
        line = line.split()
        atomic_symbols.append(line[0])
        atomic_positions.append(list(map(float, line[1:])))

    if element_symbols:
        pt = Chem.GetPeriodicTable()
        atomic_numbers = list()
        for atom in atomic_symbols:
            atomic_numbers.append(pt.GetAtomicNumber(atom))
        return atomic_numbers, atomic_positions
    else:
        return atomic_symbols, atomic_positions
