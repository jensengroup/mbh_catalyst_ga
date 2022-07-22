from rdkit import Chem
from rdkit.Chem import AllChem
import os
import sys
import copy
import numpy as np

catalyst_dir = os.path.dirname(__file__)

sys.path.append(catalyst_dir)

from xtb_utils import xtb_optimize
from utils import read_xyz, hartree2kcalmol
from make_structures import connect_cat_2d, ConstrainedEmbedMultipleConfsMultipleFrags


ts_file = os.path.join(catalyst_dir, "input_files/ts3_dummy.sdf")
ts_dummy = Chem.SDMolSupplier(ts_file, removeHs=False, sanitize=True)[0]

frag_energies = np.sum(
    [-8.232710038092, -19.734652802142, -32.543971411432]
)  # 34 atoms


def ts_scoring(cat, idx=(0, 0), ncpus=1, n_confs=10, cleanup=False):
    """Calculates electronic energy difference in kcal/mol between TS and reactants

    Args:
        cat (rdkit.Mol): Molecule containing one tertiary amine
        n_confs (int, optional): Nubmer of confomers used for embedding. Defaults to 10.
        cleanup (bool, optional): Clean up files after calculation. 
                                  Defaults to False, needs to be False to work with submitit.

    Returns:
        Tuple: Contains energy difference, Geom of TS and Geom of Cat
    """

    ts2ds = connect_cat_2d(ts_dummy, cat)
    if len(ts2ds) > 1:
        print(
            f"{Chem.MolToSmiles(Chem.RemoveHs(cat))} contains more than one tertiary amine"
        )
    ts2d = ts2ds[0]

    # Embed TS
    ts3d = ConstrainedEmbedMultipleConfsMultipleFrags(
        mol=ts2d,
        core=ts_dummy,
        numConfs=n_confs,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    # Calc Energy of TS
    ts3d_energy, ts3d_geom = xtb_optimize(
        ts3d,
        gbsa="methanol",
        opt_level="tight",
        name=f"{idx[0]:03d}_{idx[1]:03d}_ts",
        input=os.path.join(catalyst_dir, "input_files/constr.inp"),
        numThreads=ncpus,
        cleanup=cleanup,
    )

    # Embed Catalyst
    cat3d = copy.deepcopy(cat)
    cat3d = Chem.AddHs(cat3d)
    cids = Chem.rdDistGeom.EmbedMultipleConfs(
        cat3d, numConfs=n_confs, pruneRmsThresh=0.1
    )
    if len(cids) == 0:
        raise ValueError(
            f"Could not embed catalyst {Chem.MolToSmiles(Chem.RemoveHs(cat))}"
        )

    # Calc Energy of Cat
    cat3d_energy, cat3d_geom = xtb_optimize(
        cat3d,
        gbsa="methanol",
        opt_level="tight",
        name=f"{idx[0]:03d}_{idx[1]:03d}_cat",
        numThreads=ncpus,
        cleanup=cleanup,
    )

    # Calculate electronic activation energy
    De = (ts3d_energy - frag_energies - cat3d_energy) * hartree2kcalmol
    return De, ts3d_geom, cat3d_geom
