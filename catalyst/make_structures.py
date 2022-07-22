from rdkit import Chem
from rdkit.Chem import AllChem

from io import StringIO

import sys

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def connect_cat_2d(mol_with_dummy, cat):
    """Replaces Dummy Atom [*] in Mol with Cat via tertiary Amine, return list of all possible regioisomers"""
    dummy = Chem.MolFromSmiles("*")
    mols = []
    cat = Chem.AddHs(cat)
    AllChem.AssignStereochemistry(cat)
    tert_amines = cat.GetSubstructMatches(Chem.MolFromSmarts("[#7X3;H0;D3;!+1]"))
    if len(tert_amines) == 0:
        raise Exception(
            f"{Chem.MolToSmiles(Chem.RemoveHs(cat))} constains no tertiary amine."
        )
    for amine in tert_amines:
        mol = AllChem.ReplaceSubstructs(
            mol_with_dummy, dummy, cat, replacementConnectionPoint=amine[0]
        )[0]
        quart_amine = mol.GetSubstructMatch(Chem.MolFromSmarts("[#7X4;H0;D4;!+1]"))[0]
        mol.GetAtomWithIdx(quart_amine).SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        mol.RemoveAllConformers()
        mols.append(mol)
    return mols


def frags2bonded(mol, atoms2join=((1, 11), (11, 29))):
    make_bonded = Chem.EditableMol(mol)
    for atoms in atoms2join:
        i, j = atoms
        make_bonded.AddBond(i, j)
    mol_bonded = make_bonded.GetMol()
    # Chem.SanitizeMol(mol_bonded)
    return mol_bonded


def bonded2frags(mol, atoms2frag=((1, 11), (11, 29))):
    make_frags = Chem.EditableMol(mol)
    for atoms in atoms2frag:
        i, j = atoms
        make_frags.RemoveBond(i, j)
    mol_frags = make_frags.GetMol()
    # Chem.SanitizeMol(mol_frags)
    return mol_frags


def ConstrainedEmbedMultipleConfsMultipleFrags(
    mol,
    core,
    numConfs=10,
    useTethers=True,
    coreConfId=-1,
    randomseed=2342,
    getForceField=AllChem.UFFGetMoleculeForceField,
    numThreads=1,
    force_constant=1e3,
    pruneRmsThresh=1,
    atoms2join=((1, 11), (11, 29)),
):
    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    sio = sys.stderr = StringIO()
    if not AllChem.UFFHasAllMoleculeParams(mol):
        raise Exception(Chem.MolToSmiles(mol), sio.getvalue())

    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    mol_bonded = frags2bonded(mol, atoms2join=atoms2join)
    cids = AllChem.EmbedMultipleConfs(
        mol=mol_bonded,
        numConfs=numConfs,
        coordMap=coordMap,
        randomSeed=randomseed,
        numThreads=numThreads,
        pruneRmsThresh=pruneRmsThresh,
        useRandomCoords=True,
    )
    mol = bonded2frags(mol_bonded, atoms2frag=atoms2join)
    Chem.SanitizeMol(mol)

    cids = list(cids)
    if len(cids) == 0:
        print(coordMap, Chem.MolToSmiles(mol_bonded))
        raise ValueError("Could not embed molecule.")

    algMap = [(j, i) for i, j in enumerate(match)]

    # rotate the embedded conformation onto the core:
    for cid in cids:
        rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=cid, ignoreInterfragInteractions=False
        )
        for i, _ in enumerate(match):
            ff.UFFAddPositionConstraint(i, 0, force_constant)
        ff.Initialize()
        n = 4
        more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        while more and n:
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            n -= 1
        # realign
        rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
    return mol
