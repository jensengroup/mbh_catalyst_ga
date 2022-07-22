from rdkit import Chem
from rdkit.Chem import AllChem

import subprocess
import concurrent.futures

import os
import random
import shutil
import string
import sys
import numpy as np
from datetime import datetime

def write_xtb_input_files(fragment, name, destination="."):
    number_of_atoms = fragment.GetNumAtoms()
    symbols = [a.GetSymbol() for a in fragment.GetAtoms()]
    conformers = fragment.GetConformers()
    file_paths = []
    for i, conf in enumerate(conformers):
        conf_path = os.path.join(destination, f"conf{i:03d}")
        os.makedirs(conf_path)
        file_name = f"{name}{i:03d}.xyz"
        file_path = os.path.join(conf_path, file_name)
        with open(file_path, "w") as _file:
            _file.write(str(number_of_atoms) + "\n")
            _file.write(f"{Chem.MolToSmiles(fragment)}\n")
            for atom, symbol in enumerate(symbols):
                p = conf.GetAtomPosition(atom)
                line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                _file.write(line)
        file_paths.append(file_path)
    return file_paths


def run_xtb(args):
    xyz_file, xtb_cmd, numThreads = args
    print(f"running {xyz_file} on {numThreads} core(s) starting at {datetime.now()}")
    cwd = os.path.dirname(xyz_file)
    xyz_file = os.path.basename(xyz_file)
    cmd = f"{xtb_cmd} -- {xyz_file} | tee out.out"
    os.environ["OMP_NUM_THREADS"] = f"{numThreads},1"
    os.environ["MKL_NUM_THREADS"] = f"{numThreads}"
    os.environ["OMP_STACKSIZE"] = "2G"
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
        cwd=cwd,
    )
    output, err = popen.communicate()
    energy = read_energy(output, err)
    return energy


def read_energy(output, err):
    if not "normal termination" in err:
        raise Warning(err)
    lines = output.splitlines()
    energy = None
    structure_block = False
    atoms = []
    coords = []
    for l in lines:
        if "final structure" in l:
            structure_block = True
        elif structure_block:
            s = l.split()
            if len(s) == 4:
                atoms.append(s[0])
                coords.append(list(map(float, s[1:])))
            elif len(s) == 0:
                structure_block = False
        elif "TOTAL ENERGY" in l:
            energy = float(l.split()[3])
    return energy, {"atoms": atoms, "coords": coords}


def xtb_optimize(
    mol,
    gbsa="methanol",
    alpb=None,
    opt_level="tight",
    input=None,
    name=None,
    cleanup=False,
    numThreads=1,
):
    # check mol input
    assert isinstance(mol, Chem.rdchem.Mol)
    if mol.GetNumAtoms(onlyExplicit=True) < mol.GetNumAtoms(onlyExplicit=False):
        raise Exception("Implicit Hydrogens")
    conformers = mol.GetConformers()
    n_confs = len(conformers)
    if not conformers:
        raise Exception("Mol is not embedded")
    elif not conformers[-1].Is3D():
        raise Exception("Conformer is not 3D")

    if not name:
        name = "tmp_" + "".join(
            random.choices(string.ascii_uppercase + string.digits, k=4)
        )

    # set SCRATCH if environmental variable
    try:
        scr_dir = os.environ["SCRATCH"]
    except:
        scr_dir = os.getcwd()
    print(f"SCRATCH DIR = {scr_dir}")

    charge = Chem.GetFormalCharge(mol)
    xyz_files = write_xtb_input_files(
        mol, "xtbmol", destination=os.path.join(scr_dir, name)
    )

    # xtb options
    XTB_OPTIONS = {
        "opt": opt_level,
        "chrg": charge,
        "gbsa": gbsa,
        "alpb": alpb,
        "input": input,
    }

    cmd = "xtb"
    for key, value in XTB_OPTIONS.items():
        if value:
            cmd += f" --{key} {value}"

    workers = np.min([numThreads, n_confs])
    cpus_per_worker = numThreads // workers
    args = [(xyz_file, cmd, cpus_per_worker) for xyz_file in xyz_files]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(run_xtb, args)

    energies = []
    geometries = []
    for e, g in results:
        energies.append(e)
        geometries.append(g)

    minidx = np.argmin(energies)

    # Clean up
    if cleanup:
        shutil.rmtree(name)

    return energies[minidx], geometries[minidx]
