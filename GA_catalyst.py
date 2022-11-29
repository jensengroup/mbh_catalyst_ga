import heapq
import random
import shutil
from pathlib import Path
from typing import List

import numpy as np
import submitit
from rdkit import Chem
from scipy.stats import rankdata
from tabulate import tabulate

import crossover as co
import filters
import mutate as mu
from catalyst import ts_scoring
from catalyst.utils import Individual
from sa import neutralize_molecules, sa_target_score_clipped

SLURM_SETUP = {
    "slurm_partition": "kemi1",
    "timeout_min": 30,
    "slurm_array_parallelism": 10,
}


def slurm_scoring(sc_function, population, ids, cpus_per_task=4, cleanup=False):
    """Evaluates a scoring function for population on SLURM cluster

    Args:
        sc_function (function): Scoring function which takes molecules and id (int,int) as input
        population (List): List of rdkit Molecules
        ids (List of Tuples of Int): Index of each molecule (Generation, Individual)

    Returns:
        List: List of results from scoring function
    """
    executor = submitit.AutoExecutor(
        folder="scoring_tmp",
        slurm_max_num_timeout=0,
    )
    executor.update_parameters(
        name=f"sc_g{ids[0][0]}",
        cpus_per_task=cpus_per_task,
        slurm_mem_per_cpu="1GB",
        timeout_min=SLURM_SETUP["timeout_min"],
        slurm_partition=SLURM_SETUP["slurm_partition"],
        slurm_array_parallelism=SLURM_SETUP["slurm_array_parallelism"],
    )
    args = [cpus_per_task for p in population]
    jobs = executor.map_array(sc_function, population, ids, args)

    results = [
        catch(job.result, handle=lambda e: (np.nan, None)) for job in jobs
    ]  # catch submitit exceptions and return same output as scoring function (np.nan, None) for (energy, geometry)
    if cleanup:
        shutil.rmtree("scoring_tmp")
    return results


def catch(func, *args, handle=lambda e: e, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return handle(e)


def make_initial_population(population_size, file_name, random=True):
    if random:
        with open(file_name) as fin:
            sample = heapq.nlargest(population_size, fin, key=lambda L: random.random())
    else:
        with open(file_name) as fin:
            sample = [smiles for smiles in fin][:population_size]
    population = [Chem.MolFromSmiles(smi.rstrip()) for smi in sample]

    return population


def calculate_normalized_fitness(scores):
    sum_scores = sum(scores)
    normalized_fitness = [score / sum_scores for score in scores]

    return normalized_fitness


def calculate_fitness(scores, minimization=False, selection="roulette", selection_pressure=None):
    if minimization:
        scores = [-s for s in scores]
    if selection == "roulette":
        fitness = scores
    elif selection == "rank":
        scores = [float("-inf") if np.isnan(x) else x for x in scores]  # works for minimization
        ranks = rankdata(scores, method="ordinal")
        n = len(ranks)
        if selection_pressure:
            fitness = [
                2 - selection_pressure + (2 * (selection_pressure - 1) * (rank - 1) / (n - 1))
                for rank in ranks
            ]
        else:
            fitness = [r / n for r in ranks]
    else:
        raise ValueError(
            f"Rank-based ('rank') or roulette ('roulette') selection are available, you chose {selection}."
        )

    return fitness


def make_mating_pool(population, fitness, mating_pool_size):
    mating_pool = []
    for i in range(mating_pool_size):
        mating_pool.append(random.choices(population, weights=fitness, k=1)[0])
    return mating_pool


def reproduce(mating_pool, population_size, mutation_rate, molecule_filter, generation):
    new_population = []
    counter = 0
    while len(new_population) < population_size:
        if random.random() > mutation_rate:
            parent_A = random.choice(mating_pool)
            parent_B = random.choice(mating_pool)
            new_child = co.crossover(parent_A.rdkit_mol, parent_B.rdkit_mol, molecule_filter)
            if new_child != None:
                idx = (generation, counter)
                counter += 1
                new_child = Individual(rdkit_mol=new_child, idx=idx)
                new_population.append(new_child)
        else:
            parent = random.choice(mating_pool)
            mutated_child = mu.mutate(parent.rdkit_mol, 1, molecule_filter)
            if mutated_child != None:
                idx = (generation, counter)
                counter += 1
                mutated_child = Individual(
                    rdkit_mol=mutated_child,
                    idx=idx,
                )
                new_population.append(mutated_child)
    return new_population


def sanitize(population, population_size, prune_population):
    if prune_population:
        sanitized_population = []
        for ind in population:
            if ind.smiles not in [si.smiles for si in sanitized_population]:
                sanitized_population.append(ind)
    else:
        sanitized_population = population

    sanitized_population.sort(
        key=lambda x: float("inf") if np.isnan(x.score) else x.score
    )  # np.nan is highest value, works for minimization of score

    new_population = sanitized_population[:population_size]
    return new_population  # selects individuals with lowest values


def reweigh_scores_by_sa(population: List[Chem.Mol], scores: List[float]) -> List[float]:
    """Reweighs scores with synthetic accessibility score
    :param population: list of RDKit molecules to be re-weighted
    :param scores: list of docking scores
    :return: list of re-weighted docking scores
    """
    sa_scores = [sa_target_score_clipped(p) for p in population]
    return sa_scores, [
        ns * sa for ns, sa in zip(scores, sa_scores)
    ]  # rescale scores and force list type


def print_results(population, fitness, generation):
    print(f"\nGeneration {generation+1}", flush=True)
    print(
        tabulate(
            [
                [ind.idx, fit, ind.score, ind.energy, ind.sa_score, ind.smiles]
                for ind, fit in zip(population, fitness)
            ],
            headers=[
                "idx",
                "normalized fitness",
                "score",
                "energy",
                "sa score",
                "smiles",
            ],
        ),
        flush=True,
    )


def GA(args):
    (
        population_size,
        file_name,
        scoring_function,
        generations,
        mating_pool_size,
        mutation_rate,
        scoring_args,
        prune_population,
        seed,
        minimization,
        selection_method,
        selection_pressure,
        molecule_filters,
        path,
    ) = args

    np.random.seed(seed)
    random.seed(seed)

    generations_file = Path(path) / "generations.gen"
    generations_list = []

    molecules = make_initial_population(population_size, file_name, random=False)

    # write starting popultaion
    pop_file = Path(path) / "starting_pop.smi"
    with open(str(generations_file.resolve()), "w+") as f:
        f.writelines([str(Chem.MolToSmiles(m) for m in molecules)])

    ids = [(0, i) for i in range(len(molecules))]
    results = slurm_scoring(scoring_function, molecules, ids)
    energies = [res[0] for res in results]
    geometries = [res[1] for res in results]

    prescores = [energy - 100 for energy in energies]
    sa_scores, scores = reweigh_scores_by_sa(neutralize_molecules(molecules), prescores)

    population = [
        Individual(
            idx=idx,
            rdkit_mol=mol,
            score=score,
            energy=energy,
            sa_score=sa_score,
            structure=structure,
        )
        for idx, mol, score, energy, sa_score, structure in zip(
            ids, molecules, scores, energies, sa_scores, geometries
        )
    ]
    population = sanitize(population, population_size, False)

    fitness = calculate_fitness(
        [ind.score for ind in population],
        minimization,
        selection_method,
        selection_pressure,
    )
    fitness = calculate_normalized_fitness(fitness)

    print_results(population, fitness, -1)

    for generation in range(generations):
        mating_pool = make_mating_pool(population, fitness, mating_pool_size)

        new_population = reproduce(
            mating_pool,
            population_size,
            mutation_rate,
            molecule_filters,
            generation + 1,
        )

        new_resuls = slurm_scoring(
            scoring_function,
            [ind.rdkit_mol for ind in new_population],
            [ind.idx for ind in new_population],
        )
        new_energies = [res[0] for res in new_resuls]
        new_geometries = [res[1] for res in new_resuls]

        new_prescores = [energy - 100 for energy in new_energies]
        new_sa_scores, new_scores = reweigh_scores_by_sa(
            neutralize_molecules([ind.rdkit_mol for ind in new_population]),
            new_prescores,
        )

        for ind, score, energy, sa_score, structure in zip(
            new_population,
            new_scores,
            new_energies,
            new_sa_scores,
            new_geometries,
        ):
            ind.score = score
            ind.energy = energy
            ind.sa_score = sa_score
            ind.structure = structure

        population = sanitize(population + new_population, population_size, prune_population)

        fitness = calculate_fitness(
            [ind.score for ind in population],
            minimization,
            selection_method,
            selection_pressure,
        )
        fitness = calculate_normalized_fitness(fitness)

        generations_list.append([ind.idx for ind in population])
        print_results(population, fitness, generation)

    with open(str(generations_file.resolve()), "w+") as f:
        f.writelines(str(generations_list))


### ----------------------------------------------------------------------------------


if __name__ == "__main__":

    package_directory = Path(__file__).parent.resolve()

    co.average_size = 8.0  # 14 24.022840038202613
    co.size_stdev = 4.0  # 8 4.230907997270275
    population_size = 8
    file_name = package_directory / "ZINC_amines.smi"
    scoring_function = ts_scoring
    generations = 5
    mating_pool_size = population_size
    mutation_rate = 0.50
    scoring_args = None
    prune_population = True
    seed = 101
    minimization = True
    selection_method = "rank"
    selection_pressure = 1.5
    molecule_filters = filters.get_molecule_filters(
        ["MBH"], package_directory / "filters/alert_collection.csv"
    )
    # file_name = argv[-1]

    path = "."

    args = [
        population_size,
        file_name,
        scoring_function,
        generations,
        mating_pool_size,
        mutation_rate,
        scoring_args,
        prune_population,
        seed,
        minimization,
        selection_method,
        selection_pressure,
        molecule_filters,
        path,
    ]

    # Run GA
    GA(args)
