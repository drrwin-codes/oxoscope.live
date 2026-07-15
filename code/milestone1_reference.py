"""
Milestone 1: Does the reaction actually fire?

Before building any 3D viewer, we need to answer one question honestly:
if we place a hydroxyl radical near C8 of guanine and run short MD with
UMA, does the radical actually bond to C8 in a reasonable fraction of
short trajectories? If yes, this project is real. If not, we need to
either nudge conditions (temperature, starting distance) or rethink scope
BEFORE investing time in visualization.

Run this locally once your UMA access is approved (huggingface.co/facebook/UMA).

Setup:
    pip install rdkit ase huggingface_hub
    pip install fairchem-core   # gives you FAIRChemCalculator + pretrained_mlip
    huggingface-cli login       # paste your token once you have UMA access
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate import pretrained_mlip


# ---------- Step 1: build the reactive starting geometry ----------
# (This part is already verified working -- tested separately before
# writing this script.)

def build_guanine_OH_complex(approach_dist=3.2, seed=42):
    """Guanine + hydroxyl radical, oriented for pi-face attack at C8,
    guanine's known site of oxidative radical attack."""
    guanine_smiles = "Nc1nc2[nH]cnc2c(=O)[nH]1"
    mol = Chem.MolFromSmiles(guanine_smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    c8_idx = 5  # the C bonded to N, N, H -- confirmed via neighbor inspection
    c8_pos = np.array(conf.GetAtomPosition(c8_idx))

    ring_atoms = [1, 3, 5]
    p0, p1, p2 = [np.array(conf.GetAtomPosition(i)) for i in ring_atoms]
    normal = np.cross(p1 - p0, p2 - p0)
    normal = normal / np.linalg.norm(normal)

    O_pos = c8_pos + normal * approach_dist
    H_pos = O_pos + normal * 0.97

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()] + ["O", "H"]
    positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    positions += [list(O_pos), list(H_pos)]

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 2  # doublet -- one unpaired electron from the OH radical
    return atoms, c8_idx, len(symbols) - 2  # also return index of the radical O atom


# ---------- Step 2: run one short trajectory, track the reaction coordinate ----------

def run_single_trajectory(calc, temp_k=350, timestep_fs=0.5, steps=2000,
                           bond_forming_cutoff=1.7, seed=0):
    """Returns (reacted: bool, distance_trace: list[float])
    bond_forming_cutoff in Angstrom -- a C-O single bond is ~1.4 A, so if the
    O...C8 distance drops below ~1.7 A and stays there, treat that as reacted."""
    atoms, c8_idx, o_idx = build_guanine_OH_complex(seed=seed)
    atoms.calc = calc

    MaxwellBoltzmannDistribution(atoms, temperature_K=temp_k, rng=np.random.RandomState(seed))
    dyn = Langevin(atoms, timestep=timestep_fs * units.fs,
                    temperature_K=temp_k, friction=0.01 / units.fs)

    distances = []

    def record():
        d = atoms.get_distance(o_idx, c8_idx, mic=False)
        distances.append(d)

    dyn.attach(record, interval=5)
    dyn.run(steps=steps)

    # Reacted if the LAST several recorded distances are consistently
    # below the bond-forming cutoff (not just a brief close pass)
    tail = distances[-10:]
    reacted = all(d < bond_forming_cutoff for d in tail)
    return reacted, distances


# ---------- Step 3: swarm of trajectories -- this IS the honest science ----------
# A single trajectory reacting or not reacting tells you almost nothing.
# Running many short trajectories with randomized initial velocities and
# reporting the REACTIVE FRACTION is standard practice in reactive
# scattering studies, and it's also exactly what gives you the
# "not every collision reacts" honesty that makes this project defensible.

def run_swarm(n_trajectories=20, temp_k=350):
    print("Loading UMA model (this takes a minute the first time)...")
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
    calc = FAIRChemCalculator(predictor, task_name="omol")

    import time
    results = []
    for i in range(n_trajectories):
        print(f"\nTrajectory {i+1}/{n_trajectories} (seed={i})...")
        t0 = time.time()
        reacted, distances = run_single_trajectory(calc, temp_k=temp_k, seed=i)
        elapsed = time.time() - t0
        print(f"  Reacted: {reacted}  |  final O...C8 distance: {distances[-1]:.2f} A"
              f"  |  took {elapsed:.1f} sec")
        results.append({"seed": i, "reacted": reacted, "distances": distances})

    n_reacted = sum(r["reacted"] for r in results)
    print(f"\n{'='*50}")
    print(f"REACTIVE FRACTION at {temp_k}K: {n_reacted}/{n_trajectories} "
          f"({100*n_reacted/n_trajectories:.0f}%)")
    print(f"{'='*50}")

    if n_reacted == 0:
        print("\nNo reactions observed. Before concluding this doesn't work, try:")
        print("  - closer starting distance (e.g. approach_dist=2.6-2.8)")
        print("  - higher temperature (e.g. 500K) -- this is standard practice")
        print("    for accelerating rare events, just be honest about it in the writeup")
        print("  - more steps per trajectory")
    elif n_reacted == n_trajectories:
        print("\nEvery trajectory reacted -- good sign this is real chemistry, but")
        print("also worth showing SOME non-reactive trajectories for honesty.")
        print("Try increasing approach_dist slightly to get a more realistic mix.")

    return results


if __name__ == "__main__":
    # Testing with just 1 trajectory first to see how long CPU takes per run.
    # Once you know the timing, bump n_trajectories back up to 20 for the real test.
    results = run_swarm(n_trajectories=1, temp_k=350)
