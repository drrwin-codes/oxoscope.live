"""
# Milestone 1: Does the OH radical + guanine reaction actually fire?

**Before running:** go to `Runtime > Change runtime type` and select a GPU
(T4 is fine, it's free). Without this, you'll get the same CPU speed as
your laptop and none of the benefit of switching here.
"""

!pip install -q rdkit ase huggingface_hub fairchem-core

"""## Log in to Hugging Face
Paste your token when prompted (same one you used with `hf auth login` locally).
You can find/create it at huggingface.co/settings/tokens
"""

from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get('HF_TOKEN'))

import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate import pretrained_mlip
from ase.constraints import Hookean

# Quick sanity check -- confirm Colab actually gave us a GPU
import torch
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: no GPU detected. Go to Runtime > Change runtime type > select GPU, then rerun.")

def build_guanine_OH_complex(approach_dist=2.8, seed=42):
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
    return atoms, c8_idx, len(symbols) - 2

def run_single_trajectory(calc, temp_k=350, timestep_fs=0.5, steps=2000,
                           bond_forming_cutoff=1.7, seed=0, approach_dist=2.8, cage_distance=4.0, cage_k=1.0):
    atoms, c8_idx, o_idx = build_guanine_OH_complex(approach_dist=approach_dist, seed=seed)
    atoms.calc = calc
    atoms.set_constraint(Hookean(a1=o_idx, a2=c8_idx, rt=cage_distance, k=cage_k))

    MaxwellBoltzmannDistribution(atoms, temperature_K=temp_k, rng=np.random.RandomState(seed))
    dyn = Langevin(atoms, timestep=timestep_fs * units.fs,
                    temperature_K=temp_k, friction=0.01 / units.fs)

    distances = []
    step_counter = {"n": 0}
    t_start = time.time()

    def record():
        d = atoms.get_distance(o_idx, c8_idx, mic=False)
        distances.append(d)
        step_counter["n"] += 5
        if step_counter["n"] % 200 == 0:
            elapsed = time.time() - t_start
            print(f"    step {step_counter['n']}/{steps}  |  O...C8 = {d:.2f} A  |  {elapsed:.0f}s elapsed")

    dyn.attach(record, interval=5)
    dyn.run(steps=steps)

    tail = distances[-10:]
    reacted = all(d < bond_forming_cutoff for d in tail)
    return reacted, distances

def run_swarm(n_trajectories=5, temp_k=350, approach_dist=2.8, cage_distance=3.0):
    print("Loading UMA model (first time only, downloads weights)...")
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
    calc = FAIRChemCalculator(predictor, task_name="omol")

    results = []
    for i in range(n_trajectories):
        print(f"\nTrajectory {i+1}/{n_trajectories} (seed={i})...")
        t0 = time.time()
        reacted, distances = run_single_trajectory(calc, temp_k=temp_k, seed=i, approach_dist=approach_dist, cage_distance=cage_distance)
        elapsed = time.time() - t0
        print(f"  Reacted: {reacted}  |  final O...C8 distance: {distances[-1]:.2f} A  |  took {elapsed:.1f} sec")
        results.append({"seed": i, "reacted": reacted, "distances": distances})

    n_reacted = sum(r["reacted"] for r in results)
    print(f"\n{'='*50}")
    print(f"REACTIVE FRACTION at {temp_k}K: {n_reacted}/{n_trajectories} ({100*n_reacted/n_trajectories:.0f}%)")
    print(f"{'='*50}")
    return results

# Start small -- 5 trajectories -- to confirm GPU speed before committing to the full 20
results = run_swarm(n_trajectories=5, temp_k=350, approach_dist=2.8, cage_distance=2.2)
