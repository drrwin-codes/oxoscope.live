
# **Introduction to Oxoscope**

Reactive molecular dynamics of oxidative DNA damage, using `FAIRChem's UMA machine-learning potential` to
simulate a hydroxyl radical attacking C8 of guanine.
This is the first step toward 8-oxoguanine,
one of the most-studied mutagenic DNA lesions, and a real driver of cancer-relevant genome
instability. Classical force fields can't represent this (no bond-breaking/forming).

---

<img width="1496" height="1034" alt="image" src="https://github.com/user-attachments/assets/e0ac7e0a-9fa2-4e34-887c-110fd5df4e68" />

---

## **Repository Information**

| file | purpose |
|---|---|
| `milestone1_reaction_test.py` | Core reactive MD test: builds the guanine + •OH complex (correct π-face approach geometry at C8, correct doublet spin state), runs a swarm of short trajectories, classifies reacted vs. not via O···C8 distance |
| `oxoscope_colab.ipynb` | Same pipeline, GPU-accelerated on Colab (CPU runs are ~4 min/trajectory; GPU roughly halves that) |
| `capture_trajectory.py` | Saves full per-atom positions at every frame for a single trajectory — needed for visualization, not just the summary distance |
| `reaction_trajectory.xyz` | First captured reactive trajectory (seed=0, 350K, cage_distance=2.2Å) — commits from encounter complex to bonded product in ~400 steps and holds stable |

---

for those interested in reproducing code

```bash
pip install rdkit ase huggingface_hub fairchem-core
hf auth login   # needs UMA access approved at huggingface.co/facebook/UMA
python milestone1_reaction_test.py
```

GPU strongly recommended (`oxoscope_colab.ipynb` on Colab's free T4 tier). CPU works but each trajectory takes **~3.5–4 minutes.**
