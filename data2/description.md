# Quantum Shot Denoising Dataset (v4) — Full Process Description

This document describes what `generate_data_v4.py` does, in what order, where
everything is stored, and what the data does and does not guarantee.

---

## 1. Goal

Build a dataset for training a model that corrects noisy measurement shots of a
quantum circuit towards the shots an ideal (noise-free) computer would produce.

For every circuit there are two sets of shots:

- **clean shots**: sampled from the ideal circuit;
- **noisy shots**: sampled from the same circuit run through a noise model of a
  real IBM device (FakeSherbrooke).

The two sets are stored in two files (`clean_*.h5`, `noisy_*.h5`) under the same
circuit key, and are **paired row by row** (row `i` of the clean array
corresponds to row `i` of the noisy array). Section 5 explains exactly what that
pairing means.

---

## 2. Important: what "paired" means (read this first)

For gate noise (T1/T2 relaxation, depolarizing errors) there is **no physical
"same shot, with noise added"**. Errors happen in the middle of the circuit, and
the measured outcome is then sampled from the altered state. Even on real
hardware you cannot run one shot with and without noise.

So the pairing in this dataset is a **construction**, not a physical fact:

1. Clean shots and noisy shots are generated as two independent Aer runs.
2. They are then matched by an **empirical maximal coupling** (`pair_shots`):
   - for every bitstring `x`, exactly `min(count_clean(x), count_noisy(x))` shots
     are paired with an identical shot (the noise did not change them);
   - the remaining noisy shots are matched with the remaining clean shots at
     random;
   - all pairs are shuffled so the order carries no information.

Properties:

- Both shot sets are only **permuted**, never altered. Their distributions are
  exactly what Aer produced.
- The fraction of unchanged shots (`frac_same`) equals the numerator of the
  Weighted Jaccard similarity used in the reference paper (Σ min / total shots).
  It is stored per circuit in `circuit_meta/frac_same`.
- For shots that noise did change, the partner is **arbitrary**. A model that
  sees only a single shot cannot learn a deterministic correction for those; it
  can learn the conditional distribution. Giving the model context (the circuit,
  or the whole set of noisy shots) makes the task meaningful.

The exact ideal distribution is also stored (`probs/`), which is the cleanest
target if you train on distributions.

---

## 3. Parameter space

### 3.1 Number of qubits N

`N ∈ {3, 5, 6, 8, 9, 11}`

### 3.2 Circuit types (5)

| Key | Circuit | Construction |
|---|---|---|
| `qft` | Quantum Fourier Transform | `QFT(num_qubits=N)` |
| `phase_estimation` | Phase Estimation | random 1-qubit unitary + (N−1) evaluation qubits |
| `hlf` | Hidden Linear Function | random connected graph on N nodes |
| `bv` | Bernstein-Vazirani | (N−1) input qubits + 1 ancilla, random secret |
| `iqp` | Instantaneous Quantum Polynomial | random connected graph on N nodes |

### 3.3 Bricks

After the base circuit, `num_bricks` random layers are appended. One brick =
one random rotation (`rx`/`ry`/`rz`, random angle in [0, 2π]) on every qubit,
followed by a chain of CX gates along a random permutation of the qubits.

`num_bricks` is sampled uniformly from the range below for every circuit:

| Circuit | N=3 | N=5 | N=6 | N=8 | N=9 | N=11 |
|---|---|---|---|---|---|---|
| qft | 1–2 | 1–2 | 1–2 | 1 | 1 | 1 |
| phase_estimation | 1–2 | 1–2 | 1–2 | 1 | 1 | 1 |
| hlf | 3–4 | 3–4 | 3 | 2–3 | 2–3 | 2 |
| bv | 1 | 1 | 1 | 1 | 1 | 1 |
| iqp | 2–3 | 2–3 | 2–3 | 2 | 2 | 2 |

### 3.4 Shots

`NUM_SHOTS = 2^11 × 45 = 92160`, the same for every N (override with
`--shots`).

---

## 4. Dataset splits

| Split | Circuits | Composition |
|---|---|---|
| train | 60,000 | per-N counts below, equal across the 5 circuit types |
| val | 300 | 10 per (N, circuit type) |
| test | 300 | 10 per (N, circuit type) |

Train circuits per circuit type, per N:

| N | per type | total for N |
|---|---|---|
| 3 | 860 | 4,300 |
| 5 | 1,430 | 7,150 |
| 6 | 1,715 | 8,575 |
| 8 | 2,285 | 11,425 |
| 9 | 2,570 | 12,850 |
| 11 | 3,140 | 15,700 |

Train, val and test use different seed bases, and every random stream is derived
through `numpy.random.SeedSequence([base, split_id, N, type_index, i, extra])`,
so no two circuits across splits, N values, or types share a seed.

---

## 5. Processing order for each circuit

For each `(split, N)` the script loops over circuit types in the fixed order
`qft, phase_estimation, hlf, bv, iqp`, and over `i = 0 … count-1` inside each
type. The running index is the circuit number (`circuit_00000`, `circuit_00001`,
…). For every circuit:

1. **Build** the circuit: base circuit of the chosen type + random bricks. All
   randomness comes from `rng_for("circuit", split, N, type_index, i)`.
2. **Logical form**: transpile to the `u`/`cx` basis (optimization level 0).
   This is the circuit stored as QASM and used for logical gate/depth counts.
3. **Exact ideal distribution** `p_clean`: `Statevector(logical).probabilities()`,
   normalized.
4. **Clean shots**: logical circuit + `measure_all()`, run on
   `AerSimulator(method="statevector")` with `shots` shots and `memory=True`
   (keeps per-shot order).
5. **Physical circuit**: the circuit with `measure_all()` is transpiled to
   FakeSherbrooke with `initial_layout` = the chosen qubit subset,
   `optimization_level=1`. The result is checked to use only qubits inside the
   subset (re-tried with a different transpiler seed up to 8 times otherwise).
6. **Noisy shots**: physical circuit run on
   `AerSimulator.from_backend(FakeSherbrooke, method="density_matrix")`,
   `memory=True`.
7. **Verification** (every 100th circuit): the physical circuit is run **without
   noise** and its empirical distribution must match `p_clean` (total variation
   < 0.2, a loose threshold that allows for sampling error). This catches
   layout / measurement-mapping bugs.
8. **Convert** both memories to integer indices (bit j = logical qubit j).
9. **Pair** the shots with `pair_shots` (Section 2) and compute `frac_same`.
10. **Pack** each bit matrix with `np.packbits(axis=1)` and **write** to HDF5
    (noisy file first, then the clean file and a metadata marker).
11. At the end of the `(split, N)` job, the per-circuit metadata is assembled
    into `circuit_meta/` arrays and the files are marked `complete=True`.

---

## 6. Noise model

- Source: `FakeSherbrooke` snapshot (127-qubit IBM Eagle device) from
  `qiskit_ibm_runtime.fake_provider`.
- Built with `AerSimulator.from_backend(...)`: thermal relaxation (T1/T2 and gate
  durations), depolarizing errors for 1- and 2-qubit gates, and readout errors.
  This is the "backend-derived simulator" twin variant of
  *Evaluating Calibration-Based Digital Twins for IBM Quantum Hardware
  Simulation* (arXiv:2603.14607), run with the `density_matrix` method.
- The same noise model is used for train, val and test.
- Note: a fake backend is a **historical calibration snapshot**. In the cited
  paper, CSV-based twins matched real hardware more often. The noise source can
  be replaced later by changing only how `sim_noisy` is created; the rest of the
  pipeline stays the same.

### Nested qubit subsets

The qubit subsets are nested: `3 ⊂ 5 ⊂ 6 ⊂ 8 ⊂ 9 ⊂ 11`. They are chosen greedily
from the device's coupling graph: start from the qubit with the best score
(readout error plus a small T1/T2 term), then repeatedly add the connected
neighbor that minimizes qubit score + average two-qubit error to already
selected qubits. Each smaller subset is chosen inside the next larger one. The
selected physical qubits are printed at startup and saved in every file
(`noise_model/subset_qubits`).

---

## 7. Files and layout

Output directory: `data_v4/` (change with `--out`).

For each `(split, N)` there are exactly two files:

```
data_v4/
├── clean_train_N3.h5   noisy_train_N3.h5
├── clean_train_N5.h5   noisy_train_N5.h5
├── ... (N = 6, 8, 9, 11)
├── clean_val_N{3,5,6,8,9,11}.h5     noisy_val_N{...}.h5
└── clean_test_N{3,5,6,8,9,11}.h5    noisy_test_N{...}.h5
```

Total: 36 files (18 pairs).

### 7.1 Contents of every file (identical in clean and noisy)

| Path | Content |
|---|---|
| `all_bitstrings` | `(2^N, N)` uint8, reference table; row i has bit j = `(i >> j) & 1` |
| `noise_model/subset_qubits` | physical qubits used |
| `noise_model/coupling_map_directed` | directed edges inside the subset (physical indices) |
| `qasm/circuit_XXXXX` | QASM2 of the logical circuit (`u`/`cx` basis) |
| `probs/circuit_XXXXX` | exact ideal distribution, `(2^N,)` float32, gzip |
| `circuit_meta/…` | per-circuit metadata arrays (below) |

`circuit_meta` arrays (length = number of circuits):

`num_bricks`, `num_gates_logical`, `num_gates_physical`, `depth_logical`,
`depth_physical`, `frac_same`, `circuit_type` (string).

### 7.2 Only in `clean_*.h5`

`clean_bitstrings/circuit_XXXXX`: `(shots, ceil(N/8))` uint8, packed.

### 7.3 Only in `noisy_*.h5`

`noisy_bitstrings/circuit_XXXXX`: `(shots, ceil(N/8))` uint8, packed.

### 7.4 File attributes

`num_qubits`, `num_circuits`, `num_shots`, `split`, `kind` ("clean"/"noisy"),
`pairing` ("empirical_max_coupling"), `seed_circuits`, `seed_shots`,
`opt_level`, `complete`, `bit_order`.

### 7.5 Bit order and packing

- Column `j` of a shot = measured logical qubit `j`.
- Probability index `i` has bit `j` = `(i >> j) & 1`.
- Shots are packed with `np.packbits(axis=1)` (column 0 in the most significant
  bit of the first byte). Unpack with
  `np.unpackbits(x, axis=1)[:, :N]`.

### 7.6 Pairing guarantee in the files

- `circuit_00042` in the clean file and in the noisy file are the same circuit
  (same QASM, same metadata).
- Row `i` of `clean_bitstrings/circuit_00042` and row `i` of
  `noisy_bitstrings/circuit_00042` are a pair in the sense of Section 2.

---

## 8. Loading a pair

```python
import h5py
import numpy as np

def load_pair(split, n, data_dir="data_v4"):
    cp = f"{data_dir}/clean_{split}_N{n}.h5"
    npth = f"{data_dir}/noisy_{split}_N{n}.h5"
    with h5py.File(cp, "r") as fc, h5py.File(npth, "r") as fn:
        assert fc.attrs["complete"] and fn.attrs["complete"]
        meta = fc["circuit_meta"]
        for cid in range(int(fc.attrs["num_circuits"])):
            key = f"circuit_{cid:05d}"
            noisy = np.unpackbits(fn[f"noisy_bitstrings/{key}"][()], axis=1)[:, :n]
            clean = np.unpackbits(fc[f"clean_bitstrings/{key}"][()], axis=1)[:, :n]
            probs = fc[f"probs/{key}"][()]
            yield dict(
                noisy=noisy, clean=clean, probs=probs,
                circuit_type=meta["circuit_type"][cid].decode(),
                num_bricks=int(meta["num_bricks"][cid]),
                frac_same=float(meta["frac_same"][cid]),
                qasm=fc[f"qasm/{key}"][()].decode(),
            )
```

---

## 9. Running the script

```bash
# quick smoke test (2 circuits per type, 4096 shots, writes to data_v4_quick/)
python generate_data_v4.py --quick

# parts of the dataset
python generate_data_v4.py --N 3 5 6 --splits val test
python generate_data_v4.py --N 11 --splits train      # separate process

# options
#   --N            list of qubit counts
#   --splits       train / val / test
#   --shots        shots per circuit (default 92160)
#   --out          output directory (default data_v4)
#   --train-scale  multiplier on train circuit counts (for trial runs)
#   --val-count / --test-count   circuits per (N, type)
```

Requirements: `qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`, `h5py`, `numpy`.
The script was tested with Qiskit 2.6.2, Aer 0.18.2, qiskit-ibm-runtime 0.52.1.
`QFT` / `PhaseEstimation` may print deprecation warnings; these are harmless.

### Resuming

Datasets are written circuit by circuit. If a run is interrupted, start the same
command again: circuits already stored (marker: their metadata entry) are
skipped and half-written ones are rewritten. A pair of files is only marked
`complete=True` after its metadata arrays are assembled; finished pairs are
skipped entirely on later runs.

### Parallel runs

Run separate processes for different `(N, split)` combinations (for example one
per N), ideally with `OMP_NUM_THREADS=1` each. Do not run two processes on the
same `(split, N)`.

---

## 10. Seeds

| Purpose | Base |
|---|---|
| Circuit generation | train 42, val 777, test 9999 |
| Shot sampling / pairing | train 123, val 778, test 10000 |

Each circuit's streams are derived via `SeedSequence([base, split_id, N,
type_index, i, extra])`. The Aer shot seed of the noisy run is the clean seed
+ 1. Transpiler seeds are derived the same way (and incremented on retry).
Bases are stored as file attributes. The Aer noise model itself is deterministic
(calibration snapshot), so there is no separate noise seed.

---

## 11. Size and run-time estimates

These are estimates, not measurements on your machine.

**Storage** (uncompressed shot arrays; N ≤ 8 use 1 byte/shot, N = 9 and 11 use
2 bytes/shot):

| N | circuits | one file (shots) |
|---|---|---|
| 3 | 4,300 | ~0.4 GB |
| 5 | 7,150 | ~0.7 GB |
| 6 | 8,575 | ~0.8 GB |
| 8 | 11,425 | ~1.1 GB |
| 9 | 12,850 | ~2.4 GB |
| 11 | 15,700 | ~2.9 GB |

Train, clean + noisy: roughly 16–17 GB total, plus `probs` and QASM (small,
`probs` is gzip-compressed). Val and test are small. Reduce with `--shots`.

**Time**: in a sandbox smoke test (4096 shots), N = 8 and N = 11 together took
about 105 s for 20 circuits, i.e. a few seconds per circuit. N = 11 is the
slowest (density-matrix simulation scales about 4× per extra qubit). A rough
extrapolation for N = 11 train (15,700 circuits) is on the order of a day or
more on one core. Time the `--quick` run on your machine and extrapolate; split
N = 11 across processes or reduce its circuit count if needed. The number of
shots has little influence on run time because the density matrix is evolved
once per circuit and shots are only sampled.

---

## 12. Smoke-test observations (sandbox, 4096 shots, 2 circuits per type)

- N = 3: `frac_same` ≈ 0.90–0.97; N = 5: ≈ 0.77–0.87 (fraction of shots that
  noise left unchanged). Larger N will generally have lower values.
- Empirical clean distribution vs. exact `probs`: total variation ≈ 0.006 (N = 3)
  and ≈ 0.026 (N = 5), consistent with sampling error.
- Physical depths were roughly 3–4× the logical depths after transpilation to
  the device basis.

The full dataset (60,000 train circuits, 92160 shots) has not been generated in
the test environment.

---

## 13. Known caveats

1. **Pairing is a coupling, not physics** (Section 2). For changed shots the
   partner is arbitrary.
2. **Fake backend snapshot**: not the live calibration of the real machine.
   Validate the twin against real QPU runs (for example with Weighted Jaccard,
   as in the reference paper) before relying on the noise realism.
3. **Reference paper scope**: it validated five-qubit random circuits; agreement
   for N = 8–11 and for QFT/PE/HLF/BV/IQP circuits is not established there.
4. **Transpilation affects noise**: results depend on `OPT_LEVEL` (default 1;
   the paper found levels 0–1 matched hardware best). It is stored as a file
   attribute.
5. **Routing**: transpilation is restricted to the chosen subset; if a circuit
   cannot be routed inside it within 8 attempts, the run stops with an error.
6. **Memory**: density-matrix simulation for N = 11 needs a few hundred MB per
   process; check this when running many processes in parallel.
7. **`circuit_type` and ordering**: types are stored explicitly in
   `circuit_meta/circuit_type`; the type blocks are ordered `qft,
   phase_estimation, hlf, bv, iqp` within each file.

---

## 14. Summary

| Aspect | Value |
|---|---|
| N values | 3, 5, 6, 8, 9, 11 |
| Circuit types | QFT, PhaseEstimation, HLF, Bernstein-Vazirani, IQP |
| Shots per circuit | 92,160 (configurable) |
| Train / val / test circuits | 60,000 / 300 / 300 |
| Files | 36 (18 clean/noisy pairs) in `data_v4/` |
| Noise | FakeSherbrooke via `AerSimulator.from_backend`, density matrix |
| Clean shots | ideal Aer (statevector), `memory=True` |
| Noisy shots | noisy Aer on the transpiled circuit, `memory=True` |
| Pairing | empirical maximal coupling (`frac_same` stored per circuit) |
| Exact targets | `probs/` (ideal distribution) |
| Transpilation | `optimization_level=1`, restricted to the nested qubit subset |
| Resumable | yes (per-circuit) |