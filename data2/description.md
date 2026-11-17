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

`N ∈ {3, 5, 6, 8, 9, 11}`, and for evaluation of larger circuits `N ∈ {12, 13}`
(`LARGER_NUM_QUBITS`), with the brick ranges of N = 11.

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

The subsets of `LARGER_NUM_QUBITS` grow upward from the N = 11 subset with the same
greedy rule (`11 ⊂ 12 ⊂ 13`), so adding them does not change the subsets of the
smaller N or the files generated for them.

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
| `noise_model/readout_p01`, `readout_p10` | `(N,)` P(read 1 \| prepared 0) and P(read 0 \| prepared 1) of every subset qubit |
| `noise_model/t1`, `t2` | `(N,)` relaxation and dephasing times in seconds |
| `noise_model/error_1q` | `(N,)` error of one `sx` gate |
| `noise_model/error_2q` | `(N, N)` error of the two-qubit gate between two subset qubits, 0 where they are not connected |
| `qasm/circuit_XXXXX` | QASM2 of the logical circuit (`u`/`cx` basis) |
| `probs/circuit_XXXXX` | exact ideal distribution, `(2^N,)` float32, gzip |
| `circuit_meta/…` | per-circuit metadata arrays (below) |

`circuit_meta` arrays (length = number of circuits):

`num_bricks`, `num_gates_logical`, `num_gates_physical`, `depth_logical`,
`depth_physical`, `frac_same`, `circuit_type` (string).

Gate counts of the physical (transpiled) circuit, indexed by the position of a
qubit in `noise_model/subset_qubits`, the same index as the calibration arrays:

- `gates_1q_per_qubit`: `(circuits, N)`, one-qubit gates on every qubit. `rz` is
  not counted: it is virtual on the device and has no error.
- `gates_2q_per_pair`: `(circuits, N, N)`, symmetric, two-qubit gates between
  every pair of qubits.
- `measured_qubit`: `(circuits, N)`, entry `j` is the qubit measured into bit `j`
  of a shot. Routing can move a logical qubit, so this is not always `j`.

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
#   --types        circuit types to generate (default: all five)
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

### Generated version

The files used by the project were generated with 40 validation and 40 test
circuits per (N, type) and 500 training circuits per (N, type):

```bash
python data2/generation.py --N 3 --splits val test train --val-count 40 --test-count 40 --train-scale 0.5813953488372093
python data2/generation.py --N 5 --splits val test train --val-count 40 --test-count 40 --train-scale 0.34965034965034963
python data2/generation.py --N 6 --splits val test train --val-count 40 --test-count 40 --train-scale 0.2915451895043732
python data2/generation.py --N 8 --splits val test train --val-count 40 --test-count 40 --train-scale 0.2188183807439825 --types qft hlf bv iqp
python data2/generation.py --N 9 --splits val test train --val-count 40 --test-count 40 --train-scale 0.19455252918287938 --types qft hlf bv iqp
python data2/generation.py --N 11 --splits val test train --val-count 40 --test-count 40 --train-scale 0.1592356687898089 --types qft hlf bv iqp
```

Phase estimation is left out for N >= 8. Its transpiled circuit has 3,000 to
17,000 gates there, and the noisy distribution is almost unrelated to the ideal
one (L1 distance 1.4 to 1.8 out of 2), so there is nothing to mitigate.

On Windows set `PYTHONIOENCODING=utf-8` when the output is redirected to a file.

Measured time per circuit: about 1 s (N = 3, 5), 2.5 s (N = 6, 8), 5 s (N = 9) on
one thread, and 7 s for N = 11 with `OMP_NUM_THREADS=8` (25 s on one thread).

The larger circuits have small training sets, used only to fit the baseline
parameters, and are meant for evaluating models trained on N ≤ 11:

```bash
python data2/generation.py --N 12 --splits val train test --val-count 20 --test-count 20 --train-scale 0.0127389 --types qft hlf bv iqp
python data2/generation.py --N 13 --splits val train test --val-count 10 --test-count 10 --train-scale 0.00477707 --types qft hlf bv iqp
```

That is 40 train, 20 val and 20 test circuits per type for N = 12 and 15, 10 and 10
for N = 13. With `OMP_NUM_THREADS=10` a circuit takes about 8 s (N = 12) and 30 to
60 s (N = 13). N = 14 needs a 4.3 GB density matrix per simulation and was not
generated.

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
5. **Routing**: the transpiler gets only the couplings inside the chosen subset,
   so routing cannot leave it. The check that all used qubits are in the subset
   stays as a safeguard.
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

---

## 15. Clifford circuits on up to 63 qubits (`clifford_generation.py`)

The density matrix needs 4^N numbers and stops at 13 qubits. For larger N,
`clifford_generation.py` samples Clifford circuits under Pauli noise with Pauli
frames: one noiseless reference outcome, and for every shot a Pauli operator that
is pushed through the Clifford gates as two bit vectors (X and Z part). The X part
at a measurement flips the reference bit. All shots are processed at once in NumPy.

Noise: the gate errors of `AerSimulator.from_backend(FakeSherbrooke)` (depolarising
and thermal relaxation) are replaced by their Pauli twirl, the Pauli channel on the
diagonal of the chi matrix; readout errors are kept. Checks:

| Check (13 qubits, 200,000 shots unless noted) | Result |
|---|---|
| Frames vs Aer stabilizer, same twirled noise | TV 0.010 to 0.024, the same as between two runs of the sampler (0.009 to 0.027); P(ideal support) within 0.002 |
| Time per circuit | frames 0.8 to 2.6 s, Aer 103 to 281 s |
| Twirled vs full noise (8 qubits, 92,160 shots, density matrix) | TV 0.006 to 0.014; P(ideal support) within 0.01 |

Circuits (no bricks, so the ideal output stays small):

| Type | Circuit | Ideal support |
|---|---|---|
| `bv` | Bernstein-Vazirani as in `generation.py` | 2 bitstrings (the last qubit is random) |
| `hidden_shift` | hidden shift of the bent function sum x_2i x_2i+1 | 1 bitstring |
| `ghz` | GHZ grown from a random qubit along a breadth-first tree of the couplings | 2 bitstrings |

Hidden linear function circuits were left out: without bricks their output is
uniform over an affine subspace of dimension N-2 to N-10, which no correction can
change at these sizes.

Sizes and splits: N = 11, 13, 16, 20, 24, 32, 40, 48, 56, 63; qubit subsets of
`generation.py` up to 13, grown greedily beyond; `val` and `test` with 20 circuits
per type and 92,160 shots; seeds differ from those of `generation.py`. Run time:
about 25 minutes for both splits (two processes), 221 MB.

Files `data_clifford/clifford_{split}_N{N}.h5`:

| Path | Content |
|---|---|
| `noisy_bitstrings/circuit_XXXXX` | distinct noisy outcomes, int64, bit j = qubit j, sorted |
| `noisy_counts/circuit_XXXXX` | their counts |
| `target_bitstrings/circuit_XXXXX`, `target_probs/circuit_XXXXX` | support of P_target and its probabilities |
| `qasm/circuit_XXXXX` | logical circuit |
| `circuit_meta/`, `noise_model/` | as in the v4 files (gate counts per qubit and pair, measured qubit, calibration) |

```bash
cd data2
python clifford_generation.py --check                       # compare with Aer on 13 qubits
python clifford_generation.py --splits val test --out ../data_clifford
```

Noisy probability of the ideal support (one validation circuit): 13 qubits 0.63
(bv) to 0.78 (ghz); 32 qubits 0.21 (bv), 0.31 (hidden shift), 0.47 (ghz); 63 qubits
0.018 (bv), 0.064 (hidden shift), 0.18 (ghz).
