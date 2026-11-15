"""
generate_data_v4.py — датасет для денойзингу шотів (FakeSherbrooke).

Що робить:
  * шум: AerSimulator.from_backend(FakeSherbrooke, method="density_matrix")
    (T1/T2 + деполяризація + readout зі знімка калібрування) — "backend-derived
    twin" зі статті arXiv:2603.14607;
  * чисті шоти: ідеальний Aer (statevector), memory=True;
  * зашумлені шоти: та сама схема, транспільована на підмножину кубітів
    пристрою, Aer з шумом, memory=True;
  * парування: емпіричне максимальне зчеплення (pair_shots):
      - для кожного бітстрінга x рівно min(c_clean(x), c_noisy(x)) шотів
        стоять в парі самі з собою (шум їх не змінив);
      - решта зашумлених шотів зіставляється з рештою чистих;
      - пари перемішуються. Розподіли обох наборів лишаються РІВНО такими,
        як видав Aer (набори лише переставляються).

ВАЖЛИВО: фізично "той самий шот + шум" для гейтового шуму не існує.
Зчеплення — це конструкція, яка зберігає обидва розподіли і максимізує
кількість незмінених шотів. Частка незмінених шотів зберігається в
circuit_meta/frac_same (дорівнює чисельнику Weighted Jaccard).

Порядок бітів скрізь: колонка j шоту = логічний кубіт j;
індекс розподілу i має біт j = (i >> j) & 1.
Упаковка: np.packbits(axis=1); розпаковка: np.unpackbits(x, axis=1)[:, :N].

Запуск:
  python generate_data_v4.py --N 3 5 --splits val test      # частково
  python generate_data_v4.py --quick                        # швидка проба
  python generate_data_v4.py --N 11 --splits train          # окремий процес
Скрипт відновлюється після збою (пропускає вже записані схеми).
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import IQP, QFT, HiddenLinearFunction, PhaseEstimation
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# Конфігурація
# ----------------------------------------------------------------------
NUM_QUBITS_LIST = [3, 5, 6, 8, 9, 11]
CIRCUIT_TYPES = ["qft", "phase_estimation", "hlf", "bv", "iqp"]
NUM_SHOTS = (2**11) * 45  # 92160

TRAIN_COUNTS_PER_N = {3: 860, 5: 1430, 6: 1715, 8: 2285, 9: 2570, 11: 3140}
VAL_COUNT_PER_COMBO = 10
TEST_COUNT_PER_COMBO = 10

SEEDS = {
    "train": {"circuits": 42, "shots": 123},
    "val": {"circuits": 777, "shots": 778},
    "test": {"circuits": 9999, "shots": 10000},
}
SPLIT_ID = {"train": 0, "val": 1, "test": 2}

OPT_LEVEL = 1  # у статті найкращі збіги були на рівнях 0 і 1
MAX_TRANSPILE_TRIES = 8
VERIFY_EVERY = 100  # перевірка транспільованої схеми проти логічної
VERIFY_SHOTS = 50000
VERIFY_TV_MAX = 0.2  # поріг (враховує статистичний шум семплювання)
OUTPUT_DIR = "data_v4"

BRICK_RANGES = {
    "qft": {3: (1, 2), 5: (1, 2), 6: (1, 2), 8: (1, 1), 9: (1, 1), 11: (1, 1)},
    "phase_estimation": {3: (1, 2), 5: (1, 2), 6: (1, 2), 8: (1, 1), 9: (1, 1), 11: (1, 1)},
    "hlf": {3: (3, 4), 5: (3, 4), 6: (3, 3), 8: (2, 3), 9: (2, 3), 11: (2, 2)},
    "bv": {3: (1, 1), 5: (1, 1), 6: (1, 1), 8: (1, 1), 9: (1, 1), 11: (1, 1)},
    "iqp": {3: (2, 3), 5: (2, 3), 6: (2, 3), 8: (2, 2), 9: (2, 2), 11: (2, 2)},
}


# ----------------------------------------------------------------------
# Сіди (SeedSequence -> жодних перетинів між split/N/типом/номером)
# ----------------------------------------------------------------------
def _ss(kind, split, n, ct_idx, i, extra=0):
    base = SEEDS[split]["circuits" if kind == "circuit" else "shots"]
    return np.random.SeedSequence([base, SPLIT_ID[split], n, ct_idx, i, extra])


def rng_for(kind, split, n, ct_idx, i):
    return np.random.default_rng(_ss(kind, split, n, ct_idx, i))


def int_seed(kind, split, n, ct_idx, i, extra=7):
    return int(_ss(kind, split, n, ct_idx, i, extra).generate_state(1)[0] % (2**31 - 1))


# ----------------------------------------------------------------------
# Допоміжне
# ----------------------------------------------------------------------
def all_bitstrings_table(n):
    idx = np.arange(2**n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.uint8)


def idx_to_bits(idx, n):
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.uint8)


def mem_to_idx(mem):
    """get_memory() -> цілі індекси (біт j = логічний кубіт j)."""
    return np.fromiter((int(s.replace(" ", ""), 2) for s in mem), dtype=np.int64, count=len(mem))


# ----------------------------------------------------------------------
# Підмножини кубітів Sherbrooke (вкладені 3 ⊂ 5 ⊂ 6 ⊂ 8 ⊂ 9 ⊂ 11)
# ----------------------------------------------------------------------
def _device_info(backend):
    target = backend.target
    g2 = None
    for name in target.operation_names:
        op = target.operation_from_name(name)
        if getattr(op, "num_qubits", 0) == 2:
            g2 = name
            break
    if g2 is None:
        raise RuntimeError("у target немає двокубітного гейта")

    directed = [tuple(e) for e in target[g2].keys()]
    nbrs = {}
    for a, b in directed:
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)

    def qscore(q):
        ro = 1.0
        try:
            e = target["measure"][(q,)].error
            ro = 1.0 if e is None else e
        except Exception:
            pass
        s = ro
        try:
            qp = target.qubit_properties[q]
            if qp.t1:
                s += 0.01 * (1e-4 / qp.t1)
            if qp.t2:
                s += 0.01 * (1e-4 / qp.t2)
        except Exception:
            pass
        return s

    def escore(a, b):
        for k in ((a, b), (b, a)):
            if k in target[g2]:
                e = target[g2][k].error
                return 1.0 if e is None else e
        return 1.0

    return directed, nbrs, qscore, escore, g2


def _greedy_subset(info, k, allowed=None):
    _, nbrs, qscore, escore, _ = info
    cand = set(allowed) if allowed is not None else set(nbrs)
    start = min(cand, key=qscore)
    sel = [start]
    frontier = (nbrs[start] & cand) - set(sel)
    while len(sel) < k and frontier:
        best, best_s = None, float("inf")
        for q in sorted(frontier):
            ns = [s for s in sel if s in nbrs[q]]
            s = qscore(q) + float(np.mean([escore(q, x) for x in ns]))
            if s < best_s:
                best, best_s = q, s
        sel.append(best)
        frontier |= nbrs[best] & cand
        frontier -= set(sel)
    if len(sel) != k:
        raise RuntimeError(f"не вдалося вибрати {k} зв'язних кубітів")
    return sel


def build_nested_subsets(backend):
    info = _device_info(backend)
    ordered = sorted(NUM_QUBITS_LIST, reverse=True)
    subsets = {ordered[0]: _greedy_subset(info, ordered[0])}
    for prev, nxt in zip(ordered, ordered[1:]):
        subsets[nxt] = _greedy_subset(info, nxt, allowed=subsets[prev])
    return subsets, info[0], info[4]


def calibration_of_subset(backend, subset, g2):
    """Калібрування вибраних кубітів зі знімка пристрою (індекс = позиція в subset)."""
    target, props = backend.target, backend.properties()
    k = len(subset)
    cal = {
        # P(виміряли 1 | підготували 0) та P(виміряли 0 | підготували 1)
        "readout_p01": np.array([props.qubit_property(q, "prob_meas1_prep0")[0] for q in subset]),
        "readout_p10": np.array([props.qubit_property(q, "prob_meas0_prep1")[0] for q in subset]),
        "t1": np.array([target.qubit_properties[q].t1 for q in subset]),
        "t2": np.array([target.qubit_properties[q].t2 for q in subset]),
        "error_1q": np.array([target["sx"][(q,)].error for q in subset]),
    }
    # помилка двокубітного гейта між позиціями a, b; 0 там, де з'єднання немає
    e2 = np.zeros((k, k))
    for a, qa in enumerate(subset):
        for b, qb in enumerate(subset):
            for key in ((qa, qb), (qb, qa)):
                if key in target[g2]:
                    e2[a, b] = target[g2][key].error
    cal["error_2q"] = e2
    return cal


def physical_gate_counts(qc_t, subset, n):
    """Гейти транспільованої схеми на кожному фізичному кубіті (за позицією в subset).

    rz на пристрої віртуальний і безпомилковий, тому не рахується.
    """
    pos = {q: i for i, q in enumerate(subset)}
    one = np.zeros(n, dtype=np.int32)
    two = np.zeros((n, n), dtype=np.int32)
    measured = np.full(n, -1, dtype=np.int32)  # measured[j] = позиція кубіта, виміряного в біт j
    for inst in qc_t.data:
        name = inst.operation.name
        qs = [pos[qc_t.find_bit(q).index] for q in inst.qubits]
        if name == "measure":
            measured[qc_t.find_bit(inst.clbits[0]).index] = qs[0]
        elif name in ("rz", "barrier", "delay"):
            continue
        elif len(qs) == 1:
            one[qs[0]] += 1
        elif len(qs) == 2:
            two[qs[0], qs[1]] += 1
            two[qs[1], qs[0]] += 1
    return one, two, measured


# ----------------------------------------------------------------------
# Схеми
# ----------------------------------------------------------------------
def _random_unitary_1q(rng):
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


def _random_adjacency(n, rng):
    adj = np.zeros((n, n), dtype=int)
    order = rng.permutation(n)
    for i in range(1, n):
        a, b = order[i], order[rng.integers(0, i)]
        adj[a, b] = adj[b, a] = 1
    for _ in range(max(1, n // 3)):
        a, b = rng.choice(n, size=2, replace=False)
        adj[a, b] = adj[b, a] = 1
    return adj


def _make_qft(n, rng):
    return QFT(num_qubits=n, do_swaps=True, inverse=False)


def _make_pe(n, rng):
    u = QuantumCircuit(1)
    u.unitary(_random_unitary_1q(rng), 0)
    return PhaseEstimation(num_evaluation_qubits=n - 1, unitary=u)


def _make_hlf(n, rng):
    return HiddenLinearFunction(adjacency_matrix=_random_adjacency(n, rng))


def _make_bv(n, rng):
    secret = rng.integers(0, 2, size=n - 1)
    qc = QuantumCircuit(n)
    for q in range(n - 1):
        qc.h(q)
    qc.x(n - 1)
    qc.h(n - 1)
    for q in range(n - 1):
        if secret[q]:
            qc.cx(q, n - 1)
    for q in range(n - 1):
        qc.h(q)
    return qc


def _make_iqp(n, rng):
    return IQP(interactions=_random_adjacency(n, rng))


BUILDERS = {
    "qft": _make_qft,
    "phase_estimation": _make_pe,
    "hlf": _make_hlf,
    "bv": _make_bv,
    "iqp": _make_iqp,
}


def _random_brick(qc, n, rng):
    for q in range(n):
        g = str(rng.choice(["rx", "ry", "rz"]))
        getattr(qc, g)(float(rng.uniform(0, 2 * np.pi)), q)
    order = rng.permutation(n)
    for i in range(n - 1):
        qc.cx(int(order[i]), int(order[i + 1]))


def build_circuit(ct, n, rng):
    lo, hi = BRICK_RANGES[ct][n]
    nb = int(rng.integers(lo, hi + 1))
    base = BUILDERS[ct](n, rng)
    qc = QuantumCircuit(n)
    qc.compose(base, inplace=True)
    for _ in range(nb):
        _random_brick(qc, n, rng)
    return qc, nb


# ----------------------------------------------------------------------
# Транспіляція та парування
# ----------------------------------------------------------------------
def transpile_in_subset(qm, backend, subset, seed0):
    """Транспіляція на пристрій; роутинг не має виходити за межі підмножини."""
    allowed = set(subset)
    # з'єднання лише всередині підмножини: роутер не може вивести схему за її межі
    inside = CouplingMap(
        [list(e) for e in backend.coupling_map.get_edges() if e[0] in allowed and e[1] in allowed]
    )
    for t in range(MAX_TRANSPILE_TRIES):
        qc_t = transpile(
            qm,
            backend=backend,
            coupling_map=inside,
            initial_layout=subset,
            optimization_level=OPT_LEVEL,
            seed_transpiler=seed0 + t,
        )
        used = {qc_t.find_bit(q).index for inst in qc_t.data for q in inst.qubits}
        if used <= allowed:
            return qc_t
    raise RuntimeError("роутинг виходить за межі підмножини кубітів")


def pair_shots(clean_idx, noisy_idx, n, rng):
    """Емпіричне максимальне зчеплення двох незалежних наборів шотів."""
    size = 2**n
    cc = np.bincount(clean_idx, minlength=size)
    nc = np.bincount(noisy_idx, minlength=size)
    keep = np.minimum(cc, nc)
    v = np.arange(size)

    same = np.repeat(v, keep)  # шум нічого не змінив
    ex_c = np.repeat(v, cc - keep)  # надлишок чистих
    ex_n = np.repeat(v, nc - keep)  # надлишок зашумлених
    rng.shuffle(ex_n)  # випадкове зіставлення надлишків

    clean_p = np.concatenate([same, ex_c])
    noisy_p = np.concatenate([same, ex_n])
    perm = rng.permutation(len(clean_p))  # прибрати сортування за значенням
    return clean_p[perm], noisy_p[perm], len(same) / len(clean_p)


def process_circuit(
    qc, n, subset, backend, sim_clean, sim_noisy, shots, seed_sim, seed_tr, rng, verify
):
    # логічна схема в базисі u/cx (для QASM і точного розподілу)
    qc_log = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
    p_clean = Statevector(qc_log).probabilities()  # біт j = кубіт j
    p_clean = np.asarray(p_clean, dtype=np.float64)
    p_clean /= p_clean.sum()

    qm = qc.copy()
    qm.measure_all()

    # чисті шоти (ідеальний Aer) — на розкладеній схемі (Aer не знає QFT/PE як інструкцій)
    qm_clean = qc_log.copy()
    qm_clean.measure_all()
    clean_mem = (
        sim_clean.run(qm_clean, shots=shots, memory=True, seed_simulator=seed_sim)
        .result()
        .get_memory()
    )

    # зашумлені шоти (твін Sherbrooke)
    qc_t = transpile_in_subset(qm, backend, subset, seed_tr)
    noisy_mem = (
        sim_noisy.run(qc_t, shots=shots, memory=True, seed_simulator=seed_sim + 1)
        .result()
        .get_memory()
    )

    if verify:  # транспільована схема без шуму має давати той самий розподіл
        cnt = (
            sim_clean.run(qc_t, shots=VERIFY_SHOTS, seed_simulator=seed_sim + 2)
            .result()
            .get_counts()
        )
        emp = np.zeros(2**n)
        for s, c in cnt.items():
            emp[int(s.replace(" ", ""), 2)] += c
        emp /= emp.sum()
        tv = 0.5 * np.abs(emp - p_clean).sum()
        assert tv < VERIFY_TV_MAX, f"транспільована схема не збігається з логічною (TV={tv:.3f})"

    c_idx, n_idx = mem_to_idx(clean_mem), mem_to_idx(noisy_mem)
    c_p, n_p, frac_same = pair_shots(c_idx, n_idx, n, rng)
    return (idx_to_bits(c_p, n), idx_to_bits(n_p, n), p_clean, frac_same, qc_log, qc_t)


# ----------------------------------------------------------------------
# Запис (split, N) -> clean_*.h5 + noisy_*.h5 (з відновленням після збою)
# ----------------------------------------------------------------------
def _init_file(f, kind, split, n, total, shots, subset, directed, calibration):
    f.attrs.update(
        num_qubits=n,
        num_circuits=total,
        num_shots=shots,
        split=split,
        kind=kind,
        pairing="empirical_max_coupling",
        seed_circuits=SEEDS[split]["circuits"],
        seed_shots=SEEDS[split]["shots"],
        opt_level=OPT_LEVEL,
        complete=False,
        bit_order="column j = qubit j; index bit j = (i>>j)&1",
    )
    f.create_dataset("all_bitstrings", data=all_bitstrings_table(n))
    ng = f.create_group("noise_model")
    ng.attrs["source"] = "FakeSherbrooke via AerSimulator.from_backend (density_matrix)"
    ng.create_dataset("subset_qubits", data=np.array(subset, dtype=np.int32))
    sub = set(subset)
    ng.create_dataset(
        "coupling_map_directed",
        data=np.array([[a, b] for a, b in directed if a in sub and b in sub], dtype=np.int32),
    )
    for name, values in calibration.items():
        ng.create_dataset(name, data=values.astype(np.float64))


def write_split_for_n(
    split, n, counts_per_type, shots, backend, subset, directed, calibration, sim_clean, sim_noisy
):
    cpath = os.path.join(OUTPUT_DIR, f"clean_{split}_N{n}.h5")
    npath = os.path.join(OUTPUT_DIR, f"noisy_{split}_N{n}.h5")
    total = sum(counts_per_type.values())

    # вже завершено?
    if os.path.exists(cpath) and os.path.exists(npath):
        try:
            with h5py.File(cpath, "r") as a, h5py.File(npath, "r") as b:
                if bool(a.attrs.get("complete")) and bool(b.attrs.get("complete")):
                    print(f"  skip {split} N={n} (готово)")
                    return
        except OSError:
            pass

    fresh = not (os.path.exists(cpath) and os.path.exists(npath))
    mode = "w" if fresh else "a"
    with h5py.File(cpath, mode) as fc, h5py.File(npath, mode) as fn:
        if fresh:
            _init_file(fc, "clean", split, n, total, shots, subset, directed, calibration)
            _init_file(fn, "noisy", split, n, total, shots, subset, directed, calibration)
            for g in ("probs", "clean_bitstrings", "qasm", "meta_json"):
                fc.create_group(g)
            for g in ("probs", "noisy_bitstrings", "qasm"):
                fn.create_group(g)
            fc["clean_bitstrings"].attrs.update(packed=True, bits_per_shot=n)
            fn["noisy_bitstrings"].attrs.update(packed=True, bits_per_shot=n)

        done = set(fc["meta_json"].keys()) if "meta_json" in fc else set()
        t0 = time.perf_counter()
        cid = 0
        for ct_idx, ct in enumerate(CIRCUIT_TYPES):
            for i in range(counts_per_type.get(ct, 0)):
                key = f"circuit_{cid:05d}"
                cid_now, cid = cid, cid + 1
                if key in done:
                    continue
                # очистити можливі напівзаписані дані попереднього збою
                for grp, f in (
                    ("probs", fc),
                    ("clean_bitstrings", fc),
                    ("qasm", fc),
                    ("probs", fn),
                    ("noisy_bitstrings", fn),
                    ("qasm", fn),
                ):
                    if key in f[grp]:
                        del f[grp][key]

                qc, nb = build_circuit(ct, n, rng_for("circuit", split, n, ct_idx, i))
                sh_c, sh_n, p_c, frac_same, qc_log, qc_t = process_circuit(
                    qc,
                    n,
                    subset,
                    backend,
                    sim_clean,
                    sim_noisy,
                    shots,
                    int_seed("shots", split, n, ct_idx, i),
                    int_seed("circuit", split, n, ct_idx, i),
                    rng_for("shots", split, n, ct_idx, i),
                    verify=(cid_now % VERIFY_EVERY == 0),
                )

                q = qasm2.dumps(qc_log)
                one_q, two_q, measured = physical_gate_counts(qc_t, subset, n)
                # спочатку noisy, clean + meta_json — наприкінці (маркер завершення)
                fn["noisy_bitstrings"].create_dataset(key, data=np.packbits(sh_n, axis=1))
                fn["probs"].create_dataset(key, data=p_c.astype(np.float32), compression="gzip")
                fn["qasm"].create_dataset(key, data=np.bytes_(q))
                fc["clean_bitstrings"].create_dataset(key, data=np.packbits(sh_c, axis=1))
                fc["probs"].create_dataset(key, data=p_c.astype(np.float32), compression="gzip")
                fc["qasm"].create_dataset(key, data=np.bytes_(q))
                meta = dict(
                    num_bricks=nb,
                    circuit_type=ct,
                    num_gates_logical=int(qc_log.size()),
                    num_gates_physical=int(qc_t.size()),
                    depth_logical=int(qc_log.depth()),
                    depth_physical=int(qc_t.depth()),
                    frac_same=float(frac_same),
                    gates_1q_per_qubit=one_q.tolist(),
                    gates_2q_per_pair=two_q.tolist(),
                    measured_qubit=measured.tolist(),
                )
                fc["meta_json"].create_dataset(key, data=np.bytes_(json.dumps(meta)))

                if (cid_now + 1) % 50 == 0:
                    fc.flush()
                    fn.flush()
                    el = time.perf_counter() - t0
                    print(
                        f"  [{split} N={n}] {cid_now + 1}/{total}  "
                        f"({el:.0f}s, frac_same={frac_same:.3f})",
                        flush=True,
                    )

        # фіналізація метаданих
        keys = [f"circuit_{c:05d}" for c in range(total)]
        metas = [json.loads(bytes(fc["meta_json"][k][()]).decode()) for k in keys]
        for f in (fc, fn):
            if "circuit_meta" in f:
                del f["circuit_meta"]
            m = f.create_group("circuit_meta")
            for name in [
                "num_bricks",
                "num_gates_logical",
                "num_gates_physical",
                "depth_logical",
                "depth_physical",
            ]:
                m.create_dataset(name, data=np.array([d[name] for d in metas], dtype=np.int32))
            for name in ["gates_1q_per_qubit", "gates_2q_per_pair", "measured_qubit"]:
                m.create_dataset(name, data=np.array([d[name] for d in metas], dtype=np.int32))
            m.create_dataset(
                "frac_same", data=np.array([d["frac_same"] for d in metas], dtype=np.float32)
            )
            m.create_dataset(
                "circuit_type",
                data=np.array([d["circuit_type"] for d in metas], dtype=object),
                dtype=h5py.string_dtype(),
            )
        del fc["meta_json"]
        fc.attrs["complete"] = True
        fn.attrs["complete"] = True


# ----------------------------------------------------------------------
def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="*", default=NUM_QUBITS_LIST)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument(
        "--types",
        nargs="*",
        default=CIRCUIT_TYPES,
        choices=CIRCUIT_TYPES,
        help="типи схем (сіди інших типів не змінюються)",
    )
    ap.add_argument("--shots", type=int, default=NUM_SHOTS)
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument(
        "--train-scale", type=float, default=1.0, help="множник на кількість train-схем (для проб)"
    )
    ap.add_argument("--val-count", type=int, default=VAL_COUNT_PER_COMBO)
    ap.add_argument("--test-count", type=int, default=TEST_COUNT_PER_COMBO)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="швидка проба: 2 схеми на тип, 4096 шотів, окремий каталог",
    )
    args = ap.parse_args()

    OUTPUT_DIR = args.out
    if args.quick:
        OUTPUT_DIR = "data_v4_quick"
        args.shots = 4096
        args.val_count = args.test_count = 2
        args.train_scale = 0.0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    backend = FakeSherbrooke()
    subsets, directed, g2 = build_nested_subsets(backend)
    for n in sorted(subsets):
        print(f"N={n}: {subsets[n]}")

    sim_noisy = AerSimulator.from_backend(backend, method="density_matrix")
    sim_clean = AerSimulator(method="statevector")

    for split in args.splits:
        for n in args.N:
            if split == "train":
                c = TRAIN_COUNTS_PER_N[n]
                c = (
                    max(2, int(round(c * args.train_scale)))
                    if args.quick
                    else int(round(c * args.train_scale))
                )
            else:
                c = args.val_count if split == "val" else args.test_count
            if c <= 0:
                continue
            cpt = {ct: c for ct in args.types}
            print(f"== {split} N={n}: {c} схем/тип, {args.shots} шотів ==", flush=True)
            write_split_for_n(
                split,
                n,
                cpt,
                args.shots,
                backend,
                subsets[n],
                directed,
                calibration_of_subset(backend, subsets[n], g2),
                sim_clean,
                sim_noisy,
            )
    print("Готово:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
