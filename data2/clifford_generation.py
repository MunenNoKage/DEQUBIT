"""Clifford circuits on up to 112 qubits of FakeSherbrooke with Pauli-twirled device noise.

The density matrix of generation.py needs 4**n numbers and stops at 13 qubits. Clifford
circuits under Pauli noise can be sampled exactly for any size by Pauli frames: the noise
of every shot is a Pauli operator pushed through the Clifford gates as two bit vectors,
and the outcome is a noiseless reference outcome flipped by the X part of that operator.
The device noise of every gate (depolarising and thermal relaxation, as in
AerSimulator.from_backend) is replaced by its Pauli twirl, the Pauli channel with the
same Pauli error probabilities; readout errors are kept as they are.

Qubit subsets, transpilation, calibration and gate counts come from generation.py, so the
qubits of N <= 13 are the qubits of the v4 dataset. Families have a small ideal support,
so the L1 distance to P_target stays exact without 2**n vectors:
  bv            Bernstein-Vazirani as in generation.py without bricks, two bitstrings
                (the last qubit is the ancilla, in |->, and its bit is random);
  hidden_shift  hidden shift of the bent function sum x_2i x_2i+1, one bitstring (two
                for odd n: the qubit outside every pair ends in |+>);
  ghz           GHZ state grown along the couplings of the subset, two bitstrings.

Output per (split, N): clifford_{split}_N{N}.h5 with, per circuit, the distinct noisy
bitstrings and their counts, the support of P_target with its probabilities, the qasm of
the logical circuit, and the circuit_meta and noise_model groups of v4. A bitstring is a
row of 63-bit int64 words: bit j = qubit j is bit j % 63 of word j // 63, so up to 63
qubits it is the integer of v4.

  python data2/clifford_generation.py --N 11 13 16 20 --splits val
  python data2/clifford_generation.py --check   # compare with Aer on 13 qubits
"""

import argparse
import os
import time
from functools import lru_cache

import generation as v4
import h5py
import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import SGate
from qiskit.quantum_info import Chi, Clifford, StabilizerState, pauli_basis
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, device, pauli_error

NUM_QUBITS_LIST = [11, 13, 16, 20, 24, 32, 40, 48, 56, 63, 80, 100, 112]
# The greedy order adds qubits with readout errors of 0.25 to 1 and broken couplings
# (error 1) after the 112th; they are left out, as on hardware.
MAX_NUM_QUBITS = 112
# Bits of one int64 word of a bitstring, as in dequbit.sparse; the sign bit stays unused.
WORD_BITS = 63
CIRCUIT_TYPES = ["bv", "hidden_shift", "ghz"]
NUM_SHOTS = v4.NUM_SHOTS
COUNT_PER_TYPE = {"val": 20, "test": 20}
OUTPUT_DIR = "data_clifford"
# Seeds differ from those of generation.py, so no circuit repeats a v4 circuit.
SEED_OFFSET = 1000
# Pauli error probabilities below this are dropped from a twirled channel.
MIN_PAULI_PROBABILITY = 1e-12
# Rotation angles of rz are multiples of pi / 2 up to this tolerance in a Clifford circuit.
ANGLE_TOLERANCE = 1e-6
CHECK_NUM_QUBITS = 13
CHECK_SHOTS = 200_000


# ----------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------
def make_bv(n, rng, neighbours):
    """Bernstein-Vazirani circuit; the ideal output is the secret on qubits 0..n-2."""
    return v4._make_bv(n, rng)


def make_hidden_shift(n, rng, neighbours):
    """Hidden shift of f(x) = sum x_2i x_2i+1, which is its own dual; outputs the shift.

    For odd n the last qubit is in no pair and ends in |+>, so its bit is random.
    """
    shift = rng.integers(0, 2, size=n)
    pairs = [(q, q + 1) for q in range(0, n - 1, 2)]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for q in np.flatnonzero(shift):
        qc.x(int(q))
    for a, b in pairs:
        qc.cz(a, b)
    for q in np.flatnonzero(shift):
        qc.x(int(q))
    qc.h(range(n))
    for a, b in pairs:
        qc.cz(a, b)
    qc.h(range(n))
    return qc


def make_ghz(n, rng, neighbours):
    """GHZ state spread from a random qubit along a breadth-first tree of the couplings.

    Logical qubit i sits on qubit i of the subset, so every CX is on a coupled pair and
    the transpiler adds no SWAP.
    """
    root = int(rng.integers(n))
    qc = QuantumCircuit(n)
    qc.h(root)
    reached, frontier = {root}, [root]
    while frontier:
        parent = frontier.pop(0)
        for child in sorted(neighbours[parent] - reached):
            qc.cx(parent, child)
            reached.add(child)
            frontier.append(child)
    return qc


def subset_neighbours(subset, directed):
    """Coupled positions of every position of the subset, {position: set of positions}."""
    position = {q: i for i, q in enumerate(subset)}
    neighbours = {i: set() for i in range(len(subset))}
    for a, b in directed:
        if a in position and b in position:
            neighbours[position[a]].add(position[b])
            neighbours[position[b]].add(position[a])
    return neighbours


BUILDERS = {"bv": make_bv, "hidden_shift": make_hidden_shift, "ghz": make_ghz}


def ideal_support(qc):
    """Support of P_target of a Clifford circuit.

    Returns:
        Bitstrings as words, shape (support, words), and their probabilities, shape
        (support,).
    """
    probs = StabilizerState(Clifford(qc)).probabilities_dict()
    bitstrings = np.array([to_words(int(key, 2), qc.num_qubits) for key in probs], dtype=np.int64)
    values = np.array(list(probs.values()), dtype=np.float64)
    return bitstrings, values / values.sum()


def num_words(n):
    """Number of WORD_BITS-bit words of a bitstring of n bits."""
    return -(-n // WORD_BITS)


def to_words(value, n):
    """Words of a bitstring given as a Python integer, the lowest bits first."""
    mask = (1 << WORD_BITS) - 1
    return [(value >> (WORD_BITS * word)) & mask for word in range(num_words(n))]


def reference_outcome(qc, seed):
    """One noiseless outcome of a Clifford circuit, one bit per qubit, shape (n,)."""
    state = StabilizerState(Clifford(qc))
    state.seed(seed)
    label = state.sample_memory(1)[0]
    return np.array([int(bit) for bit in label[::-1]], dtype=np.uint8)


# ----------------------------------------------------------------------
# Pauli-twirled device noise
# ----------------------------------------------------------------------
def twirl(error):
    """Pauli twirl of a quantum error: the Pauli channel on the diagonal of its chi matrix.

    Returns:
        Pauli labels (qiskit order, the last character is the first qubit) and their
        probabilities, which sum to 1.
    """
    k = error.num_qubits
    # Chi of a channel has trace 2**k, so the Pauli probabilities are its diagonal / 2**k.
    probs = np.real(np.diag(Chi(error).data)) / 2**k
    labels = pauli_basis(k).to_labels()
    keep = probs > MIN_PAULI_PROBABILITY
    probs = probs[keep] / probs[keep].sum()
    return [label for label, kept in zip(labels, keep) if kept], probs


def _label_bits(label):
    """X and Z bits of a Pauli label per qubit of the instruction, shape (k,) each."""
    chars = label[::-1]
    return (
        np.array([c in "XY" for c in chars], dtype=np.uint8),
        np.array([c in "ZY" for c in chars], dtype=np.uint8),
    )


def device_noise(backend):
    """Twirled gate errors and readout errors of every qubit of the device.

    Returns:
        Gate errors {(name, physical qubits): (error probability, cumulative probabilities
        of the non-identity Paulis, their X bits (paulis, k), their Z bits (paulis, k),
        Pauli labels, all probabilities)} and readout errors {qubit: (P(1|0), P(0|1))}.
    """
    gates = {}
    for name, qubits, error in device.basic_device_gate_errors(target=backend.target):
        labels, probs = twirl(error)
        k = len(qubits)
        flips = [i for i, label in enumerate(labels) if label != "I" * k]
        if not flips:
            continue
        flip_probs = probs[flips]
        bits = [_label_bits(labels[i]) for i in flips]
        gates[(name, tuple(qubits))] = (
            float(flip_probs.sum()),
            np.cumsum(flip_probs) / flip_probs.sum(),
            np.array([b[0] for b in bits]),
            np.array([b[1] for b in bits]),
            labels,
            probs,
        )
    readout = {}
    for qubits, error in device.basic_device_readout_errors(target=backend.target):
        matrix = np.asarray(error.probabilities)
        readout[qubits[0]] = (float(matrix[0, 1]), float(matrix[1, 0]))
    return gates, readout


def aer_noise_model(gates, readout, backend):
    """The same twirled noise as an Aer noise model, to check the frame sampler."""
    noise = NoiseModel(basis_gates=list(backend.target.operation_names))
    for (name, qubits), (_, _, _, _, labels, probs) in gates.items():
        noise.add_quantum_error(pauli_error(list(zip(labels, probs))), name, list(qubits))
    for qubit, (p01, p10) in readout.items():
        noise.add_readout_error([[1 - p01, p01], [p10, 1 - p10]], [qubit])
    return noise


# ----------------------------------------------------------------------
# Pauli frame sampler
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def _symplectic(name, power):
    """Images of X_0..X_k-1, Z_0..Z_k-1 under one gate, as rows (x bits, z bits), (2k, 2k)."""
    if name == "rz":
        qc = QuantumCircuit(1)
        for _ in range(power):
            qc.append(SGate(), [0])
        clifford = Clifford(qc)
    else:
        gate = {"sx": "sx", "x": "x", "ecr": "ecr", "id": "id"}[name]
        qc = QuantumCircuit(2 if gate == "ecr" else 1)
        getattr(qc, gate)(*range(qc.num_qubits))
        clifford = Clifford(qc)
    k = clifford.num_qubits
    return clifford.tableau[:, : 2 * k].astype(np.uint8)


def _rz_power(angle):
    """Number of S gates of rz(angle), which must be a multiple of pi / 2."""
    steps = angle / (np.pi / 2)
    power = int(round(steps))
    if abs(steps - power) > ANGLE_TOLERANCE:
        raise ValueError(f"rz({angle}) is not a Clifford gate")
    return power % 4


def sample_noisy_outcomes(qc_t, reference, gates, readout, shots, rng):
    """Sample noisy outcomes of a transpiled Clifford circuit with Pauli frames.

    Args:
        qc_t: Transpiled circuit on the device with one measurement per logical qubit.
        reference: Noiseless outcome of the logical circuit, one bit per clbit, shape (n,).
        gates: Twirled gate errors of device_noise.
        readout: Readout errors of device_noise.
        shots: Number of shots.
        rng: Random generator.

    Returns:
        Outcomes as words (bit j = clbit j), shape (shots, words).
    """
    used = sorted({qc_t.find_bit(q).index for inst in qc_t.data for q in inst.qubits})
    row = {q: i for i, q in enumerate(used)}
    # A Z frame on |0> changes nothing; drawing it at random makes the outcomes that are
    # random in the ideal circuit random in every shot, so one reference outcome suffices.
    x_frame = np.zeros((len(used), shots), dtype=np.uint8)
    z_frame = rng.integers(0, 2, size=(len(used), shots), dtype=np.uint8)
    outcomes = np.zeros((shots, num_words(len(reference))), dtype=np.int64)

    for inst in qc_t.data:
        name = inst.operation.name
        qubits = tuple(qc_t.find_bit(q).index for q in inst.qubits)
        rows = [row[q] for q in qubits]
        if name in ("barrier", "delay"):
            continue
        if name == "measure":
            _apply_error(gates.get(("measure", qubits)), x_frame, z_frame, rows, rng)
            clbit = qc_t.find_bit(inst.clbits[0]).index
            bits = reference[clbit] ^ x_frame[rows[0]]
            bits = _readout_flip(bits, readout[qubits[0]], rng)
            word, bit = divmod(clbit, WORD_BITS)
            outcomes[:, word] |= bits.astype(np.int64) << bit
            continue
        power = _rz_power(float(inst.operation.params[0])) if name == "rz" else 0
        matrix = _symplectic(name, power)
        frame = np.concatenate([x_frame[rows], z_frame[rows]])
        # Row r of matrix is the image of generator r, so the new frame bits are the XOR of
        # the images of the generators present in the old frame.
        updated = (matrix.T.astype(np.int32) @ frame.astype(np.int32)) % 2
        k = len(rows)
        x_frame[rows], z_frame[rows] = updated[:k], updated[k:]
        _apply_error(gates.get((name, qubits)), x_frame, z_frame, rows, rng)

    return outcomes


def _apply_error(error, x_frame, z_frame, rows, rng):
    """Multiply the frames of the shots hit by the twirled error of one gate."""
    if error is None:
        return
    flip_probability, cumulative, x_bits, z_bits = error[:4]
    shots = x_frame.shape[1]
    draws = rng.random(shots)
    hit = np.flatnonzero(draws < flip_probability)
    if len(hit) == 0:
        return
    which = np.searchsorted(cumulative, draws[hit] / flip_probability, side="right")
    which = np.minimum(which, len(cumulative) - 1)
    for i, r in enumerate(rows):
        x_frame[r, hit] ^= x_bits[which, i]
        z_frame[r, hit] ^= z_bits[which, i]


def _readout_flip(bits, rates, rng):
    """Flip measured bits with P(1|0) where the bit is 0 and P(0|1) where it is 1."""
    p01, p10 = rates
    flip_rate = np.where(bits == 0, p01, p10)
    return bits ^ (rng.random(len(bits)) < flip_rate).astype(np.uint8)


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def build_subsets(backend):
    """Nested qubit subsets: those of generation.py up to 13, grown greedily beyond."""
    subsets, directed, g2 = v4.build_nested_subsets(backend)
    info = v4._device_info(backend)
    previous = max(subsets)
    for n in sorted(NUM_QUBITS_LIST):
        if n > previous:
            subsets[n] = v4._grow_subset(info, subsets[previous], n)
            previous = n
    return subsets, directed, g2


def process_circuit(qc, n, subset, backend, gates, readout, shots, seed):
    """Transpile one circuit and sample its noisy outcomes and its ideal support."""
    rng = np.random.default_rng(seed)
    qc_log = transpile(qc, basis_gates=["h", "s", "sdg", "x", "cx", "cz"], optimization_level=0)
    support, probs = ideal_support(qc_log)
    reference = reference_outcome(qc_log, seed)

    qm = qc.copy()
    qm.measure_all()
    qc_t = v4.transpile_in_subset(qm, backend, subset, seed)
    outcomes = sample_noisy_outcomes(qc_t, reference, gates, readout, shots, rng)
    bitstrings, counts = np.unique(outcomes, axis=0, return_counts=True)
    return qc_log, qc_t, bitstrings, counts.astype(np.int32), support, probs


def write_split(split, n, count, shots, backend, subset, directed, calibration, gates, readout):
    """Write clifford_{split}_N{n}.h5; a finished file is skipped."""
    path = os.path.join(OUTPUT_DIR, f"clifford_{split}_N{n}.h5")
    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            if bool(f.attrs.get("complete")):
                print(f"  skip {split} N={n} (complete)")
                return

    start = time.perf_counter()
    metas = []
    with h5py.File(path, "w") as f:
        total = count * len(CIRCUIT_TYPES)
        f.attrs.update(
            num_qubits=n,
            num_circuits=total,
            num_shots=shots,
            split=split,
            kind="noisy_clifford",
            complete=False,
            bit_order="index bit j = qubit j",
        )
        noise_group = f.create_group("noise_model")
        noise_group.attrs["source"] = "FakeSherbrooke gate errors, Pauli twirled; Pauli frames"
        noise_group.create_dataset("subset_qubits", data=np.array(subset, dtype=np.int32))
        inside = set(subset)
        noise_group.create_dataset(
            "coupling_map_directed",
            data=np.array([[a, b] for a, b in directed if a in inside and b in inside]),
        )
        for name, values in calibration.items():
            noise_group.create_dataset(name, data=values.astype(np.float64))
        for group in ("noisy_bitstrings", "noisy_counts", "target_bitstrings", "target_probs"):
            f.create_group(group)
        f.create_group("qasm")

        cid = 0
        for type_index, circuit_type in enumerate(CIRCUIT_TYPES):
            for i in range(count):
                key = f"circuit_{cid:05d}"
                seed = v4.int_seed("circuit", split, n, SEED_OFFSET + type_index, i)
                neighbours = subset_neighbours(subset, directed)
                qc = BUILDERS[circuit_type](n, np.random.default_rng(seed), neighbours)
                qc_log, qc_t, bitstrings, counts, support, probs = process_circuit(
                    qc, n, subset, backend, gates, readout, shots, seed
                )
                f["noisy_bitstrings"].create_dataset(key, data=bitstrings)
                f["noisy_counts"].create_dataset(key, data=counts)
                f["target_bitstrings"].create_dataset(key, data=support)
                f["target_probs"].create_dataset(key, data=probs)
                f["qasm"].create_dataset(key, data=np.bytes_(qasm2.dumps(qc_log)))
                one_q, two_q, measured = v4.physical_gate_counts(qc_t, subset, n)
                metas.append(
                    dict(
                        circuit_type=circuit_type,
                        num_gates_physical=int(qc_t.size()),
                        depth_physical=int(qc_t.depth()),
                        gates_1q_per_qubit=one_q,
                        gates_2q_per_pair=two_q,
                        measured_qubit=measured,
                    )
                )
                cid += 1
            print(
                f"  [{split} N={n}] {circuit_type}: {cid}/{total} "
                f"({time.perf_counter() - start:.0f}s)",
                flush=True,
            )

        meta = f.create_group("circuit_meta")
        for name in ("num_gates_physical", "depth_physical"):
            meta.create_dataset(name, data=np.array([m[name] for m in metas], dtype=np.int32))
        for name in ("gates_1q_per_qubit", "gates_2q_per_pair", "measured_qubit"):
            meta.create_dataset(name, data=np.stack([m[name] for m in metas]).astype(np.int32))
        meta.create_dataset(
            "circuit_type",
            data=np.array([m["circuit_type"] for m in metas], dtype=object),
            dtype=h5py.string_dtype(),
        )
        f.attrs["complete"] = True


# ----------------------------------------------------------------------
# Check against Aer
# ----------------------------------------------------------------------
def check(backend, subsets, directed, gates, readout):
    """Compare the frame sampler with Aer's stabilizer simulator under the same noise."""
    sim = AerSimulator(method="stabilizer", noise_model=aer_noise_model(gates, readout, backend))
    n = CHECK_NUM_QUBITS
    for type_index, circuit_type in enumerate(CIRCUIT_TYPES):
        seed = v4.int_seed("circuit", "val", n, SEED_OFFSET + type_index, 0)
        neighbours = subset_neighbours(subsets[n], directed)
        qc = BUILDERS[circuit_type](n, np.random.default_rng(seed), neighbours)
        start = time.perf_counter()
        _, qc_t, bitstrings, counts, support, _ = process_circuit(
            qc, n, subsets[n], backend, gates, readout, CHECK_SHOTS, seed
        )
        frames_time = time.perf_counter() - start
        start = time.perf_counter()
        aer = sim.run(qc_t, shots=CHECK_SHOTS, seed_simulator=seed).result().get_counts()
        aer_time = time.perf_counter() - start

        ours = dict(zip(bitstrings[:, 0].tolist(), counts.tolist()))
        theirs = {int(key.replace(" ", ""), 2): value for key, value in aer.items()}
        keys = set(ours) | set(theirs)
        tv = 0.5 * sum(abs(ours.get(k, 0) - theirs.get(k, 0)) for k in keys) / CHECK_SHOTS
        success = [
            sum(d.get(int(s), 0) for s in support[:, 0]) / CHECK_SHOTS for d in (ours, theirs)
        ]
        print(
            f"{circuit_type}: TV {tv:.4f}, P(ideal support) frames {success[0]:.4f} "
            f"Aer {success[1]:.4f}, time frames {frames_time:.1f}s Aer {aer_time:.1f}s"
        )


def main():
    """Generate the files of the given splits and sizes, or check the sampler."""
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs="*", default=NUM_QUBITS_LIST)
    parser.add_argument("--splits", nargs="*", default=["val", "test"])
    parser.add_argument("--count", type=int, default=None, help="circuits per type")
    parser.add_argument("--shots", type=int, default=NUM_SHOTS)
    parser.add_argument("--out", default=OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if max(args.N) > MAX_NUM_QUBITS:
        raise ValueError(f"at most {MAX_NUM_QUBITS} qubits")

    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    backend = FakeSherbrooke()
    subsets, directed, g2 = build_subsets(backend)
    gates, readout = device_noise(backend)
    if args.check:
        check(backend, subsets, directed, gates, readout)
        return

    OUTPUT_DIR = args.out
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in args.splits:
        for n in args.N:
            count = args.count if args.count is not None else COUNT_PER_TYPE[split]
            print(f"== {split} N={n}: {count} circuits per type, {args.shots} shots ==")
            write_split(
                split,
                n,
                count,
                args.shots,
                backend,
                subsets[n],
                directed,
                v4.calibration_of_subset(backend, subsets[n], g2),
                gates,
                readout,
            )


if __name__ == "__main__":
    main()
