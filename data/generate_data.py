"""
Генерує 4 файли для QEM-проєкту (5 кубітів):

  1_clean_train.h5 — 2000 схем: точний розподіл + сирі чисті shots
  2_clean_test.h5  — 200 схем:  точний розподіл + сирі чисті shots
  3_noisy_train.h5 — 2000 схем: ті самі схеми з накладеним шумом (сирі шумні shots)
  4_noisy_test.h5  — 200 схем:  ті самі test-схеми з шумом

Схеми синхронізовані:
  - train: 2000 унікальних схем (seed 42)
  - test:  200 унікальних схем, НЕ перетинаються з train (seed 9999)
  - для кожної схеми зберігаються shots і чисті, і шумні
  - noise model однаковий для train і test

Бітстрінги пакуються через np.packbits (5 біт → 1 байт).
"""

import os
import numpy as np
import h5py
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)


NUM_QUBITS = 5
DEPTH_LEVEL = 3
NUM_BRICKS = DEPTH_LEVEL

N_TRAIN_CIRCUITS = 2000
N_TEST_CIRCUITS = 200

N_SHOTS = 10_000
N_BITSTRINGS = 2 ** NUM_QUBITS  # 32

SEED_TRAIN_CIRCUITS = 42
SEED_TEST_CIRCUITS = 9999
SEED_NOISE_TRAIN = 123
SEED_NOISE_TEST = 456

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Генерація схем
# ----------------------------------------------------------------------

def generate_pauli_gadget_brick(num_qubits, rng):
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_qubits):
        gate = rng.choice(["rx", "ry", "rz"])
        angle = rng.uniform(0, 2 * np.pi)
        qubit = int(rng.integers(0, num_qubits))
        getattr(qc, gate)(angle, qubit)
    for _ in range(num_qubits - 1):
        q1 = int(rng.integers(0, num_qubits))
        q2 = int(rng.integers(0, num_qubits))
        while q2 == q1:
            q2 = int(rng.integers(0, num_qubits))
        qc.cx(q1, q2)
    return qc


def build_circuit(num_qubits, num_bricks, rng):
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_bricks):
        qc.compose(generate_pauli_gadget_brick(num_qubits, rng), inplace=True)
    return qc


def generate_circuit_set(num_circuits, seed):
    circuits = []
    for cid in range(num_circuits):
        rng = np.random.default_rng(seed + cid)
        qc = build_circuit(NUM_QUBITS, NUM_BRICKS, rng)
        circuits.append(qc)
    return circuits


def circuit_to_qasm(qc):
    from qiskit import qasm2
    return qasm2.dumps(qc)


# ----------------------------------------------------------------------
# Noise model
# ----------------------------------------------------------------------

def build_complex_noise_model(num_qubits, seed):
    noise_model = NoiseModel()

    p1 = 0.005
    p2 = 0.02
    p_ro = 0.02
    T1 = 80e-6
    T2 = 60e-6
    t_gate_1q = 100e-9
    t_gate_2q = 300e-9

    err_1q_dep = depolarizing_error(p1, 1)
    err_1q_therm = thermal_relaxation_error(T1, T2, t_gate_1q)
    err_1q = err_1q_dep.compose(err_1q_therm)
    noise_model.add_all_qubit_quantum_error(
        err_1q, ["rx", "ry", "rz", "x", "h"]
    )

    err_2q_dep = depolarizing_error(p2, 2)
    err_2q_therm = thermal_relaxation_error(T1, T2, t_gate_2q).tensor(
        thermal_relaxation_error(T1, T2, t_gate_2q)
    )
    err_2q = err_2q_dep.compose(err_2q_therm)
    noise_model.add_all_qubit_quantum_error(err_2q, ["cx", "cz"])

    readout_err = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    noise_model.add_all_qubit_readout_error(readout_err)

    params = {
        "p1": p1, "p2": p2, "p_ro": p_ro,
        "T1": T1, "T2": T2,
        "t_gate_1q": t_gate_1q, "t_gate_2q": t_gate_2q,
        "seed": seed,
    }
    return noise_model, params


# ----------------------------------------------------------------------
# Обчислення
# ----------------------------------------------------------------------

def compute_ideal_probs(qc, sim_ideal):
    """Точний розподіл ймовірностей (2^N float32) через statevector."""
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    qc_no_meas.save_statevector()
    result = sim_ideal.run(qc_no_meas).result()
    sv = result.get_statevector()
    probs = np.abs(sv) ** 2
    probs = probs / probs.sum()
    return probs.astype(np.float32)


def compute_bitstrings(qc, simulator, shots, seed):
    """Сирі shots (shots, NUM_QUBITS) uint8.

    Працює і для чистого shot-based симулятора, і для шумного —
    просто передай відповідний `simulator`.
    """
    qc_meas = qc.copy()
    qc_meas.measure_all()
    result = simulator.run(
        qc_meas, shots=shots, seed_simulator=seed
    ).result()
    counts = result.get_counts()

    bitstrings = np.empty((shots, NUM_QUBITS), dtype=np.uint8)
    idx = 0
    for bs, n in counts.items():
        bits = np.array([int(b) for b in bs[::-1]], dtype=np.uint8)
        bitstrings[idx:idx + n] = bits
        idx += n

    rng = np.random.default_rng(seed)
    rng.shuffle(bitstrings)
    return bitstrings


def pack_bitstrings(bitstrings):
    """(shots, N) uint8 → (shots, ceil(N/8)) uint8."""
    return np.packbits(bitstrings, axis=1)


# ----------------------------------------------------------------------
# Запис файлів
# ----------------------------------------------------------------------

def write_clean_file(
    path, circuits, sim_ideal, sim_clean_shots,
    split_name, seed_circuits, seed_shots,
):
    """Файл 1/2: точний розподіл + сирі чисті shots (packed)."""
    all_bitstrings = np.array(
        [
            [(i >> (NUM_QUBITS - 1 - j)) & 1 for j in range(NUM_QUBITS)]
            for i in range(N_BITSTRINGS)
        ],
        dtype=np.uint8,
    )

    with h5py.File(path, "w") as f:
        f.attrs["num_qubits"] = NUM_QUBITS
        f.attrs["num_bricks"] = NUM_BRICKS
        f.attrs["num_circuits"] = len(circuits)
        f.attrs["num_shots"] = N_SHOTS
        f.attrs["split"] = split_name
        f.attrs["seed_circuits"] = seed_circuits
        f.attrs["seed_shots"] = seed_shots
        f.attrs["num_bitstrings"] = N_BITSTRINGS

        f.create_dataset("all_bitstrings", data=all_bitstrings)

        probs_grp = f.create_group("probs")
        qasm_grp = f.create_group("qasm")
        bits_grp = f.create_group("clean_bitstrings")
        bits_grp.attrs["packed"] = True
        bits_grp.attrs["bits_per_shot"] = NUM_QUBITS

        for cid, qc in enumerate(circuits):
            key = f"circuit_{cid:04d}"

            probs = compute_ideal_probs(qc, sim_ideal)
            probs_grp.create_dataset(key, data=probs)

            qasm_grp.create_dataset(
                key, data=np.bytes_(circuit_to_qasm(qc))
            )

            seed = seed_shots + cid
            bitstrings = compute_bitstrings(
                qc, sim_clean_shots, N_SHOTS, seed
            )
            bits_grp.create_dataset(
                key, data=pack_bitstrings(bitstrings)
            )

            if cid % 100 == 0:
                print(f"  [{split_name}] clean {cid}/{len(circuits)}")

        f.attrs["P_target_sum_check"] = float(
            probs_grp["circuit_0000"][:].sum()
        )


def write_noisy_file(
    path, circuits, noise_model, noise_params,
    split_name, seed_noise,
):
    """Файл 3/4: ті самі схеми, але з накладеним шумом (packed shots)."""
    sim_noisy = AerSimulator(noise_model=noise_model)

    with h5py.File(path, "w") as f:
        f.attrs["num_qubits"] = NUM_QUBITS
        f.attrs["num_bricks"] = NUM_BRICKS
        f.attrs["num_circuits"] = len(circuits)
        f.attrs["num_shots"] = N_SHOTS
        f.attrs["split"] = split_name
        f.attrs["seed_noise"] = seed_noise
        f.attrs["num_bitstrings"] = N_BITSTRINGS

        noise_grp = f.create_group("noise_model")
        for k, v in noise_params.items():
            noise_grp.attrs[k] = v

        bits_grp = f.create_group("noisy_bitstrings")
        bits_grp.attrs["packed"] = True
        bits_grp.attrs["bits_per_shot"] = NUM_QUBITS

        for cid, qc in enumerate(circuits):
            key = f"circuit_{cid:04d}"
            seed = seed_noise + cid
            bitstrings = compute_bitstrings(
                qc, sim_noisy, N_SHOTS, seed
            )
            bits_grp.create_dataset(
                key, data=pack_bitstrings(bitstrings)
            )

            if cid % 100 == 0:
                print(f"  [{split_name}] noisy {cid}/{len(circuits)}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"Генерація даних QEM ({NUM_QUBITS} кубітів)")
    print("=" * 60)

    print("\n[1/5] Генерація train-схем...")
    train_circuits = generate_circuit_set(
        N_TRAIN_CIRCUITS, SEED_TRAIN_CIRCUITS
    )

    print("[2/5] Генерація test-схем...")
    test_circuits = generate_circuit_set(
        N_TEST_CIRCUITS, SEED_TEST_CIRCUITS
    )

    print("\n[3/5] Побудова noise model...")
    noise_model, noise_params = build_complex_noise_model(
        NUM_QUBITS, SEED_NOISE_TRAIN
    )
    print(f"  Параметри шуму: {noise_params}")

    sim_ideal = AerSimulator(method="statevector")
    sim_clean_shots = AerSimulator()

    print("\n[4/5] Файли 1 і 2: чисті розподіли + чисті shots...")
    write_clean_file(
        os.path.join(OUTPUT_DIR, "1_clean_train.h5"),
        train_circuits, sim_ideal, sim_clean_shots,
        "train", SEED_TRAIN_CIRCUITS, SEED_NOISE_TRAIN,
    )
    write_clean_file(
        os.path.join(OUTPUT_DIR, "2_clean_test.h5"),
        test_circuits, sim_ideal, sim_clean_shots,
        "test", SEED_TEST_CIRCUITS, SEED_NOISE_TEST,
    )

    print("\n[5/5] Файли 3 і 4: ті самі схеми з шумом...")
    write_noisy_file(
        os.path.join(OUTPUT_DIR, "3_noisy_train.h5"),
        train_circuits, noise_model, noise_params,
        "train", SEED_NOISE_TRAIN,
    )
    write_noisy_file(
        os.path.join(OUTPUT_DIR, "4_noisy_test.h5"),
        test_circuits, noise_model, noise_params,
        "test", SEED_NOISE_TEST,
    )

    print("\n" + "=" * 60)
    print("Готово! Файли в директорії:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
