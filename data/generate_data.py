"""
Генерує 4 файли для QEM-проєкту:
  1_clean_train.h5  — 2000 схем × 256 бітстрінгів (ідеальні розподіли)
  2_clean_test.h5   — 200 схем × 256 бітстрінгів (ідеальні розподіли)
  3_noisy_train.h5  — 2000 схем × 10000 шумних shots
  4_noisy_test.h5   — 200 схем × 10000 шумних shots (окремий seed)

Схеми синхронізовані: test-схеми НЕ перетинаються з train.
Noise model однаковий для train і test.
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


NUM_QUBITS = 8
DEPTH_LEVEL = 3
NUM_BRICKS = DEPTH_LEVEL

N_TRAIN_CIRCUITS = 2000
N_TEST_CIRCUITS = 200

N_SHOTS = 10_000
N_BITSTRINGS = 2 ** NUM_QUBITS

SEED_TRAIN_CIRCUITS = 42
SEED_TEST_CIRCUITS = 9999
SEED_NOISE_TRAIN = 123
SEED_NOISE_TEST = 456

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_pauli_gadget_brick(num_qubits, rng):
    """Генерує один Pauli gadget brick (1- і 2-кубітні гейти)."""
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
    """Складає схему з кількох bricks, без вимірювань."""
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_bricks):
        qc.compose(generate_pauli_gadget_brick(num_qubits, rng), inplace=True)
    return qc


def generate_circuit_set(num_circuits, seed):
    """Генерує набір схем з детермінованим seed."""
    circuits = []
    for cid in range(num_circuits):
        rng = np.random.default_rng(seed + cid)
        qc = build_circuit(NUM_QUBITS, NUM_BRICKS, rng)
        circuits.append(qc)
    return circuits


def circuit_to_qasm(qc):
    """Серіалізує схему в QASM-рядок для збереження."""
    from qiskit import qasm2
    return qasm2.dumps(qc)


def build_complex_noise_model(num_qubits, seed):
    """Складний, але однаковий для всіх схем noise model."""
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
        "p1": p1,
        "p2": p2,
        "p_ro": p_ro,
        "T1": T1,
        "T2": T2,
        "t_gate_1q": t_gate_1q,
        "t_gate_2q": t_gate_2q,
        "seed": seed,
    }
    return noise_model, params


def compute_ideal_probs(qc, sim_ideal):
    """Повертає ідеальний розподіл ймовірностей (256 float32)."""
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    qc_no_meas.save_statevector()
    result = sim_ideal.run(qc_no_meas).result()
    sv = result.get_statevector()
    probs = np.abs(sv) ** 2
    probs = probs / probs.sum()
    return probs.astype(np.float32)


def compute_noisy_bitstrings(qc, sim_noisy, shots, seed):
    """Повертає масив (shots, num_qubits) uint8 з шумними бітстрінгами."""
    qc_meas = qc.copy()
    qc_meas.measure_all()
    result = sim_noisy.run(
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


def generate_clean_file(
    path, circuits, sim_ideal, split_name, seed_circuits
):
    """Генерує файл з ідеальними розподілами."""
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
        f.attrs["split"] = split_name
        f.attrs["seed_circuits"] = seed_circuits
        f.attrs["num_bitstrings"] = N_BITSTRINGS

        f.create_dataset("all_bitstrings", data=all_bitstrings)

        probs_grp = f.create_group("probs")
        qasm_grp = f.create_group("qasm")

        for cid, qc in enumerate(circuits):
            key = f"circuit_{cid:04d}"
            probs = compute_ideal_probs(qc, sim_ideal)
            probs_grp.create_dataset(key, data=probs)
            qasm_grp.create_dataset(
                key,
                data=np.bytes_(circuit_to_qasm(qc)),
            )
            if cid % 100 == 0:
                print(f"  [{split_name}] clean {cid}/{len(circuits)}")

        f.attrs["P_target_sum_check"] = float(
            probs_grp["circuit_0000"][:].sum()
        )


def generate_noisy_file(
    path, circuits, noise_model, noise_params, split_name, seed_noise
):
    """Генерує файл з шумними бітстрінгами."""
    sim_noisy = AerSimulator(noise_model=noise_model)

    with h5py.File(path, "w") as f:
        f.attrs["num_qubits"] = NUM_QUBITS
        f.attrs["num_bricks"] = NUM_BRICKS
        f.attrs["num_circuits"] = len(circuits)
        f.attrs["num_shots"] = N_SHOTS
        f.attrs["split"] = split_name
        f.attrs["seed_noise"] = seed_noise

        noise_grp = f.create_group("noise_model")
        for k, v in noise_params.items():
            noise_grp.attrs[k] = v

        bitstrings_grp = f.create_group("noisy_bitstrings")
        hist_grp = f.create_group("noisy_histograms")

        for cid, qc in enumerate(circuits):
            key = f"circuit_{cid:04d}"
            seed = seed_noise + cid
            bitstrings = compute_noisy_bitstrings(
                qc, sim_noisy, N_SHOTS, seed
            )
            bitstrings_grp.create_dataset(key, data=bitstrings)

            hist = np.zeros(N_BITSTRINGS, dtype=np.int64)
            for bs in bitstrings:
                idx = 0
                for b in bs:
                    idx = (idx << 1) | int(b)
                hist[idx] += 1
            hist_grp.create_dataset(key, data=hist)

            if cid % 100 == 0:
                print(f"  [{split_name}] noisy {cid}/{len(circuits)}")


def main():
    print("=" * 60)
    print("Генерація даних для QEM-проєкту")
    print("=" * 60)

    print("\n[1/4] Генерація train-схем...")
    train_circuits = generate_circuit_set(
        N_TRAIN_CIRCUITS, SEED_TRAIN_CIRCUITS
    )

    print("[2/4] Генерація test-схем...")
    test_circuits = generate_circuit_set(
        N_TEST_CIRCUITS, SEED_TEST_CIRCUITS
    )

    print("\n[3/4] Побудова noise model...")
    noise_model, noise_params = build_complex_noise_model(
        NUM_QUBITS, SEED_NOISE_TRAIN
    )
    print(f"  Параметри шуму: {noise_params}")

    sim_ideal = AerSimulator(method="statevector")

    print("\n[4/4] Генерація clean-файлів...")
    generate_clean_file(
        os.path.join(OUTPUT_DIR, "1_clean_train.h5"),
        train_circuits,
        sim_ideal,
        "train",
        SEED_TRAIN_CIRCUITS,
    )
    generate_clean_file(
        os.path.join(OUTPUT_DIR, "2_clean_test.h5"),
        test_circuits,
        sim_ideal,
        "test",
        SEED_TEST_CIRCUITS,
    )

    print("\n[5/5] Генерація noisy-файлів...")
    generate_noisy_file(
        os.path.join(OUTPUT_DIR, "3_noisy_train.h5"),
        train_circuits,
        noise_model,
        noise_params,
        "train",
        SEED_NOISE_TRAIN,
    )
    generate_noisy_file(
        os.path.join(OUTPUT_DIR, "4_noisy_test.h5"),
        test_circuits,
        noise_model,
        noise_params,
        "test",
        SEED_NOISE_TEST,
    )

    print("\n" + "=" * 60)
    print("Готово! Файли в директорії:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
