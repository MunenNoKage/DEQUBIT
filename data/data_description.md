# README: Датасети для QEM-проєкту

## Огляд

4 HDF5-файли з даними для QEM-проєкту. 8 кубітів, 3 bricks глибина, 10 000 shots на схему.

---

## Як генерувалися дані

- **Схеми**: Pauli gadget bricks. Кожен brick — випадкові 1-кубітні гейти (`rx`, `ry`, `rz`) + 2-кубітні `cx` між випадковими парами. 3 bricks на схему.
- **Ідеальні розподіли**: аналітично через `statevector` (без shots).
- **Шумні вимірювання**: `AerSimulator` з noise model, 10 000 shots на схему.
- **Train**: 2000 схем (seed=42). **Test**: 200 схем (seed=9999). Не перетинаються.

---

## Як генерувався шум

Через `qiskit_aer.noise.NoiseModel` — три компоненти:

1. **Depolarizing error**
   - 1-кубітні гейти: `p1 = 0.005`
   - 2-кубітні гейти: `p2 = 0.02`

2. **Thermal relaxation**
   - `T1 = 80 µs`, `T2 = 60 µs`
   - `t_gate_1q = 100 ns`, `t_gate_2q = 300 ns`

3. **Readout error**
   - `p_ro = 0.02` (однаковий для всіх кубітів)

**Однаковий noise model для train і test.** Різні seed'и для shots: train = 123, test = 456. Всі параметри збережені в `noise_model/attrs`.

---

## Що є в яких файлах

### `1_clean_train.h5`
```
attrs:        num_qubits=8, num_circuits=2000, split="train", seed=42
all_bitstrings:  (256, 8) uint8   — всі можливі бітстрінги
probs/circuit_XXXX:  (256,) float32 — P_target для схеми XXXX
qasm/circuit_XXXX:   bytes          — QASM-опис схеми XXXX
```

### `2_clean_test.h5`
Те саме, але `num_circuits=200`, `split="test"`, `seed=9999`.

### `3_noisy_train.h5`
```
attrs:        num_qubits=8, num_circuits=2000, num_shots=10000,
              split="train", seed_noise=123
noise_model/:  attrs — p1, p2, p_ro, T1, T2, ...
noisy_bitstrings/circuit_XXXX:  (10000, 8) uint8 — сирі шумні shots
noisy_histograms/circuit_XXXX:  (256,) int64      — гістограма
```

### `4_noisy_test.h5`
Те саме, але `num_circuits=200`, `split="test"`, `seed_noise=456`.

---

## Як читати

```python
import h5py

with h5py.File("data/3_noisy_train.h5", "r") as f:
    shots = f["noisy_bitstrings/circuit_0000"][:]   # (10000, 8) uint8
    hist  = f["noisy_histograms/circuit_0000"][:]   # (256,) int64
    p1    = f["noise_model"].attrs["p1"]             # 0.005
```

---

## Обмеження

- **Немає топології**: всі пари кубітів з'єднані, `cx` між будь-якими кубітами.
- **Однаковий шум**: p1, p2, p_ro, T1, T2 однакові для всіх кубітів.
- **Немає crosstalk**: сусідні гейти не впливають один на одного.

Це свідоме спрощення для baseline. Для реалістичніших результатів (Step 2, Step 3) планується додати топологію та per-qubit шум.
