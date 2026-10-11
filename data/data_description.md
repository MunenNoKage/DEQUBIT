# Датасети для QEM-проєкту

## Огляд

4 HDF5-файли з даними для QEM-проєкту. **5 кубітів**, 3 bricks глибина,
10 000 shots на схему.

Для кожної схеми зберігаються **і чисті, і шумні shots** — щоб ML-модель
могла вчитися відображати зашумлений розподіл у чистий.

Бітстрінги запаковані через `np.packbits` (5 біт → 1 байт), тому всі файли
< 100 МБ і пушляться на GitHub без LFS.

---

## Як генерувалися дані

- **Схеми**: Pauli gadget bricks. Кожен brick — випадкові 1-кубітні гейти
  (`rx`, `ry`, `rz`) + 2-кубітні `cx` між випадковими парами. 3 bricks на схему.
- **Ідеальні розподіли**: аналітично через `statevector` (без shots).
- **Чисті shots**: `AerSimulator` без noise model, 10 000 shots на схему.
- **Шумні shots**: `AerSimulator` з noise model, 10 000 shots на схему,
  **на тих самих схемах**, що й чисті.
- **Train**: 2000 схем (seed=42). **Test**: 200 схем (seed=9999).
  Схеми не перетинаються — кожна будується від `seed + cid`.

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

**Однаковий noise model для train і test.** Різні seed'и для shots:
train = 123, test = 456. Всі параметри збережені в `noise_model/attrs`.

---

## Що є в яких файлах

### `1_clean_train.h5`
```
attrs:  num_qubits=5, num_bricks=3, num_circuits=2000,
        num_shots=10000, split="train",
        seed_circuits=42, seed_shots=123, num_bitstrings=32

all_bitstrings:                  (32, 5) uint8      — всі можливі бітстрінги
probs/circuit_XXXX:              (32,) float32      — точний розподіл P_target
qasm/circuit_XXXX:               bytes              — QASM-опис схеми
clean_bitstrings/circuit_XXXX:   (10000, 1) uint8   — чисті shots (packed)
```

### `2_clean_test.h5`
Те саме, але `num_circuits=200`, `split="test"`,
`seed_circuits=9999`, `seed_shots=456`.

### `3_noisy_train.h5`
```
attrs:  num_qubits=5, num_bricks=3, num_circuits=2000,
        num_shots=10000, split="train",
        seed_noise=123, num_bitstrings=32

noise_model/:                     attrs — p1, p2, p_ro, T1, T2, ...
noisy_bitstrings/circuit_XXXX:    (10000, 1) uint8 — шумні shots (packed)
```

### `4_noisy_test.h5`
Те саме, але `num_circuits=200`, `split="test"`, `seed_noise=456`.

---

## Пакування бітстрінгів

Кожен shot — це 5 бітів (`uint8` кожен). Щоб зменшити розмір у 5 разів,
вони пакуються через `np.packbits` в 1 байт на shot.

Щоб отримати звичайний масив `(10000, 5)`:

```python
import numpy as np
packed = f["clean_bitstrings/circuit_0000"][:]   # (10000, 1) uint8
shots  = np.unpackbits(packed, axis=1)[:, :5]    # (10000, 5) uint8
```

Атрибут `clean_bitstrings.attrs["packed"] == True` і
`bits_per_shot == 5` підтверджують формат.

---

## Як читати

```python
import h5py
import numpy as np

# Чистий train: розподіл + чисті shots
with h5py.File("data/1_clean_train.h5", "r") as f:
    probs  = f["probs/circuit_0000"][:]              # (32,) float32
    packed = f["clean_bitstrings/circuit_0000"][:]   # (10000, 1) uint8
    clean  = np.unpackbits(packed, axis=1)[:, :5]    # (10000, 5) uint8

# Шумний train: ті самі схеми, шумні shots
with h5py.File("data/3_noisy_train.h5", "r") as f:
    packed = f["noisy_bitstrings/circuit_0000"][:]   # (10000, 1) uint8
    noisy  = np.unpackbits(packed, axis=1)[:, :5]    # (10000, 5) uint8
    p1     = f["noise_model"].attrs["p1"]            # 0.005
```

**Важливо**: `circuit_XXXX` у файлі `1` і `circuit_XXXX` у файлі `3` —
це **та сама схема**. Тобто для схеми №XXXX у тебе є:
- точний чистий розподіл (`probs`),
- чисті shots (`clean_bitstrings`),
- шумні shots (`noisy_bitstrings`, у файлі `3`).

Це і є пара "до зашумлення / після зашумлення" для навчання ML-моделі.

---

## Розміри файлів (приблизно)

| Файл              | Схем | Розмір    |
|-------------------|------|-----------|
| `1_clean_train.h5`| 2000 | ~12.5 МБ  |
| `2_clean_test.h5` | 200  | ~1.3 МБ   |
| `3_noisy_train.h5`| 2000 | ~12.5 МБ  |
| `4_noisy_test.h5` | 200  | ~1.3 МБ   |

Усі < 100 МБ → GitHub без LFS.

---

## Обмеження

- **Немає топології**: всі пари кубітів з'єднані, `cx` між будь-якими кубітами.
- **Однаковий шум**: p1, p2, p_ro, T1, T2 однакові для всіх кубітів.
- **Немає crosstalk**: сусідні гейти не впливають один на одного.
- **Немає shot-to-shot відповідності**: чисті та шумні shots — це дві окремі
  вибірки з одного й того самого стану, а не "той самий shot до і після шуму".
  Aer не дає API для траєкторії окремого shot. Для QEM-навчання на розподілах
  цього достатньо.

Це свідоме спрощення для baseline. Для реалістичніших результатів (Step 2,
Step 3) планується додати топологію та per-qubit шум.