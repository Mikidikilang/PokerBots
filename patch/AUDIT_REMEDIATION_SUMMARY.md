# PokerAI-NLHE — Audit Remediáció v0.3.0
# Prioritizált Javítási Összefoglaló

## Státusz: 12/12 Fix Implementálva

---

## 🔴 KRITIKUS (C1–C4)

---

### C4 — 3-Bet Tultüntetés (LEGMAGASABB PRIORITÁS)

**Fájl:** `src/training/collector.py`
**Érintett osztályok/metódusok:** `_HandAccumulator`, `_update_preflop_context()`

**Gyökérok:** Az `_update_preflop_context()` az egész `betting_history`-t újra
vizsgálta minden lépésnél. Ha a kézben 10 preflop akció volt és az 5. lépésnél
hívódott, 4 emelést adott hozzá. A 6. lépésnél 5-öt, stb. — szuperlineáris
növekedés. Eredmény: `preflop_raises_total >> valós emelések száma`, a 3-Bet%
~80-100%-ra inflálódott, az Orchestrator folyamatosan "maniac"-ot detektált, és
blöff büntetést alkalmazott az ágensre.

**Fix:** `_HandAccumulator`-ba hozzáadva `_last_seen_history_len: int = 0` mező.
`_update_preflop_context()` mostantól csak az új, utoljára feldolgozás óta
hozzáadott elemeket vizsgálja (`history[acc._last_seen_history_len:]`).
Mûveleti igény: O(delta) helyett O(N).

**Javított fájl:** `fixes/src/training/collector.py`

---

### C1 — Bootstrap Érték Timing Race Condition

**Fájlak:** `src/training/buffer.py`, `src/training/collector.py`, `src/training/runner.py`

**Gyökérok:** A `RolloutBuffer`-nek nem volt `set_last_value()` metódusa.
A `collector.collect_rollout()` `if hasattr(self.buffer, "set_last_value")` ága
sosem futott le. A `runner.py` ezután `collector.get_last_bootstrap_value(network)`-öt
hívott meg — de ekkor a collector `_current_obs`-a már potenciálisan egy lépéssel
előre lépett (ha a rollout terminális állapotban végződött, a reset azonnal
felülírta). Eredmény: biased advantage estimates minden rollout határon.

**Fix (3 fájl):**
1. `buffer.py`: `set_last_value()` és `get_last_bootstrap_value()` metódusok hozzáadva.
   `reset()`-ben `_last_bootstrap_value = 0.0`.
2. `collector.py`: A `collect_rollout()` végén, a loop után, de return előtt:
   `self.buffer.set_last_value(last_value_tensor.squeeze(-1))` (truncated esetén)
   vagy `self.buffer.set_last_value(0.0)` (terminális esetén).
3. `runner.py`: `_run_single_iteration()`-ban:
   - ELTÁVOLÍTVA: `last_value = self.collector.get_last_bootstrap_value(self.network)`
   - HOZZÁADVA: `self.buffer.compute_gae(last_value=self.buffer.get_last_bootstrap_value())`

**Javított fájlak:** `fixes/src/training/buffer.py`, `fixes/src/training/collector.py`,
`fixes/src/training/runner.py`

---

### C3 — O(N) Stagnáció Ellenőrzés (GPU Starvation)

**Fájl:** `src/orchestrator/telemetry.py`
**Érintett metódus:** `check_stagnation()`

**Gyökérok:** `itertools.islice` egy `deque`-n O(N) — az elejéről kell iterálni.
`window_size=100_000` és `eval_interval=50` mellett ez 200×-ot fut le egy 12 órás
session alatt, alkalmanként 100,000 elemet iterálva — ~100ms CPU spike, GPU starvation.

**Fix:** `TelemetryAnalyzer.__init__()`-ban hozzáadva:
`self._recent_deque: deque[float] = deque(maxlen=50)`
`record_hand()` inkrementálisan appandeli: `self._recent_deque.append(record.reward_bb)`
`check_stagnation()` az O(1) `_reward_sum / len(window)` átlagot hasonlítja a
O(1) `sum(_recent_deque) / len(_recent_deque)` recent átlaghoz.
Teljes mûveletigény: O(1) volt O(N) helyett.

**Javított fájl:** `fixes/src/orchestrator/telemetry.py`

---

### C2 — DDP FSP Snapshot Deadlock

**Fájl:** `src/orchestrator/orchestrator.py`
**Érintett metódus:** `_save_fsp_snapshot()`

**Gyökérok:** A Rank 0 FSP snapshot mentése (`torch.save()`) DDP barrier nélkül
futott. Multi-GPU futásban Rank 0 blokkolódhatott a fájlrendszer I/O-n, miközben
a többi rank tovább számolt — potenciális deadlock.

**Fix:** `_save_fsp_snapshot()` mostantól:
1. `dist.barrier()` ELŐTTE (minden rank megvárja az előző gradiens frissítést)
2. Atomikus mentés (temp fájl + `os.replace()`, M2 mintára)
3. `dist.barrier()` UTÁNA (minden rank megvárja a mentés végét)
4. `set_ddp_world_size(world_size)` új metódus — `train_local.py` beállítja.

**Javított fájl:** `fixes/src/orchestrator/orchestrator.py`

---

## 🟠 MAGAS (H1–H5)

---

### H1 — Korlátlan Log-Skála Chip Normalizáció

**Fájl:** `src/env/features.py`
**Érintett metódus:** `_encode_env_metrics()` → `_normalize_chips()`

**Gyökérok:** A log-skála normalizáció korlátlan kimenetet adott: 3× initial stack
→ ≈2.1, 5× stack → ≈2.6. A hálózat `[0, ∞)` bemenetet látott, az ortogonális
súlyinicializáció hatástalanná vált mély stack helyzetekben.

**Fix:** Egyszerű clip normalizáció:
```
capped = min(value, initial_stack * 5.0)
normalized = capped / (initial_stack * 5.0)
```
Kimenet garantáltan `[0, 1]`. A `_CHIP_NORMALIZATION_MAX_MULTIPLIER = 5.0` konstans
konfigurálható. 5× initial stack lefedi a tipikus játékhelyzeteket.

**Javított fájl:** `fixes/src/env/features.py`

---

### H4 — Scheduler Állapot Nem Mentett

**Fájl:** `scripts/train_local.py`

**Gyökérok:** `on_checkpoint()` nem adta át `scheduler=runner.trainer.scheduler`-t.
Ha `learning_rate_schedule: "linear"` be volt állítva, a `LambdaLR` scheduler
sosem került checkpoint-ba. Kaggle session resumekor az LR warmup nulláról indult.

**Fix:** `scripts/train_local_PATCH.md` részletes diff-ekkel. Két helyen módosítandó:
1. `on_checkpoint()`: `scheduler=runner.trainer.scheduler` paraméter hozzáadása
2. Resume blokkban: scheduler restore (`runner.trainer.scheduler.load_state_dict(...)`)

---

### H5 — `start_time` Nem Adódik Át `train_local.py`-ban

**Fájl:** `scripts/train_local.py`

**Gyökérok:** `main()` nem adta át `start_time`-t a `build_training_pipeline()`-nak.
A `GracefulShutdownMonitor` az inicializálása pillanatában indította el az órát —
de a pipeline build (checkpoint letöltés, stb.) perceket vehet igénybe, ami
észrevétlenül elveszett a 11.5 órás limitből.

**Fix:** `_session_start = time.monotonic()` a `build_training_pipeline()` hívás
ELŐTT, majd `start_time=_session_start` átadása. Részlet: `train_local_PATCH.md`.

---

## 🟡 KÖZEPES (M2, M4, M5)

---

### M2 — Nem-Atomikus FSP Snapshot Írás

**Fájl:** `src/training/opponent_pool.py`

**Fix:** `add_snapshot()`-ban `torch.save(...)` cseréje temp fájl + `os.replace()`
atomikus mintára. Részlet: `fixes/src/training/M2_M4_patch.md`.

---

### M4 — UCB1 Off-by-One (Első Kiválasztás)

**Fájl:** `src/orchestrator/curriculum.py`

**Fix:** `select_opponent()`-ban `effective_rounds = max(total_selections_cached, 1)`.
Részlet: `fixes/src/training/M2_M4_patch.md`.

---

### M5 — Entropia CAP Tévesen WARNING Szinten Logolt

**Fájl:** `src/orchestrator/orchestrator.py`

**Fix:** `_intervene_passivity()` és `_intervene_stagnation()`-ban a CAP elérési
log `logger.warning` → `logger.debug`. Ez normális működést jelöl, nem hibát.
Implementálva: `fixes/src/orchestrator/orchestrator.py`

---

## 🔵 ALACSONY (L3, L4, L5)

---

### L5 — Hiányzó `AcpcClient` Osztálydeklaráció (🔴 Import SyntaxError!)

**Fájl:** `src/evaluation/acpc_client.py`

**Gyökérok:** A `class AcpcClient:` sor hiányzott. Python az osztálytörzset
`HandResult`-ban string literálként értelmezte. Az import NameError-t dobott,
az egész `src/evaluation` modul megtörve.

**Fix:** Teljes `AcpcClient` osztálydeklaráció hozzáadva helyes indentálással.
**Javított fájl:** `fixes/src/evaluation/acpc_client.py`

---

### L4 — `trigger_manual_upload()` Mindig "Sikertelen"-t Logolt

**Fájl:** `src/mlops/hf_sync.py`

**Fix:** `_do_upload()` explicit `True`-t ad vissza, hogy `retry_with_backoff()`
helyesen különböztesse meg a sikert a valódi hibától.
**Patch:** `fixes/src/mlops/hf_sync_L4_patch.py`

---

### L3 — `_detect_street()` Nem Figyelmeztet Váratlan Lapszámra

**Fájl:** `src/training/collector.py`

**Fix:** 1 vagy 2 közös lap esetén `logger.warning()` hozzáadva, de a fallback
marad river (3). Implementálva: `fixes/src/training/collector.py`.

---

## Teljes Fájl Inventár (Output Könyvtár)

```
fixes/
├── src/
│   ├── training/
│   │   ├── buffer.py          ← C1 teljes fix
│   │   ├── collector.py       ← C4 + C1 + L3 teljes fix
│   │   ├── runner.py          ← C1 teljes fix
│   │   └── M2_M4_patch.md     ← M2 + M4 diff útmutató
│   ├── orchestrator/
│   │   ├── telemetry.py       ← C3 teljes fix
│   │   └── orchestrator.py    ← C2 + M5 teljes fix
│   ├── env/
│   │   └── features.py        ← H1 teljes fix
│   ├── mlops/
│   │   └── hf_sync_L4_patch.py ← L4 patch
│   └── evaluation/
│       └── acpc_client.py     ← L5 teljes fix (hiányzó osztálydeklaráció)
└── scripts/
    └── train_local_PATCH.md   ← H4 + H5 + C2 diff útmutató
```

## Alkalmazási Sorrend (Ajánlott)

1. `collector.py` (C4 — legkritikusabb: telemetria korrupcióját szünteti meg)
2. `buffer.py` (C1 — GAE bootstrap timing)
3. `runner.py` (C1 — bootstrap olvasás a bufferből)
4. `acpc_client.py` (L5 — import hiba, gyors fix)
5. `telemetry.py` (C3 — GPU starvation megelőzés)
6. `orchestrator.py` (C2 + M5 — DDP deadlock + log javítás)
7. `features.py` (H1 — chip normalizáció)
8. `train_local.py` diff alkalmazása (H4 + H5 + C2)
9. `hf_sync.py` patch (L4)
10. `opponent_pool.py` M2 patch
11. `curriculum.py` M4 patch
