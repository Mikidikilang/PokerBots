# MASTER NOTE — PokerAI-NLHE Fejlesztési Biblia

> **Dokumentum típus:** Belső fejlesztői referencia (Development Bible)
> **Projekt:** No-Limit Texas Hold'em RL AI with Auto-Adaptive Curriculum Learning
> **Verzió:** 0.1.0 (Initial Implementation)
> **Utolsó frissítés:** 2025-03-26
> **Státusz:** Fázis 1–9 implementálva, 159/159 teszt zöld

---

## Tartalomjegyzék

1. [Projekt Áttekintés](#1-projekt-áttekintés)
2. [Architektúra Döntések és Indoklások](#2-architektúra-döntések-és-indoklások)
3. [Modul Szerződések (API Contracts)](#3-modul-szerződések)
4. [Adatfolyam (Data Flow)](#4-adatfolyam)
5. [Kódolási Konvenciók](#5-kódolási-konvenciók)
6. [Konfigurációs Rendszer](#6-konfigurációs-rendszer)
7. [Tesztelési Stratégia](#7-tesztelési-stratégia)
8. [Deployment és Üzemeltetés](#8-deployment-és-üzemeltetés)
9. [Ismert Limitációk és Technikai Adósság](#9-ismert-limitációk-és-technikai-adósság)
10. [Jövőbeli Fejlesztések](#10-jövőbeli-fejlesztések)
11. [Fájl Inventár](#11-fájl-inventár)
12. [Függőségek és Verziók](#12-függőségek-és-verziók)
13. [Hibaelhárítási Útmutató](#13-hibaelhárítási-útmutató)
14. [Változásnapló](#14-változásnapló)

---

## 1. Projekt Áttekintés

### 1.1 Mi ez a projekt?

Egy **produkciós szintű RL AI rendszer**, amely a No-Limit Texas Hold'em (NLHE)
póker játékban képes a Nash-egyensúly közelítésére. A rendszer nem egy egyszerű
modell-tréning szkript, hanem egy teljes zárt láncú (closed-loop) pipeline:

- **Tanulás**: PPO (Proximal Policy Optimization) Actor-Critic architektúra
- **Felügyelet**: Auto-Adaptive Curriculum Orchestrator állapotgéppel
- **Infrastruktúra**: Megszakítástűrő Kaggle → HF Hub MLOps csővezeték

### 1.2 Miért nem elég az egyszerű self-play?

A naiv self-play a pókerben patológiás stratégiákhoz vezet:
- **Passzivitás (Folding Spam)**: A hálózat megtanulja, hogy a dobás a "biztonságos" opció
- **All-in Spam (Maniac)**: A hálózat rájön, hogy a folyamatos maximális agresszió kihasználja a gyenge ellenfeleket
- **Ciklikus degeneráció**: Kő-papír-olló dinamika, katasztrofális felejtés

Az Orchestrator ezeket a patológiákat **automatikusan detektálja** (HUD metrikákon keresztül)
és **beavatkozik** (reward shaping, entrópia injekció, ellenfél-pool rotálás).

### 1.3 Célplatform

| Környezet | Hardver | Limit | Stratégia |
|---|---|---|---|
| **Kaggle** (elsődleges) | P100/T4 GPU, 16GB RAM | 12h session, 20GB diszk | GracefulShutdown 11.5h, HF Hub sync |
| **Lokális** | Bármilyen GPU/CPU | Nincs limit | CLI (train_local.py) |
| **Cloud** (jövő) | Multi-GPU | Költségalapú | DistributedDataParallel |

### 1.4 Számok

| Metrika | Érték |
|---|---|
| Produkciós Python kód | 7,356 sor (21 fájl) |
| Teszt kód | 1,812 sor (13 fájl) |
| Szkriptek | 526 sor + 1 notebook |
| Konfiguráció | 542 sor (YAML + TOML) |
| Összes | ~10,600 sor, 43 fájl |
| Tesztek | 159/159 PASSED |
| Becsült hálózati paraméterek | ~600,000 (6-Max) |
| Observation Space dimenzió | 281 (6-Max) |
| Akciótér | 9 diszkrét akció |

---

## 2. Architektúra Döntések és Indoklások

### 2.1 Miért PPO és nem Deep CFR?

**Döntés:** PPO (on-policy, policy gradient) az elsődleges algoritmus.

**Indoklás:**
- A PPO egyszerűbb implementálni és debugolni, mint a Deep CFR
- A PPO natívan támogatja a folyamatos tanulást (nincs szükség teljes traversal-re)
- Az Orchestrator beavatkozásai (entrópia, reward shaping) közvetlenül integrálhatók
- A Deep CFR alternatívaként a jövőben implementálható (a moduláris architektúra lehetővé teszi)

**Trade-off:** A PPO on-policy jellege miatt mintavételezési hatékonysága alacsonyabb, mint az off-policy módszereké. Ezt a rollout méret növelésével kompenzáljuk.

### 2.2 Miért diszkretizált akciótér?

**Döntés:** 9 fix akció (Fold, Check/Call, Min-Raise, 0.5x/0.75x/1.0x/1.5x/2.0x Pot, All-in)

**Indoklás:**
- A Libratus és Pluribus is diszkretizált akcióteret használt
- A folytonos akciótér instabil konvergenciát eredményez a PPO-val
- Az emberi játékosok is "bucket"-ekben gondolkodnak
- A Softmax + Action Masking egyszerűen implementálható és matematikailag stabil

**Trade-off:** Elveszítünk néhány finomhangolt bet sizing-ot (pl. 33% pot block bet), de az akciók lefedik a stratégiailag releváns tartományt.

### 2.3 Miért Dict observation és nem flat vektor?

**Döntés:** A hálózat Dict[str, Tensor] bemenetet kap, elkülönített embedding rétegekkel.

**Indoklás:**
- A kártyák (bináris, ritka), a metrikák (folytonos, normalizált), és a történelem (szekvenciális)
  fundamentálisan különböző jellegű adatok
- Az elkülönített embedding rétegek lehetővé teszik, hogy minden csatorna a saját
  reprezentációját tanulja meg
- A flat vektor esetén a hálózatnak magának kellene megtanulnia a csatornák határait

**Dimenzió bontás (6-Max):**
- Hole cards: 52 → CardEmbedding → 64
- Community cards: 52 → CardEmbedding → 64
- Env metrics (9) + Position (6) = 15 → ContextEmbedding → 32
- Betting history (18×9 = 162) → HistoryEmbedding → 128→64
- **Trunk: 64+64+32+64 = 224 dim**

### 2.4 Miért nincs megosztott törzs (shared trunk) az Actor és Critic között?

**Döntés:** Az Actor és Critic fejek közvetlenül a beágyazott vektorból dolgoznak,
nincs közös rejtett réteg a fejek előtt.

**Indoklás:**
- A póker-specifikus magas variancia miatt a shared trunk instabilitást okozhat
- A value function (Critic) és a policy (Actor) különböző optimális reprezentációt igényelhet
- Az empirikus tapasztalat azt mutatja, hogy a separate heads stabilabb konvergenciát ad
  tökéletlen információs játékokban

**Trade-off:** Több paraméter (~2x a fejekben), de stabilabb tanulás.

### 2.5 Miért Singleton az Orchestrator?

**Döntés:** Az `AutoAdaptiveOrchestrator` a Singleton mintát követi.

**Indoklás:**
- A rendszerben pontosan egy felügyeleti entitás létezik
- A callback rendszeren keresztül bármelyik komponens eléri
- Az állapot (curriculum fázis, MAB eloszlások) konzisztens marad
- A hot-reload egyetlen config fájlt figyel

**Kockázat:** Teszteléskor a `reset_instance()` metódussal kell törölni az állapotot.
Ez megvan implementálva és a tesztek használják.

### 2.6 Miért time.monotonic() és nem time.time()?

**Döntés:** A GracefulShutdownMonitor `time.monotonic()`-ot használ.

**Indoklás:**
- A `time.time()` az NTP szinkronizáció miatt ugrálhat (visszafelé is!)
- A Kaggle konténerekben az óra-szinkronizáció megbízhatatlan
- A `time.monotonic()` garantáltan monoton növekvő
- A 30 perces biztonsági puffer kritikus — egy óra-ugrás katasztrofális lenne

---

## 3. Modul Szerződések

### 3.1 src/env/ — Állapottér és Környezet

| Fájl | Osztály | Bemenet | Kimenet | Felelősség |
|---|---|---|---|---|
| `features.py` | `ObservationBuilder` | `dict[str, Any]` nyers játékállapot | `dict[str, Tensor]` observation | Multi-hot kártya kódolás, BB-normalizált metrikák, zero-padded history, one-hot pozíció |
| `action_mapper.py` | `ActionMapper` | `PokerAction` enum + `GameContext` | `ResolvedAction` (akció, chip összeg, leírás) | 0-8 index → szemantikai akció, pot-relatív bet sizing, stack capping, illegális akció maszkolás (-1e8) |
| `equity.py` | `EquityCalculator` | Kártya listák | `float` [0.0, 1.0] equity | Monte Carlo szimuláció (Treys vagy fallback), pre-flop kézkategorizálás |

**Kritikus szabály:** Az `ObservationBuilder.build()` MINDIG 6 kulcsot ad vissza:
`hole_cards`, `community_cards`, `env_metrics`, `betting_history`, `position`, `action_mask`.
Ha bármelyik hiányzik, a hálózat forward pass-ja crash-el.

### 3.2 src/model/ — Neurális Architektúra

| Fájl | Osztály | Bemenet | Kimenet | Felelősség |
|---|---|---|---|---|
| `networks.py` | `ActorCriticNetwork` | `dict[str, Tensor]` observation | `(Categorical, Tensor)` — (akció eloszlás, V(s) érték) | Beágyazás, Actor fej (9-dim logit → Softmax), Critic fej (skalár), akció maszkolás, súlyinicializáció |
| `networks.py` | `NetworkConfig` | YAML dict | Dataclass | `from_dict()` factory, `trunk_input_dim` property |

**Kritikus szabály:** A `forward()` metódus a logitokhoz hozzáadja az akció maszkot
(`-1e8 * (1 - mask)`) **a Softmax ELŐTT**. Ez garantálja, hogy az illegális akciók
valószínűsége algoritmikusan nulla, miközben a backward pass stabil marad.

**Inicializáció konvenció:**
- Actor kimeneti réteg: `orthogonal_(gain=0.01)` — egyenletes kezdeti policy
- Critic kimeneti réteg: `orthogonal_(gain=1.0)` — normál értéktartomány
- Minden egyéb Linear: `orthogonal_(gain=1.0)` — stabil gradiens áramlás
- Minden bias: `zeros_()` — nulla kezdőérték

### 3.3 src/training/ — RL Optimalizáció

| Fájl | Osztály | Felelősség |
|---|---|---|
| `buffer.py` | `RolloutBuffer` | On-policy tapasztalat tárolás, GAE (reverse sweep), advantage normalizálás (zero-mean, unit-var), véletlen mini-batch generálás |
| `collector.py` | `RolloutCollector` | Környezet stepping ciklus, observation building, inference mode (no_grad), GAE bootstrap utolsó értékkel |
| `trainer.py` | `PPOTrainer` | Clipped policy loss, value loss (opcionális clip), entropy bonus, Adam optimizer, gradient clipping, KL-alapú korai leállítás, **hot-reload** metódusok |
| `opponent_pool.py` | `OpponentPool` | 4 statikus archetípus (CallingStation, Maniac, Random, TightPassive), FSP snapshot pool FIFO rotációval, unified interface |
| `runner.py` | `TrainingRunner` | Event-driven fő ciklus, callback rendszer (on_iteration_end, on_eval_step, on_checkpoint), időkorlát monitoring, emergency save |

**PPO Loss formula:**
```
L_TOTAL = -L_CLIP + c1 * L_VF - c2 * H(π)

ahol:
  L_CLIP = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]
  L_VF   = 0.5 · E[(V(s) - R)²]
  H(π)   = entropy bonus (felfedezési ösztönző)
  c1     = value_loss_coefficient (0.5)
  c2     = entropy_coefficient (0.01) — HOT-RELOADABLE
  ε      = clip_epsilon (0.2)
```

**GAE formula:**
```
δ_t = r_t + γ · V(s_{t+1}) · (1-done) - V(s_t)
A_t = Σ_{l=0}^{T-t} (γ·λ)^l · δ_{t+l}
R_t = A_t + V(s_t)
```

### 3.4 src/orchestrator/ — Curriculum Vezérlés

| Fájl | Osztály | Felelősség |
|---|---|---|
| `telemetry.py` | `TelemetryAnalyzer` | 6 HUD metrika mozgóablakos számítása (VPIP, PFR, 3-Bet, AF, WTSD, win_rate), O(1) frissítés deque-vel, anomália detektálás GTO mátrix alapján, stagnáció detektálás, GTO távolság számítás |
| `curriculum.py` | `CurriculumManager` | 3-fázisú curriculum (Phase 0→1→2), feltételes átmenetek, UCB1 MAB ellenfél-kiválasztás, állapot mentés/betöltés |
| `reward_shaper.py` | `RewardShaper` | Blöff büntetés (`λ · bluff_intensity · I(lost_showdown)`), passzivitás bonus, hot-reloadable paraméterek |
| `orchestrator.py` | `AutoAdaptiveOrchestrator` | Singleton állapotgép, event-driven callback, beavatkozási logika (passzivitás→entrópia boost, maniac→blöff penalty, stagnáció→exploration), config hot-reload (YAML mtime watcher) |

**Curriculum fázisok és átmenetek:**
```
Phase 0 (Static Bots)
    │  Feltétel: win_rate ≥ 50 mbb/h AND hands ≥ 100k
    ▼
Phase 1 (SFT Opponents)
    │  Feltétel: win_rate ≥ 30 mbb/h AND hands ≥ 200k AND exploit < 1%
    ▼
Phase 2 (Fictitious Self-Play + MAB)
    │  Cél: Nash Distance < 0.3% pot
    ▼
  [VÉGSŐ]
```

**UCB1 algoritmus:**
```
UCB(i) = avg_reward(i) + c · √(ln(N) / n_i)

ahol:
  avg_reward(i) = az i-edik ellenfél elleni átlagos jutalom
  N             = összes kiválasztás
  n_i           = az i-edik ellenfél kiválasztásainak száma
  c             = exploration factor (alapértelmezett: 2.0)
```

**Beavatkozási mátrix:**

| Patológia | Trigger | Primer beavatkozás | Szekunder beavatkozás |
|---|---|---|---|
| Passzivitás | VPIP < 16% VAGY gap > 10% (6-Max) | Entrópia koefficienshez × boost_factor | Agresszió bonus aktiválás |
| All-in Spam | PFR > 28% VAGY 3-Bet > 15% VAGY AF > 4.0 (6-Max) | Blöff penalty lambda aktiválás | Calling Station botok berotálása |
| Stagnáció | |Δreward| < threshold ablakban | Entrópia boost | (Rollout scaling — jövő) |

### 3.5 src/mlops/ — Infrastruktúra

| Fájl | Osztály | Felelősség |
|---|---|---|
| `state_manager.py` | `RNGStateManager` | 5-komponensű RNG capture/restore (Python, NumPy, Torch CPU, CUDA, DataLoader), `set_global_seed()` cold start |
| `state_manager.py` | `CheckpointManager` | Egységes checkpoint csomag (model + optimizer + RNG + orchestrator + meta), FIFO rotáció, `restore_full_state()` |
| `hf_sync.py` | `AsyncModelUploader` | CommitScheduler wrapper (15 perces háttérszálas polling), GIL-mentes I/O |
| `hf_sync.py` | `HuggingFaceStateManager` | snapshot_download / upload_folder, symlink letiltás Kaggle-hez |
| `hf_sync.py` | `configure_headless_auth()` | 3-szintű token feloldás: közvetlen → env var → Kaggle Secrets |
| `fault_tolerance.py` | `GracefulShutdownMonitor` | time.monotonic() prediktív időzítő (11.5h), SIGTERM/SIGINT kezelők |
| `fault_tolerance.py` | `FaultHandler` | NaN loss rollback (retry/abort), OOM batch csökkentés, error log |

**Checkpoint csomag tartalma:**
```python
{
    "model_state_dict":     network.state_dict(),
    "optimizer_state_dict":  optimizer.state_dict(),
    "rng_states": {
        "python_stdlib":  random.getstate(),
        "numpy":          np.random.get_state(),
        "torch_cpu":      torch.get_rng_state(),
        "torch_cuda":     torch.cuda.get_rng_state_all(),
        "dataloader":     generator.get_state(),
    },
    "orchestrator_state": {
        "current_phase":    int,
        "phase_history":    list,
        "ucb_arms":         dict,
        "total_selections": int,
    },
    "training_meta": {
        "total_steps":     int,
        "total_episodes":  int,
    },
    "iteration":          int,
    "config_snapshot":    dict,   # A mentéskori teljes YAML
}
```

---

## 4. Adatfolyam

### 4.1 Egyetlen Training Iteráció

```
┌──────────────────────────────────────────────────────────────┐
│                    TrainingRunner.run()                        │
│                                                               │
│  1. ADATGYŰJTÉS (collector.collect_rollout)                   │
│     ┌─────────────────────────────────────────────────┐       │
│     │ for step in range(rollout_steps):                │       │
│     │   obs_dict = obs_builder.build(raw_state)        │       │
│     │   action, lp, ent, val = network(obs_dict)       │       │
│     │   next_state, reward, done, info = env.step(act) │       │
│     │   buffer.add(obs, action, reward, lp, val, done) │       │
│     └─────────────────────────────────────────────────┘       │
│     buffer.compute_gae(last_value)                            │
│                                                               │
│  2. GRADIENS FRISSÍTÉS (trainer.train_on_buffer)              │
│     ┌─────────────────────────────────────────────────┐       │
│     │ for epoch in range(num_epochs):                  │       │
│     │   for batch in buffer.get_mini_batches():        │       │
│     │     _, new_lp, ent, new_val = network(obs, act)  │       │
│     │     ratio = exp(new_lp - old_lp)                 │       │
│     │     policy_loss = -min(ratio*A, clip(ratio)*A)   │       │
│     │     value_loss = 0.5 * (V - R)²                  │       │
│     │     total_loss = pl + c1*vl - c2*H               │       │
│     │     optimizer.step()                              │       │
│     └─────────────────────────────────────────────────┘       │
│                                                               │
│  3. ORCHESTRATOR CALLBACK (on_iteration_end)                  │
│     ┌─────────────────────────────────────────────────┐       │
│     │ orchestrator.on_iteration_callback(iter, stats)  │       │
│     │   → hot-reload check (YAML mtime)                │       │
│     │   → telemetry.get_current_metrics()              │       │
│     │   → telemetry.detect_anomalies(gto, thresholds)  │       │
│     │   → _execute_interventions() ha anomália          │       │
│     │   → curriculum.check_phase_transition()           │       │
│     └─────────────────────────────────────────────────┘       │
│                                                               │
│  4. CHECKPOINT (ha save_interval elérve)                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ rng_states = RNGStateManager.capture_states()    │       │
│     │ ckpt_manager.save(network, optimizer, rng, orch) │       │
│     │ → AsyncModelUploader háttérben szinkronizál      │       │
│     └─────────────────────────────────────────────────┘       │
│                                                               │
│  5. SHUTDOWN CHECK                                            │
│     if shutdown_monitor.should_shutdown(): break              │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Kaggle Session Életciklus

```
[Session Start]
  │
  ├─ 1. git clone (GitHub → Kaggle /working)
  ├─ 2. pip install -e . (szerkeszthető telepítés)
  ├─ 3. HF headless auth (Kaggle Secrets → os.environ)
  ├─ 4. snapshot_download (HF Hub → lokális checkpoint)
  ├─ 5. restore_full_state (model + optimizer + RNG + orchestrator)
  │
  ├─ 6. Training Loop ─────────────────────────────────────────
  │     │                                                       │
  │     │  Háttérben: CommitScheduler (15 percenként sync)      │
  │     │  Háttérben: GracefulShutdownMonitor (monotonic clock) │
  │     │                                                       │
  │     ├─ Iteráció 1...N                                       │
  │     │   ├─ collect_rollout → GAE                            │
  │     │   ├─ train_on_buffer → PPO update                     │
  │     │   ├─ orchestrator callback                            │
  │     │   └─ checkpoint (ha interval elérve)                  │
  │     │                                                       │
  │     └─ 11.5 óra: should_shutdown() → True                  │
  │                                                             │
  ├─ 7. Graceful Shutdown                                       │
  │     ├─ Végső checkpoint mentés                              │
  │     ├─ CommitScheduler.trigger() (utolsó upload)            │
  │     └─ upload_folder (szinkron fallback)                    │
  │                                                             │
  └─ 8. sys.exit(0) (tiszta kilépés, nem 137/SIGKILL)
```

---

## 5. Kódolási Konvenciók

### 5.1 Python Stílus

- **Python 3.10+** kompatibilitás (union types: `X | None`, match statements nincs)
- **PEP 484 Type Hints** minden publikus metóduson és osztályon
- **`from __future__ import annotations`** minden fájl tetején (lazy evaluation)
- **Docstring:** Google stílus, minden publikus osztályon és metóduson
- **Line length:** 120 karakter (ruff beállítás)
- **Import sorrend:** stdlib → third-party → local (ruff isort)

### 5.2 Logging Konvenció

Minden modul a saját loggerét használja:
```python
import logging
logger = logging.getLogger(__name__)
```

**Szintek használata:**
| Szint | Mikor |
|---|---|
| `DEBUG` | Tenzor dimenziók, belső állapotváltozások, iterációnkénti részletek |
| `INFO` | Inicializálás, fázisátmenetek, checkpoint mentés/betöltés, összefoglaló statisztikák |
| `WARNING` | Anomália detektálás, graceful shutdown figyelmeztetés, fallback aktiválás |
| `ERROR` | NaN loss, OOM, fájl I/O hiba, érvénytelen konfiguráció |

**Minta:**
```python
logger.debug("Latens vektor: shape=%s, norm=%.4f", latent.shape, latent.norm().item())
logger.info("Checkpoint mentve: %s (%.2f MB, iter=%d)", filepath, size_mb, iteration)
logger.warning("PATOLOGIA: Passzivitas! VPIP=%.1f%% < %.1f%%", vpip, threshold)
logger.error("KRITIKUS: NaN loss detektalva! pl=%.4f, vl=%.4f", p_loss, v_loss)
```

### 5.3 Naming Konvenciók

| Elem | Konvenció | Példa |
|---|---|---|
| Osztályok | PascalCase | `ActorCriticNetwork`, `RolloutBuffer` |
| Metódusok/Függvények | snake_case | `compute_gae`, `get_current_metrics` |
| Privát metódusok | _prefix | `_apply_action_mask`, `_check_hot_reload` |
| Konstansok | UPPER_SNAKE | `DECK_SIZE`, `ILLEGAL_ACTION_LOGIT` |
| Config dataclassok | PascalCase + "Config" suffix | `NetworkConfig`, `TrainerConfig` |
| Fájlok | snake_case | `action_mapper.py`, `state_manager.py` |

### 5.4 Config from_dict() Konvenció

Minden konfigurációs dataclass rendelkezik `from_dict(cfg)` osztálymetódussal,
amely a teljes YAML szótárat fogadja (nem egy részt):

```python
@classmethod
def from_dict(cls, cfg: dict[str, Any]) -> TrainerConfig:
    ppo = cfg.get("ppo", {})
    return cls(
        learning_rate=ppo.get("learning_rate", 3e-4),
        ...
    )
```

Ez biztosítja, hogy a hívó oldalon mindig `Config.from_dict(full_yaml)` a minta,
nem kell tudni melyik alkulcs kell.

### 5.5 Error Handling Konvenció

- **ValueError**: Érvénytelen bemeneti adatok (rossz kártyaformátum, tartományon kívüli érték)
- **KeyError**: Hiányzó kötelező kulcs a nyers állapotban
- **RuntimeError**: Előfeltétel nem teljesül (pl. `compute_gae()` nem lett meghívva)
- **FileNotFoundError**: Hiányzó konfig vagy checkpoint fájl
- **Soha nem csendes hiba**: Minden catch-elt kivétel logolva van (`logger.error`)

---

## 6. Konfigurációs Rendszer

### 6.1 Single Source of Truth

A `config.yaml` az **EGYETLEN** hely ahol hiperparaméterek definiálva vannak.
A kódban nincs hardcode-olt hiperparaméter. A `from_dict()` metódusok
alapértelmezett értékeket használnak fallback-ként, de ezeknek meg kell
egyezniük a YAML-ben definiált értékekkel.

### 6.2 Hot-Reloadable Paraméterek

Ezek a paraméterek futásidőben módosíthatók a YAML fájl szerkesztésével:

| Paraméter | YAML útvonal | Hatás |
|---|---|---|
| Tanulási ráta | `ppo.learning_rate` | Adam optimizer param_groups frissítés |
| Entrópia koefficiensz | `ppo.entropy_coefficient` | PPO loss entrópia tag súlya |
| Blöff penalty lambda | `reward_shaping.bluff_penalty_lambda` | All-in Spam büntetés erőssége |
| Agresszió bonus | `reward_shaping.preflop_aggression_bonus` | Passzivitás elleni jutalom |

**Mechanizmus:** Az `orchestrator.py` `_check_hot_reload()` metódusa
a config fájl `mtime`-ját figyeli `hot_reload_interval_sec` másodpercenként.
Ha változott, beolvassa az új YAML-t és meghívja az érintett update metódusokat.

### 6.3 GTO Mátrix Kulcsok

A `gto_matrix` és `degeneration_thresholds` szekciók **string kulcsokat** használnak
az asztalmérethez (`"2"`, `"6"`, `"9"`), nem int-et. Ez a YAML parszolás sajátossága.

```python
table_key = str(config.num_players)  # "6"
gto = yaml_config["gto_matrix"][table_key]
```

---

## 7. Tesztelési Stratégia

### 7.1 Teszt Architektúra

```
tests/
├── conftest.py              # Torch mock + fixture-ök (MINDEN teszt használja)
├── test_env/test_env.py     # 38 teszt — ObservationBuilder, ActionMapper, Equity
├── test_model/test_model.py # 17 teszt — NetworkConfig, dimenziók, maszkolás matek
├── test_training/           # 27 teszt — Buffer/GAE, OpponentPool, Config from_dict
├── test_orchestrator/       # 34 teszt — Telemetria, Curriculum/UCB, RewardShaper
├── test_mlops/              # 29 teszt — RNG, Checkpoint, Shutdown, FaultHandler
└── test_integration/        # 14 teszt — Cross-module wiring, config konzisztencia
```

### 7.2 Torch Mock Rendszer

A `conftest.py` egy `FakeTensor` osztályt definiál, amely a `torch.Tensor`
API-jának minimális részét implementálja (aritmetika, shape, item, stb.).
Ez lehetővé teszi, hogy a tesztek **PyTorch telepítés nélkül is fussanak** (CI/CD).

A mock **CSAK akkor aktiválódik**, ha a valódi torch nem elérhető:
```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # ... FakeTensor setup ...
```

### 7.3 Mi NEM tesztelhető mock-kal

- A `networks.py` teljes forward pass-ja (nn.Module, autograd)
- A `trainer.py` tényleges gradiens számítása (backward, optimizer.step)
- A `collector.py` valódi környezettel való interakciója
- A `hf_sync.py` CommitScheduler háttérszálas működése

Ezek **integrációs tesztek** valódi PyTorch környezetben szükségesek
(jelölés: `@pytest.mark.gpu` vagy `@pytest.mark.integration`).

### 7.4 Tesztelési Elvek

1. **Minden `from_dict()` tesztelve van** a valódi `config.yaml`-lel
2. **Minden anomália detektálás tesztelve van** szintetikus adatokkal
3. **Determinisztikus RNG resume** igazolva: capture → változás → restore → azonos szekvencia
4. **Checkpoint roundtrip**: save → load → restore → azonos állapot
5. **Edge case-ek**: üres buffer, 0 legális akció, NaN retry limit, 0 óra timeout

---

## 8. Deployment és Üzemeltetés

### 8.1 Első Futtatás (Cold Start)

```bash
# 1. Repó klónozás
git clone https://github.com/your-username/PokerAI-Project.git
cd PokerAI-Project

# 2. Telepítés
pip install -e ".[dev]"

# 3. Tesztek futtatása
pytest tests/ -v

# 4. Első training (CPU, rövid)
python -m scripts.train_local --config config.yaml --max-iter 10 --log-level DEBUG

# 5. GPU training
python -m scripts.train_local --config config.yaml --device cuda
```

### 8.2 Kaggle Secrets Beállítása

1. **GITHUB_TOKEN**: Settings → Developer → Personal Access Tokens → Fine-grained
   - Scope: `repo` (read-only elegendő)
2. **HF_TOKEN**: huggingface.co → Settings → Access Tokens → New Token
   - Scope: `write` (kötelező az upload-hoz)
3. Kaggle notebook: Add-ons → Secrets → mindkét token hozzáadása

### 8.3 Resume Training

A rendszer automatikusan resume-ol, ha talál checkpointot:

```bash
# Lokálisan: az utolsó checkpoint-ból
python -m scripts.train_local --config config.yaml --resume

# Specifikus checkpoint-ból
python -m scripts.train_local --config config.yaml --resume --checkpoint checkpoints/checkpoint_iter_00005000.pt
```

Kaggle-on a notebook minden session-ben `resume=True`-val fut, és a HF Hub-ról
tölti le az előző session checkpointját.

### 8.4 Monitoring

A `logs/training.log` fájlba DEBUG szintű naplózás megy. A konzolra az INFO szintű.

Fontos log minták keresése:
```bash
# Fázisátmenet
grep "FAZISATMENET" logs/training.log

# Anomáliák
grep "PATOLOGIA" logs/training.log

# Beavatkozások
grep "intervenció" logs/training.log

# NaN/hiba
grep "KRITIKUS\|NaN\|OOM" logs/training.log
```

---

## 9. Ismert Limitációk és Technikai Adósság

### 9.1 Környezeti Wrapper Hiányzik

**Probléma:** A `collector.py` egy `PokerEnvironment` Protocol-t definiál,
de nincs konkrét wrapper implementáció az RLCard vagy PettingZoo motorokhoz.
A `train_local.py` közvetlenül `rlcard.make()`-kel példányosít, ami nem
teljes mértékben kompatibilis a Protocol interfészekkel.

**Megoldás:** Egy `src/env/wrappers.py` fájl, amely RLCard és PettingZoo
adaptereket implementál az egységes `PokerEnvironment` Protocol-hoz.

**Prioritás:** MAGAS — ez az első teendő a tényleges training indítás előtt.

### 9.2 A Telemetria Nem Automatizált

**Probléma:** A `TelemetryAnalyzer.record_hand()` metódust manuálisan kell
hívni a collector vagy a runner callback-jéből. Nincs automatikus konverzió
a nyers rollout adatokból `HandRecord` objektumokba.

**Megoldás:** Egy bridge komponens a collector és a telemetria között, amely
az akció indexekből és a játékállapotból automatikusan kitölti a `HandRecord` mezőit.

**Prioritás:** MAGAS — nélküle az Orchestrator nem kap valós HUD metrikákat.

### 9.3 Nincs W&B / TensorBoard Integráció

**Probléma:** A training statisztikák csak logba mennek, nincs vizuális dashboard.

**Megoldás:** W&B (wandb) vagy TensorBoard hook a runner callback-jében.

**Prioritás:** KÖZEPES — hasznos a debugging-hoz, de nem blokkolja a training-et.

### 9.4 A networks.py Docstringek ASCII

**Probléma:** A `networks.py` docstringjei ASCII karaktereket használnak az ékezetes
magyar helyett, mert a bash heredoc speciális karakterkezelése problémás volt a
generálás során.

**Megoldás:** Manuális javítás ékezetes magyar szövegre, vagy teljes angol docstringek.

**Prioritás:** ALACSONY — funkcionálisan nem befolyásol semmit.

### 9.5 Nincs Multi-GPU Támogatás

**Probléma:** A jelenlegi implementáció egyetlen GPU-ra van optimalizálva.

**Megoldás:** `torch.nn.DataParallel` vagy `DistributedDataParallel` integráció.

**Prioritás:** ALACSONY — a Kaggle P100/T4 egyetlen GPU-t biztosít.

---

## 10. Jövőbeli Fejlesztések

### 10.1 Rövid táv (a training indítás előtt)

1. **RLCard/PettingZoo Wrapper** (`src/env/wrappers.py`)
   - Az `env.reset()` → nyers állapot szótár konverzió
   - Az `env.step(action)` → akció index konverzió
   - Reward normalizáció (BB egységbe)

2. **Telemetria Bridge** (a collector és telemetria közötti híd)
   - Automatikus `HandRecord` generálás az akciókból
   - Pre-flop/post-flop akció kategorizálás

3. **Smoke Test** (valódi PyTorch-cal, 10 iteráció)
   - Teljes pipeline end-to-end teszt
   - Forward/backward pass dimenzió verifikáció
   - Checkpoint save/load roundtrip valódi súlyokkal

### 10.2 Közép táv (a training során)

4. **W&B Monitoring Dashboard**
5. **Learning Rate Scheduler** (Cosine Annealing with Warm Restarts)
6. **Opponent Pool Bővítés** (SFT ellenfelek betöltése)
7. **Slumbot API Integráció** (HUNL benchmark)

### 10.3 Hosszú táv

8. **Deep CFR Alternatív Trainer** (a moduláris architektúra lehetővé teszi)
9. **LLM-alapú Stratégiai Visszacsatolás** (a specifikációban leírt szemantikus korrektúra)
10. **Nash Distance Számítás** (LBR és ISMCTS-BR implementáció)
11. **Multi-GPU / Multi-node Training**

---

## 11. Fájl Inventár

### 11.1 Produkciós Kód (src/)

| Fájl | Sorok | Fő osztály/függvény |
|---|---|---|
| `src/__init__.py` | 17 | Csomag metaadatok |
| `src/env/__init__.py` | 18 | Exportok |
| `src/env/features.py` | 537 | `ObservationBuilder`, `ObservationConfig` |
| `src/env/action_mapper.py` | 430 | `ActionMapper`, `PokerAction`, `GameContext` |
| `src/env/equity.py` | 606 | `EquityCalculator` |
| `src/model/__init__.py` | 11 | Exportok |
| `src/model/networks.py` | 551 | `ActorCriticNetwork`, `NetworkConfig`, embeddings |
| `src/orchestrator/__init__.py` | 18 | Exportok |
| `src/orchestrator/telemetry.py` | 476 | `TelemetryAnalyzer`, `HandRecord` |
| `src/orchestrator/curriculum.py` | 468 | `CurriculumManager`, `UCBArm`, `CurriculumPhase` |
| `src/orchestrator/reward_shaper.py` | 284 | `RewardShaper`, `RewardShapingConfig` |
| `src/orchestrator/orchestrator.py` | 523 | `AutoAdaptiveOrchestrator` (Singleton) |
| `src/training/__init__.py` | 20 | Exportok |
| `src/training/buffer.py` | 374 | `RolloutBuffer`, `RolloutBufferConfig` |
| `src/training/collector.py` | 328 | `RolloutCollector`, `CollectorConfig` |
| `src/training/trainer.py` | 410 | `PPOTrainer`, `TrainerConfig` |
| `src/training/opponent_pool.py` | 463 | `OpponentPool`, 4 archetípus bot |
| `src/training/runner.py` | 462 | `TrainingRunner`, `RunnerConfig` |
| `src/mlops/__init__.py` | 29 | Exportok |
| `src/mlops/state_manager.py` | 492 | `RNGStateManager`, `CheckpointManager` |
| `src/mlops/hf_sync.py` | 416 | `AsyncModelUploader`, `HuggingFaceStateManager` |
| `src/mlops/fault_tolerance.py` | 423 | `GracefulShutdownMonitor`, `FaultHandler` |

### 11.2 Szkriptek

| Fájl | Sorok | Leírás |
|---|---|---|
| `scripts/__init__.py` | 3 | Csomag |
| `scripts/train_local.py` | 523 | CLI pipeline (argparse, logging, bootstrap, run) |
| `scripts/train_kaggle.ipynb` | 14 cella | Kaggle notebook (clone, install, auth, train, upload) |

### 11.3 Tesztek

| Fájl | Tesztek | Sorok |
|---|---|---|
| `tests/conftest.py` | — | 275 (torch mock + fixture-ök) |
| `tests/test_env/test_env.py` | 38 | 312 |
| `tests/test_model/test_model.py` | 17 | 240 |
| `tests/test_training/test_training.py` | 27 | 224 |
| `tests/test_orchestrator/test_orchestrator.py` | 34 | 349 |
| `tests/test_mlops/test_mlops.py` | 29 | 286 |
| `tests/test_integration/test_integration.py` | 14 | 126 |

### 11.4 Konfiguráció és Dokumentáció

| Fájl | Sorok | Leírás |
|---|---|---|
| `config.yaml` | 353 | Központi hiperparaméterek (Single Source of Truth) |
| `pyproject.toml` | 189 | Build rendszer, 20+ függőség, tool konfiguráció |
| `README.md` | 213 | Projekt dokumentáció, architektúra, használat |
| `ROADMAP.md` | ~170 | Fejlesztési terv (Fázis 1-9) |
| `MASTER_NOTE.md` | Ez a fájl | Fejlesztési biblia |

---

## 12. Függőségek és Verziók

### 12.1 Kötelező (Runtime)

| Csomag | Min. verzió | Szerep |
|---|---|---|
| `torch` | ≥2.1.0 | Neurális hálózat, autograd |
| `torchrl` | ≥0.4.0 | CompressedListStorage (Zstandard buffer) |
| `tensordict` | ≥0.4.0 | TensorDict struktúrák |
| `rlcard` | ≥1.1.0 | NLHE játékmotor |
| `pettingzoo` | ≥1.24.0 | Multi-ágens környezet (alternatív) |
| `gymnasium` | ≥0.29.0 | Env wrapper API |
| `numpy` | ≥1.24.0 | Numerikus számítások |
| `scipy` | ≥1.11.0 | Statisztikai tesztek |
| `pyyaml` | ≥6.0 | Config olvasás |
| `omegaconf` | ≥2.3.0 | Strukturált konfiguráció (jövő) |
| `huggingface-hub` | ≥0.21.0 | CommitScheduler, sync |
| `tqdm` | ≥4.66.0 | Progress bar |
| `rich` | ≥13.0.0 | Konzol formázás |
| `zstandard` | ≥0.22.0 | Buffer tömörítés |
| `treys` | ≥0.1.8 | Poker hand evaluator |

### 12.2 Fejlesztői (Dev)

`pytest`, `pytest-cov`, `mypy`, `ruff`, `pre-commit`, `ipykernel`

### 12.3 Kiértékelési (Eval)

`matplotlib`, `seaborn`, `pandas`

---

## 13. Hibaelhárítási Útmutató

### 13.1 "ModuleNotFoundError: No module named 'src'"

**Ok:** A projekt nincs telepítve szerkeszthető csomagként.
**Megoldás:**
```bash
cd PokerAI-Project
pip install -e .
```

### 13.2 "CUDA Out of Memory"

**Ok:** A batch méret túl nagy a GPU memóriájához képest.
**Megoldás:**
1. Csökkentsd a `ppo.rollout_steps`-et a `config.yaml`-ben
2. Csökkentsd a `ppo.batch_size`-t
3. Csökkentsd a model réteg méreteit

### 13.3 "NaN loss detektálva"

**Ok:** A gradiens felrobbant (exploding gradients) vagy a learning rate túl magas.
**Megoldás:**
1. A `FaultHandler` automatikusan rollback-el az utolsó stabil checkpoint-ra
2. Csökkentsd a `ppo.learning_rate`-et
3. Csökkentsd a `ppo.max_grad_norm`-ot (szorosabb gradient clipping)

### 13.4 "Kaggle Exit Code 137"

**Ok:** A session elérte a 12 órás limitet és SIGKILL-t kapott.
**Megoldás:** Ellenőrizd, hogy a `mlops.graceful_shutdown.max_runtime_hours`
értéke 11.5 (nem 12.0). A 30 perces puffer kritikus.

### 13.5 "HF auth: nem találhatón token"

**Ok:** A HF_TOKEN nincs beállítva.
**Megoldás:**
- Lokálisan: `export HF_TOKEN=hf_xxxxx`
- Kaggle: Add-ons → Secrets → HF_TOKEN (write jogosultsággal)

### 13.6 A tesztek "FakeTensor" hibákat dobnak

**Ok:** A torch mock nem támogat egy adott torch API-t.
**Megoldás:** Ezek a tesztek valódi PyTorch-cal futtatandók (`pip install torch`).
A mock-kal nem tesztelhető részek a `conftest.py`-ban dokumentálva vannak.

---

## 14. Változásnapló

### v0.1.0 (2025-03-26) — Initial Implementation

**Fázis 1: Projekt Alapok**
- `pyproject.toml`: Build rendszer, 20+ függőség, mypy/ruff/pytest konfiguráció
- `config.yaml`: 353 soros központi konfiguráció, GTO mátrix 6 asztalméretre

**Fázis 2: Környezeti Modul (src/env/)**
- `features.py`: 281-dim Observation Space (multi-hot kártyák, normalizált metrikák)
- `action_mapper.py`: 9 diszkrét akció, pot-relatív sizing, -1e8 maszkolás
- `equity.py`: Monte Carlo equity + Treys evaluator + fallback heurisztika

**Fázis 3: Neurális Hálózat (src/model/)**
- `networks.py`: PPO Actor-Critic, 4 embedding modul, ~600k paraméter, ortogonális init

**Fázis 4: Training Modul (src/training/)**
- `buffer.py`: GAE (reverse sweep), advantage normalizálás, mini-batch generátor
- `collector.py`: Environment stepping, inference mode, bootstrap
- `trainer.py`: PPO clipped loss, gradient clipping, hot-reload
- `opponent_pool.py`: 4 statikus archetípus, FSP snapshot pool
- `runner.py`: Event-driven ciklus, callback rendszer, graceful shutdown

**Fázis 5: Orchestrator (src/orchestrator/)**
- `telemetry.py`: 6 HUD metrika, O(1) sliding window, anomália detektálás
- `curriculum.py`: 3-fázis, UCB1 MAB, állapot mentés/betöltés
- `reward_shaper.py`: Blöff büntetés, agresszió bonus, hot-reload
- `orchestrator.py`: Singleton, beavatkozási logika, config hot-reload

**Fázis 6: MLOps (src/mlops/)**
- `state_manager.py`: 5-RNG capture/restore, CheckpointManager rotációval
- `hf_sync.py`: CommitScheduler, headless auth, snapshot download/upload
- `fault_tolerance.py`: GracefulShutdown (monotonic), FaultHandler (NaN/OOM)

**Fázis 7: Szkriptek**
- `train_local.py`: CLI pipeline (argparse → config → bootstrap → run)
- `train_kaggle.ipynb`: Kaggle notebook (clone → install → auth → train → upload)

**Fázis 8-9: Tesztelés és Produkció**
- 159 egységteszt (torch mock-kal CI/CD kompatibilis)
- `README.md`: Produkciós dokumentáció, architektúra diagram
- `MASTER_NOTE.md`: Fejlesztési biblia (ez a dokumentum)

---

> **Megjegyzés a jövőbeli fejlesztőknek:**
> Ez a dokumentum a projekt fejlesztési döntéseinek és konvencióinak
> élő referenciája. Ha új modult adsz hozzá vagy meglévőt módosítasz,
> frissítsd a megfelelő szekciót. A cél az, hogy bárki, aki először
> nyitja meg a projektet, ebből a dokumentumból megértse az összes
> "miért"-et a kód mögött.
