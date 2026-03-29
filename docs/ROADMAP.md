# PokerAI-NLHE: Fejlesztési Roadmap és Architektúra Összefoglaló

## 1. Dokumentumok Szintézise

A 4 specifikációs dokumentum egy **No-Limit Texas Hold'em RL AI** komplett rendszertervét írja le,
amelynek lényegi elemei:

**A) Állapottér és Akciótér (Specifikáció doc)**
- POMDP keretrendszer: 104-dimenziós multi-hot kártyakódolás, normalizált környezeti metrikák,
  rögzített méretű licittörténet tenzor (~250-300 dimenziós laposított bemeneti vektor)
- 9-dimenziós diszkretizált akciótér (Fold → All-in), Softmax kimenet + Action Masking

**B) Auto-Adaptív Curriculum Orchestrator (Curriculum doc)**
- 3 fázisú tanulás: Phase 0 (statikus botok) → Phase 1 (SFT ellenfelek) → Phase 2 (FSP + MAB)
- HUD metrikák alapú anomáliadetektálás: VPIP, PFR, 3-Bet, AF, WTSD
- Beavatkozási mátrix: Reward Shaping, Entrópia injekció, Ellenfél-pool rotálás
- Hot-Reload mechanizmus a hiperparaméterek futásidejű módosításához

**C) GTO Mátrix és Benchmarking (Orchestrator Data doc)**
- Pontos GTO célsávok és degenerációs küszöbök mind a 6 asztalméretre (2-9 Max)
- Slumbot HUNL benchmark (ACPC protokoll): cél >50 mbb/hand
- Nash Distance számítás: LBR és ISMCTS-BR módszerek, cél <0.3% pot

**D) MLOps Infrastruktúra (Infrastruktúra doc)**
- Kaggle ↔ GitHub ↔ HF Hub háromszög: pip install -e ., headless autentikáció
- GracefulShutdownMonitor: time.monotonic() alapú 11.5 órás limit
- CommitScheduler aszinkron háttérszálas feltöltés
- RNGStateManager: 5-komponensű determinisztikus állapotmentés
- CompressedListStorage (TorchRL + Zstandard) a Replay Bufferekhez

---

## 2. Projekt Architektúra (Megerősítve)

```
PokerAI-Project/
├── pyproject.toml              ← [Fázis 1] Projekt metaadatok, függőségek
├── config.yaml                 ← [Fázis 1] Központi hiperparaméter definíciók
├── README.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── env/                    # ÁLLAPOTTÉR ÉS KÖRNYEZETI MODUL
│   │   ├── __init__.py
│   │   ├── features.py         # Observation Space konstruktor (104+ dim tenzor)
│   │   ├── action_mapper.py    # 0-8 index → póker akció fordító + Action Masking
│   │   └── equity.py           # Pre-flop / post-flop equity kalkulátor
│   │
│   ├── model/                  # NEURÁLIS ARCHITEKTÚRA MODUL
│   │   ├── __init__.py
│   │   └── networks.py         # PPO Actor-Critic hálózatok (PyTorch)
│   │
│   ├── orchestrator/           # CURRICULUM ÉS BEAVATKOZÁSI MODUL
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # Singleton állapotgép, fázisátmenetek, hot-reload
│   │   ├── telemetry.py        # HUD metrikák (VPIP, PFR, AF) mozgóablakos számítása
│   │   ├── curriculum.py       # GTO validálás, patológia-detektálás, MAB ellenfél-kiválasztás
│   │   └── reward_shaper.py    # Dinamikus jutalom-módosítások és büntetések
│   │
│   ├── training/               # RL OPTIMALIZÁCIÓS MODUL
│   │   ├── __init__.py
│   │   ├── runner.py           # Event-Driven fő végrehajtási ciklus
│   │   ├── collector.py        # Rollout adatgyűjtés (trajectories)
│   │   ├── trainer.py          # PPO gradiens frissítés, policy/value loss
│   │   ├── buffer.py           # CompressedListStorage + TensorDict integráció
│   │   └── opponent_pool.py    # FSP snapshot-ok és statikus archetípusok kezelése
│   │
│   └── mlops/                  # INFRASTRUKTÚRA ÉS HIBATŰRÉS MODUL
│       ├── __init__.py
│       ├── hf_sync.py          # Aszinkron HF CommitScheduler, headless auth
│       ├── state_manager.py    # Checkpoint szerializáció, RNGStateManager
│       └── fault_tolerance.py  # GracefulShutdownMonitor, időzítő logika
│
├── scripts/
│   ├── train_kaggle.ipynb      # Kaggle-specifikus inicializáló notebook
│   └── train_local.py          # CLI belépési pont lokális teszteléshez
│
└── tests/                      # Egység- és integrációs tesztek
    ├── test_env/
    ├── test_model/
    ├── test_orchestrator/
    ├── test_training/
    └── test_mlops/
```

---

## 3. Fejlesztési Fázisok (Roadmap)

### Fázis 1: Projekt Alapok és Konfiguráció ✅
**Státusz: KÉSZ**
- `pyproject.toml` — függőségek, build rendszer, tool konfiguráció
- `config.yaml` — teljes hiperparaméter definíció, GTO mátrix, küszöbértékek
- Mappastruktúra generálás, `__init__.py` fájlok

### Fázis 2: Környezeti Modul (`src/env/`)
**Prioritás: KÖVETKEZŐ**
- `features.py` — Observation Space konstruktor:
  - Multi-hot kártyakódolás (52-dim × 2)
  - Környezeti metrikák normalizálása (BB-relatív)
  - Licittörténet rögzített méretű tenzor (18×9)
  - Pozíció one-hot kódolás
- `action_mapper.py` — Diszkrét akciótér kezelés:
  - 9-dimenziós akció index → szemantikai leképezés
  - Action Masking implementáció (logit maszkolás -1e8-cal)
  - Illegális akciók szűrése a játékmotor szabályai szerint
- `equity.py` — Kézerő kalkulátor (Treys integráció)

### Fázis 3: Neurális Hálózat (`src/model/`)
- `networks.py` — PPO Actor-Critic architektúra:
  - Beágyazó rétegek (card, context, history embedding)
  - Megosztott törzs (shared trunk) + szétválás Actor/Critic ágakra
  - Akció maszkolás a Softmax előtt
  - Ortogonális súlyinicializáció
  - Típusos forward pass (observation dict → action dist + value)

### Fázis 4: Training Modul (`src/training/`)
- `buffer.py` — CompressedListStorage integráció TorchRL-lel
- `collector.py` — Rollout adatgyűjtő (environment stepping + tenzor készítés)
- `trainer.py` — PPO optimalizáció (clip, entropy, value loss, GAE)
- `opponent_pool.py` — Statikus archetípusok + FSP snapshot kezelés
- `runner.py` — Event-Driven fő ciklus (adatgyűjtés → frissítés → telemetria → mentés)

### Fázis 5: Orchestrator Modul (`src/orchestrator/`)
- `telemetry.py` — VPIP, PFR, 3-Bet, AF, WTSD mozgóablakos számítása
- `curriculum.py` — GTO mátrix validálás, patológia detektálás, MAB logika
- `reward_shaper.py` — Dinamikus jutalom módosítások (blöff büntetés, passzivitás bonus)
- `orchestrator.py` — Singleton állapotgép, fázisátmenetek, DynamicConfigReloader

### Fázis 6: MLOps Infrastruktúra (`src/mlops/`)
- `state_manager.py` — Checkpoint szerializáció + RNGStateManager (5 komponens)
- `hf_sync.py` — CommitScheduler wrapper, headless autentikáció
- `fault_tolerance.py` — GracefulShutdownMonitor (monotonic clock)

### Fázis 7: Integrációs Szkriptek
- `scripts/train_local.py` — CLI belépési pont (argparse + config betöltés)
- `scripts/train_kaggle.ipynb` — Kaggle notebook (git clone, pip install, runner indítás)

### Fázis 8: Tesztelés és Validáció
- Egységtesztek minden modulhoz
- Integrációs tesztek (teljes tréning iteráció kis mérettel)
- Benchmark pipeline (Slumbot API, Nash Distance)

### Fázis 9: Finomhangolás és Produkció
- Teljesítmény optimalizálás (profiling, batch méret hangolás)
- Multi-GPU támogatás (DataParallel / DistributedDataParallel)
- Monitoring dashboard (W&B / TensorBoard integráció)

---

## 4. Iteráció Szabálya

Minden fázis a következő szekvenciát követi:
1. **Kód generálás** — Clean Code, Type Hints, Docstrings, Logging
2. **Ellenőrzés** — mypy, ruff, egységtesztek
3. **Integráció** — A meglévő modulokkal való összekapcsolás
4. **Review** — Architektúra szerződések validálása

A következő iteráció a **Fázis 2: Környezeti Modul** fejlesztésével folytatódik.
