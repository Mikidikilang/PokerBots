# PokerAI-NLHE: Fejlesztői Memória & Protokoll

## 1. Együttműködési Modell (Architect-Coder Pattern)
- **Architect (Gemini):** Felelős a magas szintű tervezésért, a Roadmap betartásáért és a technikai specifikációk (Architect Prompts) összeállításáért.
- **Coder (Local Qwen-2.5-Coder):** Felelős a konkrét Python kód, egységtesztek és implementáció kidolgozásáért a megadott promptok alapján.
- **Bridge (User):** A közvetítő, aki futtatja a teszteket, kezeli a fájlrendszert és visszacsatolást ad a két AI között.

## 2. Projekt Kontextus
- **Cél:** No-Limit Texas Hold'em RL AI (PPO alapú).
- **Architektúra:** POMDP keretrendszer, 104-dim kártyakódolás, 9-dim diszkrét akciótér.
- **Fő technológiák:** PyTorch, TorchRL, RLCard, Zstandard.

## 3. Fejlesztési Protokoll (Lépések)
1. **Architect Prompt:** Az Architect angol nyelvű, részletes technikai utasítást ad (Task, Specs, Requirements, Test).
2. **Implementation:** A User beadja a promptot a lokális Qwennek.
3. **Verification:** A User lefuttatja a generált egységteszteket (pytest).
4. **Memory Update:** Siker esetén a User jelzi az Architectnek, aki frissíti ezt a dokumentumot.

## 4. Jelenlegi Állapot
- **Aktuális Fázis:** Fázis 2 (Környezeti Modul)
- **Folyamatban lévő fájl:** `src/env/features.py`
- **Befejezett:** Fázis 1 (Struktúra és Konfiguráció).

## 5. Technikai Szerződések (Constraints)
- **Kártyák:** Multi-hot 52-dim (Rank*4 + Suit).
- **Normalizáció:** Monetáris értékek / Big Blind, [0, 1] tartományban.
- **Történet:** (18, 9) alakú tenzor, zero-padding.