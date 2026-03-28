"""
M2 + M4 Javitasok — Patch dokumentacio

=============================================================================
FIX M2 — opponent_pool.py: OpponentPool.add_snapshot() atomikus iras
=============================================================================

A korabbi torch.save() kozvetlen fajlba irt, amit SIGKILL (Kaggle 12h limit)
korrumalhatott. A javitas: temp fajl + os.replace() atomikus minta.

ELOTT (opponent_pool.py L357-360):
    torch.save(model_state_dict, filepath)

UTAN:
    # [FIX M2] Atomikus mentes: temp fajl → rename (SIGKILL-stabil)
    tmp_filepath = filepath + ".tmp"
    try:
        torch.save(model_state_dict, tmp_filepath)
        os.replace(tmp_filepath, filepath)
    except Exception as save_exc:
        if os.path.exists(tmp_filepath):
            try:
                os.remove(tmp_filepath)
            except OSError:
                pass
        raise save_exc

Es a fajl tetejere hozzaadni: import os

=============================================================================
FIX M4 — curriculum.py: UCBArm.ucb_score() UCB1 off-by-one javitas
=============================================================================

Az UCB1 formula log(0) = -inf-et ad ha total_rounds=0 (elso hivas).
Az existing guard (selection_count == 0 → inf) jol kezeli az elso kivalasztast,
de a MASODIK hivas total_selections_cached=1-gyel fut, ahol log(1) = 0,
tehat az exploration bonus 0. Az igazi UCB1 tipikusan minden kart egyszer
latogat eloszor.

Az eletendo valtozas ketteagazu:

1. CurriculumManager.select_opponent() — total_rounds=0 eseten fallback:

ELOTT (curriculum.py):
    total_selections_cached: int = self._total_selections
    self._total_selections += 1
    ...
    score: float = arm.ucb_score(total_selections_cached, c)

UTAN:
    total_selections_cached: int = self._total_selections
    self._total_selections += 1
    # [FIX M4] Ha total_selections_cached == 0 (elso hivas), az UCB1
    # log(0) = -inf-et adna. A total_rounds-t minimum 1-re korlátozzuk.
    effective_rounds: int = max(total_selections_cached, 1)
    ...
    score: float = arm.ucb_score(effective_rounds, c)

2. UCBArm.ucb_score() — opcionalis, de ajanlott extra guard:

ELOTT:
    if self.selection_count == 0:
        return float("inf")

UTAN:
    if self.selection_count == 0:
        return float("inf")
    # total_rounds minimum 1 garantalasa (ket vedemi szint)
    safe_total_rounds = max(total_rounds, 1)
    exploitation: float = self.average_reward
    exploration: float = c * math.sqrt(
        math.log(safe_total_rounds) / self.selection_count
    )
    return exploitation + exploration

=============================================================================
BEILLESZTESI UTMUTATO
=============================================================================

opponent_pool.py:
1. Hozzaadni: import os (a fajl tetejen mar lehet, ellenorizd)
2. Megkeresni a `torch.save(model_state_dict, filepath)` sort (~L357)
3. Kicserelni a fenti atomikus mintra

curriculum.py:
1. Megkeresni a select_opponent() metodust (~L188)
2. A total_selections_cached sor utan hozzaadni az effective_rounds szamitast
3. Az arm.ucb_score(total_selections_cached, c) sort lecserelni:
   arm.ucb_score(effective_rounds, c)
4. Opcionalis: UCBArm.ucb_score()-ban is hozzaadni a safe_total_rounds guard-ot
"""

# =============================================================================
# Ez a fajl dokumentacios celra keszult (patch utmutato).
# =============================================================================
