# 📚 Development Documentation Index

Welcome to the Poker AI V5 documentation hub. Start here for an overview, then explore specific guides below.

---

## 📖 Start Here

**See also**: [MASTER_NOTE.md](../MASTER_NOTE.md) — High-level project status and architecture overview

---

## 📑 Documentation Files

### Phase Implementation Guides

| File | Purpose | Phase | Status |
|------|---------|-------|--------|
| [PHASE2_INTEGRATION_GUIDE.md](PHASE2_INTEGRATION_GUIDE.md) | MCCFR + Regret buffer setup | 2 | ✅ Complete |
| [PHASE2_COMPLETION_REPORT.md](PHASE2_COMPLETION_REPORT.md) | Phase 2 milestone summary | 2 | ✅ Complete |
| [PHASE3_IMPLEMENTATION_GUIDE.md](PHASE3_IMPLEMENTATION_GUIDE.md) | Strategy network training | 3 | ✅ Complete |
| [PHASE4_IMPLEMENTATION_GUIDE.md](PHASE4_IMPLEMENTATION_GUIDE.md) | Real-time subgame solving | 4 | ✅ Complete |
| [PHASE5_IMPLEMENTATION_GUIDE.md](PHASE5_IMPLEMENTATION_GUIDE.md) | Exploitability measurement | 5 | ⏳ In Progress |

### Audit & Quality Assurance

| File | Purpose | Status |
|------|---------|--------|
| [AUDIT_FIXES.md](AUDIT_FIXES.md) | Tier 1 audit fixes implementation | ✅ COMPLETE |
| [SELF_AUDIT_REPORT.md](SELF_AUDIT_REPORT.md) | Initial self-audit findings | ✅ Complete |

### Project Planning

| File | Purpose | Status |
|------|---------|--------|
| [ROADMAP.md](ROADMAP.md) | High-level project roadmap | ✅ Updated |

---

## 🧪 Verification Scripts

### `verify_audit_fixes.py`
**Location**: `docs/verify_audit_fixes.py`

Automated verification for all Tier 1 Deep CFR audit fixes.

**Usage**:
```bash
python docs/verify_audit_fixes.py
```

**Verifies**:
- ✅ **Fix #1**: Pure DCFR (no importance weighting)
- ✅ **Fix #2.5**: Reach probability documentation
- ✅ **Fix #3**: Strategy averaging (convergence key)
- ✅ **Fix #4**: Game form extraction framework

**Output**: All 4 tests passing = system is mathematically sound

---

## 🎯 Quick Reference

### For Deep CFR Algorithm
→ [PHASE2_INTEGRATION_GUIDE.md](PHASE2_INTEGRATION_GUIDE.md)

### For Network Training
→ [PHASE3_IMPLEMENTATION_GUIDE.md](PHASE3_IMPLEMENTATION_GUIDE.md)

### For Real-time Subgame Solving
→ [PHASE4_IMPLEMENTATION_GUIDE.md](PHASE4_IMPLEMENTATION_GUIDE.md)

### For Exploitability Testing (Phase 5)
→ [PHASE5_IMPLEMENTATION_GUIDE.md](PHASE5_IMPLEMENTATION_GUIDE.md)

### For Architecture Overview
→ [../MASTER_NOTE.md](../MASTER_NOTE.md)

---

## ✅ Recent Changes (March 29, 2026)

**All Tier 1 Audit Fixes Implemented**:
- Pure DCFR formula (O(1/√T) convergence guaranteed)
- Strategy averaging enabled
- Reach probability documentation added
- Game form extraction framework ready

**Documentation Reorganized**:
- All development docs moved to `docs/` folder
- MASTER_NOTE.md remains at project root (entry point)
- This README.md serves as documentation index

---

## 📞 Next Steps

### Immediate (Tier 2):
1. Implement bucket-weighted reach probability calculation
2. Complete full game form tree traversal
3. Run verification tests against Kuhn poker

### Current (Pre-Phase 5):
1. Validate Pure DCFR convergence on small games
2. Benchmark against theoretical GTO bounds
3. Test on heads-up NLHE

### Phase 5:
1. Implement exact exploitability measurement
2. Run full training on Kaggle
3. Benchmark against Slumbot/Libratus
