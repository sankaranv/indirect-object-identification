#!/bin/bash
# Run all IOI experiments in dependency order.
# Log goes to run_all.log in this directory.

set -uo pipefail
cd "$(dirname "$0")"

LOG="run_all.log"
PASS=0
FAIL=0
FAILED_EXPS=()

run_exp() {
    local name="$1"
    echo "" >> "$LOG"
    echo "================================================================" >> "$LOG"
    echo "$(date '+%H:%M:%S') START  $name" | tee -a "$LOG"
    if python3 "experiments/$name" >> "$LOG" 2>&1; then
        echo "$(date '+%H:%M:%S') PASS   $name" | tee -a "$LOG"
        PASS=$((PASS + 1))
    else
        echo "$(date '+%H:%M:%S') FAIL   $name (exit $?)" | tee -a "$LOG"
        FAIL=$((FAIL + 1))
        FAILED_EXPS+=("$name")
    fi
}

echo "$(date): run_all_experiments.sh starting" | tee "$LOG"

# ── Phase 1: Name movers ────────────────────────────────────────────────────
# fig3 generates results/name_movers/head_to_logits_causal_effect.csv
# which fig3b and fig_backup_name_movers depend on.
run_exp discovery/fig3_name_movers.py
run_exp discovery/fig3b_head_effects.py
run_exp discovery/fig3c_name_mover_copying.py

# ── Phase 2: S-Inhibition heads ─────────────────────────────────────────────
run_exp discovery/fig4_s2_inhibition.py
run_exp discovery/fig4c_s2_inhibition_combined.py

# ── Phase 3: Early heads (DT / PT / IH) ─────────────────────────────────────
run_exp discovery/fig5_early_heads.py

# ── Phase 4: Circuit validation ──────────────────────────────────────────────
run_exp validation/fig6_circuit_validation.py
run_exp validation/fig7_minimality.py
run_exp validation/fig8_performance_summary.py

# ── Phase 5: Appendix experiments (all independent) ─────────────────────────
run_exp appendix/appA_signal_decomposition.py
run_exp appendix/appC_s2_inhibition_key_signals.py
run_exp appendix/appD_induction_key_signals.py
run_exp appendix/appE_templates.py
run_exp appendix/appF_backup_name_mover_discovery.py
run_exp appendix/appF_backup_name_mover_effects.py
run_exp appendix/appF_backup_name_mover_copying.py
run_exp appendix/appH_head_copy_strength.py
run_exp appendix/appHI_induction_pattern_scores.py
run_exp appendix/appJ_mlp_knockout.py
run_exp appendix/appK_minimality_sets.py
run_exp appendix/appM_greedy_completeness.py

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "$(date): DONE  passed=$PASS  failed=$FAIL" | tee -a "$LOG"
if [ ${#FAILED_EXPS[@]} -gt 0 ]; then
    echo "Failed experiments:" | tee -a "$LOG"
    for e in "${FAILED_EXPS[@]}"; do
        echo "  - $e" | tee -a "$LOG"
    done
fi
