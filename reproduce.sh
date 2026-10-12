#!/usr/bin/env bash
# Regenerate every result, figure and check of the paper (main.tex) (run after the README's Setup, inside the
# activated f_models environment). Long runs resume where they stopped.
#
# Usage: bash reproduce.sh [--dry-run] PART...
#   check       tests and the four independent audits (CPU, ~20 min; needs the checkpoint and data)
#   timing      baseline, split check, spike-timing sweep, binned ridge control (GPU, ~5.5 h)
#   embeddings  unit properties, probes, embedding statistics (CPU, ~1 h)
#   lesions     missing neurons: lesion runs and refitted ridge (GPU, ~3 h)
#   transfer    cross-day embedding swaps and ridge (GPU, ~3 h)
#   codebook    neuron letters (GPU, ~2.5 h)
#   figures     all figures of the paper
#   all         every part above in this order, then check (~15 h)
# --dry-run prints the commands without running them.

set -euo pipefail
cd "$(dirname "$0")"

DRY=0
run() {
    echo "+ $*"
    if [ "$DRY" -eq 0 ]; then "$@"; fi
}

timing() {
    run python scripts/setup/baseline.py
    run python scripts/setup/old_split_check.py c_20131003_center_out_reaching \
        c_20131009_random_target_reaching m_20140203_center_out_reaching \
        j_20160405_center_out_reaching m_20140114_random_target_reaching \
        c_20150316_random_target_reaching
    run python -u scripts/timing/run_perturb.py --out sweep
    run python scripts/timing/summarize_sweep.py sweep
    run python -u scripts/timing/ridge_control.py
    run python scripts/timing/ridge_control_summary.py
}

embeddings() {
    run python scripts/embeddings/check_embeddings.py
    run python -u scripts/embeddings/unit_properties.py --workers 6
    run python -u scripts/embeddings/run_probes.py
    run python scripts/embeddings/embedding_stats.py
}

lesions() {
    run python scripts/lesions/dropout_sim.py
    run python -u scripts/lesions/run_lesions.py --out lesion --seeds 0 1 2
    run python -u scripts/lesions/lesion_ridge.py --workers 6
    run python scripts/lesions/lesion_analysis.py
}

transfer() {
    run python -u scripts/transfer/run_transfer.py --out transfer
    run python -u scripts/transfer/run_transfer.py --set matching --out transfer_matching
    run python -u scripts/transfer/run_transfer.py --set combined --out transfer_combined
    run python -u scripts/transfer/ridge_cross_day.py --workers 6
    run python -u scripts/transfer/ridge_cross_day.py --workers 6 --pooled
    run python scripts/transfer/transfer_analysis.py
}

codebook() {
    run python -u scripts/codebook/run_codebook.py --out codebook
    run python -u scripts/codebook/run_codebook.py --set direction --out codebook_direction
    run python scripts/codebook/codebook_analysis.py
}

figures() {
    run python scripts/docs/make_diagrams.py
    run python scripts/docs/paper_figures.py
}

check() {
    run pytest tests
    run python scripts/audit/audit_timing.py
    run python scripts/audit/audit_embeddings_lesions.py
    run python scripts/audit/audit_transfer.py
    run python scripts/audit/audit_codebook.py
}

all() {
    timing; embeddings; lesions; transfer; codebook; figures; check
}

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

parts=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=1 ;;
        -h|--help) usage 0 ;;
        check|timing|embeddings|lesions|transfer|codebook|figures|all) parts+=("$arg") ;;
        *) echo "unknown part: $arg" >&2; usage 1 ;;
    esac
done
[ "${#parts[@]}" -gt 0 ] || usage 1
for part in "${parts[@]}"; do
    echo "=== $part"
    "$part"
done
