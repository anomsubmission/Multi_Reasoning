#!/usr/bin/env bash
# run_experiments.sh — run all Phase 1 (behavioral) experiments.
#
# Set LLM_BACKEND and the corresponding API key before running:
#
#   OpenAI / compatible:
#     export LLM_BACKEND=openai
#     export OPENAI_API_KEY=sk-...
#     export OPENAI_BASE_URL=https://api.together.xyz/v1  # optional
#
#   Anthropic:
#     export LLM_BACKEND=anthropic
#     export ANTHROPIC_API_KEY=sk-ant-...
#
#   Local HuggingFace (no key needed):
#     export LLM_BACKEND=hf
#
# Each script is resume-safe: re-running skips already-completed calls.
# Results land in results/ as they complete.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Phase 1: Behavioral experiments ==="
echo "Backend: ${LLM_BACKEND:-openai}"
echo ""

echo "[P1] N-scaling degradation"
python run_p1.py

echo ""
echo "[P2/P3/P4] Prior conflict, format sensitivity, hop ordering"
python run_p2_p3_p4.py --experiment all

echo ""
echo "[P5] Cross-generation format test"
python run_p5.py

echo ""
echo "[P3-extra] Maverick scaled re-run"
python run_p3_extra.py

echo ""
echo "[P6] Dose-response across families"
python run_p6_doseresponse.py

echo ""
echo "[P7] Held-out format predictor"
python run_p7_predictor.py

echo ""
echo "[D1] Naturalistic validation (HotpotQA + MuSiQue)"
python run_d1.py

echo ""
echo "=== All done. Run analysis: ==="
echo "  python analyze_spectrum.py"
echo "  python analyze_p7.py"
