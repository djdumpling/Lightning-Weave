#!/usr/bin/env bash
# Stages of the agent-efficiency synthesis experiments on Modal; see docs/agent_efficiency_synthesis.md.
#   bash scripts/run_agent_eff.sh plan
#   bash scripts/run_agent_eff.sh download --donors klear,decs,deepscaler
#   bash scripts/run_agent_eff.sh prepare  --donors agent_acc,klear,decs,deepscaler
#   bash scripts/run_agent_eff.sh score    --donors klear,decs,deepscaler --workers 8   # server-side chain, then verify
#   bash scripts/run_agent_eff.sh precision --donors decs             # independent numerical re-score
#   bash scripts/run_agent_eff.sh verify   --donors klear,decs,deepscaler
#   bash scripts/run_agent_eff.sh onboard  --donors l1max,dler --tag census   # download..verify, then analyze with the core
#   bash scripts/run_agent_eff.sh analyze  --donors klear,decs,deepscaler --tag core
#   bash scripts/run_agent_eff.sh probe    --donors klear,decs,deepscaler --tag core --workers 4
#   bash scripts/run_agent_eff.sh compose  --variant acc-clean+decs-deepscaler
#   bash scripts/run_agent_eff.sh train    --variant acc-clean+decs-deepscaler --seed 1234
#   bash scripts/run_agent_eff.sh export   --variant acc-clean+decs-deepscaler --seed 1234
#   bash scripts/run_agent_eff.sh build    --variant acc-legacy+l1max,acc-legacy+nemotron --seeds 1234,5678
#   bash scripts/run_agent_eff.sh build --variant paper-acc-legacy,paper-acc-legacy+decs,paper-half-acc-legacy,acc-legacy+decs-protect-turn-starts,acc-legacy+decs-gate-random-multi-turn --seeds 1234,5678
set -euo pipefail
cd "$(dirname "$0")/.."
action="${1:-plan}"
shift || true
if [[ "$action" == "plan" ]]; then
  exec python3 -m configs.agent_eff.config
fi
export MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-alex-dev-2}"
detach=()
case "$action" in
  train|score|onboard|build|probe|analyze|precision) detach=(--detach) ;;
esac
exec uv run --no-project --python 3.12 --with modal==1.5.5 \
  modal run ${detach[@]+"${detach[@]}"} configs/agent_eff/modal_pipeline.py --action "$action" "$@"
