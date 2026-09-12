#!/usr/bin/env bash
# Bridges SageMaker's two container conventions onto the same job code.
#
#   Processing: the job definition provides the full command, which runs unchanged.
#   Training:   SageMaker runs `<image> train` and writes the parameters to
#               /opt/ml/input/config/hyperparameters.json, so translate them into flags.
set -euo pipefail

HYPERPARAMETERS_FILE="/opt/ml/input/config/hyperparameters.json"

if [[ "${1:-}" == "train" ]]; then
  if [[ ! -f "${HYPERPARAMETERS_FILE}" ]]; then
    echo "Training job started without ${HYPERPARAMETERS_FILE}" >&2
    exit 2
  fi
  experiment_id="$(jq -r '."experiment-id"' "${HYPERPARAMETERS_FILE}")"
  artifact_base="$(jq -r '."artifact-base"' "${HYPERPARAMETERS_FILE}")"
  model_name="$(jq -r '."model-name"' "${HYPERPARAMETERS_FILE}")"

  for value in "${experiment_id}" "${artifact_base}" "${model_name}"; do
    if [[ -z "${value}" || "${value}" == "null" ]]; then
      echo "Training job is missing a required hyperparameter" >&2
      exit 2
    fi
  done

  exec python -m jobs.training.main \
    --experiment-id "${experiment_id}" \
    --artifact-base "${artifact_base}" \
    --model-name "${model_name}"
fi

exec "$@"
