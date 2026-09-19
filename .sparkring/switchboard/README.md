# GLM TP2 switchboard adapter

`control.py` coordinates the immutable SparkRing R33 TP2 generation. It never
constructs a Docker create command: rank creation and start are delegated to
`runtime.common.tp2.execute` using pre-rendered `plans/rank0.json` and
`plans/rank1.json`. Rank 1 is created/started first; rank 0 follows. Shutdown
uses the reverse order and only touches containers that match the plans' image,
labels, rank, and exact names.

The plan files are deployment artifacts and stay ignored by Git because they
contain site paths and environment. Render each plan before registering the
recipe. From `/home/khalil/sparkring-glm53f-tp2`, set the rank-specific inputs
and run this once for each rank:

```bash
mkdir -p .sparkring/switchboard/plans
RANK=0                         # use 1 for the worker plan
MASTER=10.100.88.2
MODEL_DIR=/srv/models/GLM-5.3-Flash-NVFP4-Spark/a608241
CACHE_DIR=/srv/cache/glm53-r33-tp2-port8888-a6082410
ENV_FILE="$PWD/.sparkring/glm53-rank${RANK}.env"
IMAGE='ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef'
SELECTION=".sparkring/switchboard/selection-rank${RANK}.json"
PLAN=".sparkring/switchboard/plans/rank${RANK}.json"

python3 -m runtime.common.launch \
  glm53-flash-spark-tp2-dcp1-sparkcache -- plan \
  --rank "$RANK" --master "$MASTER" \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
  --env-file "$ENV_FILE" --image "$IMAGE" > "$SELECTION"

python3 - "$SELECTION" "$PLAN" <<'PY'
import json, subprocess, sys
from pathlib import Path
selection = json.loads(Path(sys.argv[1]).read_text())
result = subprocess.run(selection["command"], cwd=selection["working_directory"],
                        text=True, capture_output=True, check=True)
plan = json.loads(result.stdout)
Path(sys.argv[2]).write_text(json.dumps(plan, indent=2) + "\n")
PY
```

Rank environment files must use the verified site values, including the paired
overrides `NCCL_IB_HCA==rocep1s0f1,roceP2p1s0f1` and
`NCCL_IB_GID_INDEX=3`. Generate rank 0 with `VLLM_HOST_IP=10.100.88.2` and rank
1 with its own direct-link address. Use identical model/cache paths on both
hosts and copy both final plan files to the same checkout path on `bottom`.

Read-only checks:

```bash
./control.py preflight
./control.py status       # 0 healthy, 1 inactive, 2 degraded/unsafe
./control.py health       # 0 only when both ranks and /v1/models are healthy
```

Do not run `start` or `stop` while staging or registering the recipe.
