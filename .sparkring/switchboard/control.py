#!/usr/bin/env python3
"""Production lifecycle adapter for the immutable GLM-5.3 SparkRing TP2 pair."""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from runtime.common import tp2  # noqa: E402

BOTTOM_HOST = "bottom"
PLAN_DIR = Path(__file__).resolve().parent / "plans"
RECEIPT = REPO / "runtime/sparkring/jovian-r33/public-image-receipt.json"
CONTROL = Path(__file__).resolve()
MODELS_URL = "http://127.0.0.1:8888/v1/models"
SERVED_ID = "GLM-5.3-Flash-NVFP4-Spark"
PROFILE = "tp2-dcp1-sparkcache"
GENERATION = "port8888-a6082410"
REVISION = "a608241037e4c2565356bff7ca293f2133888f88"
IMAGE = "ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef"
NAMES = {
    0: "sparkring-r33-tp2-dcp1-sparkcache-port8888-a6082410-r0",
    1: "sparkring-r33-tp2-dcp1-sparkcache-port8888-a6082410-r1",
}
STATUS_HEALTHY = 0
STATUS_INACTIVE = 1
STATUS_DEGRADED = 2


def run(argv, *, timeout=120, check=True):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {argv[0]}")
    return result


def _option(arguments, flag):
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"plan is missing {flag}") from error


def validate_plan(plan, rank):
    expected = {
        "schema": "sparkring-profile-launch-plan/v1",
        "status": "implemented",
        "name": NAMES[rank],
        "profile": PROFILE,
        "image": IMAGE,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise RuntimeError(f"rank {rank} plan {key} mismatch")
    labels = plan.get("labels") or {}
    for key, value in {
        "org.sparkring.profile": PROFILE,
        "org.sparkring.generation": GENERATION,
    }.items():
        if labels.get(key) != value:
            raise RuntimeError(f"rank {rank} plan label {key} mismatch")
    if (plan.get("environment") or {}).get("NODE_RANK") != str(rank):
        raise RuntimeError(f"rank {rank} plan NODE_RANK mismatch")
    if (plan.get("model") or {}).get("revision") != REVISION:
        raise RuntimeError(f"rank {rank} plan model revision mismatch")
    arguments = plan.get("container_args") or []
    if _option(arguments, "--port") != "8888":
        raise RuntimeError(f"rank {rank} plan port mismatch")
    if _option(arguments, "--served-model-name") != SERVED_ID:
        raise RuntimeError(f"rank {rank} plan served model mismatch")
    command = plan.get("command") or []
    if command[:2] != ["docker", "create"] or NAMES[rank] not in command:
        raise RuntimeError(f"rank {rank} plan create command mismatch")
    return plan


def load_plan(path, rank):
    try:
        plan = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load immutable rank {rank} plan: {path}") from error
    return validate_plan(plan, rank)


def execute_plan(plan_path, action, *, receipt_path=RECEIPT):
    if action not in {"create", "start"}:
        raise RuntimeError("rank action must be create or start")
    path = Path(plan_path)
    try:
        rank = int(path.stem.removeprefix("rank"))
    except ValueError as error:
        raise RuntimeError("rank plan filename must be rank0.json or rank1.json") from error
    if rank not in (0, 1):
        raise RuntimeError("rank plan filename must be rank0.json or rank1.json")
    plan = load_plan(path, rank)
    try:
        receipt = json.loads(Path(receipt_path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot load immutable R33 image receipt") from error
    tp2.execute(plan, action, receipt)


class Controller:
    def __init__(self, *, plan_dir=PLAN_DIR, runner=run, sleep=time.sleep,
                 monotonic=time.monotonic, readiness_timeout=900):
        self.plan_dir = Path(plan_dir)
        self.runner = runner
        self.sleep = sleep
        self.monotonic = monotonic
        self.readiness_timeout = readiness_timeout

    def plan_path(self, rank):
        return self.plan_dir / f"rank{rank}.json"

    def plans(self):
        return {rank: load_plan(self.plan_path(rank), rank) for rank in (0, 1)}

    def _host_command(self, rank, argv):
        if rank == 0:
            return list(argv)
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", BOTTOM_HOST, *argv]

    def inspect_rank(self, rank, required=False):
        result = self.runner(
            self._host_command(rank, ["docker", "inspect", NAMES[rank]]),
            timeout=20, check=False,
        )
        if result.returncode:
            stderr = result.stderr.casefold()
            missing = "no such object" in stderr or "no such container" in stderr
            if missing and not required:
                return None
            raise RuntimeError(f"cannot inspect rank {rank} container")
        try:
            return json.loads(result.stdout)[0]
        except (json.JSONDecodeError, IndexError, TypeError) as error:
            raise RuntimeError(f"invalid rank {rank} container inspection") from error

    def verify_container(self, rank, data, plan):
        config = data.get("Config") or {}
        labels = config.get("Labels") or {}
        if config.get("Image") != IMAGE:
            raise RuntimeError(f"rank {rank} container image digest mismatch")
        for key, value in plan["labels"].items():
            if labels.get(key) != value:
                raise RuntimeError(f"rank {rank} container label {key} mismatch")
        environment = dict(item.split("=", 1) for item in config.get("Env", []) if "=" in item)
        if environment.get("NODE_RANK") != str(rank):
            raise RuntimeError(f"rank {rank} container NODE_RANK mismatch")
        return data

    def rank_action(self, rank, action):
        argv = ["python3", str(CONTROL), "_rank", action, str(self.plan_path(rank))]
        self.runner(self._host_command(rank, argv), timeout=600)

    def stop_rank(self, rank, data):
        if data is None or not data.get("State", {}).get("Running"):
            return
        self.runner(
            self._host_command(rank, ["docker", "stop", "--timeout", "30", NAMES[rank]]),
            timeout=60,
        )

    def api_ready(self):
        try:
            with urllib.request.urlopen(MODELS_URL, timeout=2) as response:
                payload = json.load(response)
        except Exception:
            return False
        return SERVED_ID in {item.get("id") for item in payload.get("data", [])}

    def _owned_pair(self, plans):
        result = {}
        for rank in (0, 1):
            data = self.inspect_rank(rank)
            result[rank] = None if data is None else self.verify_container(rank, data, plans[rank])
        return result

    def start(self):
        plans = self.plans()
        pair = self._owned_pair(plans)
        running = {rank: bool(data and data.get("State", {}).get("Running"))
                   for rank, data in pair.items()}
        if any(running.values()) and not all(running.values()):
            raise RuntimeError("refusing partially running TP pair")
        if not all(running.values()):
            for rank in (1, 0):
                if pair[rank] is None:
                    self.rank_action(rank, "create")
                    pair[rank] = self.verify_container(rank, self.inspect_rank(rank, required=True), plans[rank])
            allowed = {"created", "exited"}
            states = {rank: pair[rank].get("State", {}).get("Status") for rank in (0, 1)}
            if any(states[rank] not in allowed for rank in (0, 1)):
                raise RuntimeError(f"refusing unsafe stopped container states: {states}")
            try:
                self.rank_action(1, "start")
                self.rank_action(0, "start")
            except Exception:
                for rank in (0, 1):
                    self.stop_rank(rank, self.inspect_rank(rank))
                raise

        deadline = self.monotonic() + self.readiness_timeout
        try:
            while self.monotonic() < deadline:
                pair = self._owned_pair(plans)
                states = {rank: (data or {}).get("State", {}).get("Status", "missing")
                          for rank, data in pair.items()}
                if all((pair[rank] or {}).get("State", {}).get("Running") for rank in (0, 1)) and self.api_ready():
                    return
                if any(state in {"dead", "exited", "missing"} for state in states.values()):
                    raise RuntimeError(f"TP rank exited before readiness: {states}")
                self.sleep(2)
            raise RuntimeError("GLM API readiness deadline exceeded")
        except Exception:
            for rank in (0, 1):
                self.stop_rank(rank, self.inspect_rank(rank))
            raise

    def stop(self):
        plans = self.plans()
        pair = self._owned_pair(plans)
        for rank in (0, 1):
            self.stop_rank(rank, pair[rank])
        after = self._owned_pair(plans)
        if any(data and data.get("State", {}).get("Running") for data in after.values()):
            raise RuntimeError("a GLM TP rank remains running after stop")

    def health(self):
        try:
            plans = self.plans()
            pair = self._owned_pair(plans)
        except RuntimeError:
            return False
        return (all(data and data.get("State", {}).get("Running") for data in pair.values())
                and self.api_ready())

    def preflight(self):
        plans = self.plans()
        self._owned_pair(plans)
        print(json.dumps({"state": "ready", "plans": [str(self.plan_path(0)), str(self.plan_path(1))]}, sort_keys=True))

    def status(self):
        try:
            plans = self.plans()
            pair = self._owned_pair(plans)
            ranks = {f"rank{rank}": (data or {}).get("State", {}).get("Status", "missing")
                     for rank, data in pair.items()}
            ready = all(value == "running" for value in ranks.values()) and self.api_ready()
            if ready:
                state, code = "healthy", STATUS_HEALTHY
            elif all(value in {"missing", "created", "exited"} for value in ranks.values()):
                state, code = "inactive", STATUS_INACTIVE
            else:
                state, code = "degraded", STATUS_DEGRADED
            payload = {"state": state, "api_ready": ready, **ranks}
        except RuntimeError as error:
            code = STATUS_DEGRADED
            payload = {"state": "unsafe", "error": str(error)}
        print(json.dumps(payload, sort_keys=True))
        return code


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 3 and argv[0] == "_rank":
        execute_plan(Path(argv[2]), argv[1])
        return 0
    if len(argv) != 1 or argv[0] not in {"start", "stop", "status", "health", "preflight"}:
        print("usage: control.py {start|stop|status|health|preflight}", file=sys.stderr)
        return STATUS_DEGRADED
    controller = Controller()
    try:
        if argv[0] == "start":
            controller.start()
            return 0
        if argv[0] == "stop":
            controller.stop()
            return 0
        if argv[0] == "health":
            return STATUS_HEALTHY if controller.health() else STATUS_INACTIVE
        if argv[0] == "preflight":
            controller.preflight()
            return 0
        return controller.status()
    except Exception as error:
        print(str(error), file=sys.stderr)
        return STATUS_DEGRADED


if __name__ == "__main__":
    raise SystemExit(main())
