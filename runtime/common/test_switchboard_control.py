import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTROL_PATH = ROOT / ".sparkring/switchboard/control.py"


def load_control():
    spec = importlib.util.spec_from_file_location("glm_switchboard_control", CONTROL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(rank):
    name = f"sparkring-r33-tp2-dcp1-sparkcache-port8888-a6082410-r{rank}"
    return {
        "schema": "sparkring-profile-launch-plan/v1",
        "status": "implemented",
        "name": name,
        "profile": "tp2-dcp1-sparkcache",
        "image": "ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef",
        "labels": {
            "org.sparkring.profile": "tp2-dcp1-sparkcache",
            "org.sparkring.generation": "port8888-a6082410",
        },
        "environment": {"NODE_RANK": str(rank)},
        "model": {"revision": "a608241037e4c2565356bff7ca293f2133888f88"},
        "container_args": [
            "/opt/sparkring/bin/sparkring", "serve", "/models/target",
            "--port", "8888", "--served-model-name", "GLM-5.3-Flash-NVFP4-Spark",
        ],
        "command": ["docker", "create", "--name", name],
    }


def write_plans(path):
    path.mkdir(parents=True)
    for rank in (0, 1):
        (path / f"rank{rank}.json").write_text(json.dumps(plan(rank)))


def container(module, rank, status="exited"):
    item = plan(rank)
    return {
        "Id": f"id-{rank}",
        "Image": "sha256:local-config-id",
        "Config": {"Image": item["image"], "Labels": item["labels"], "Env": [f"NODE_RANK={rank}"]},
        "State": {"Status": status, "Running": status == "running"},
    }


def test_plan_validation_pins_names_labels_image_revision_port_and_served_id():
    module = load_control()
    for rank in (0, 1):
        module.validate_plan(plan(rank), rank)

    mutations = [
        (["name"], "wrong"),
        (["image"], "ghcr.io/fujitsupolycom/sparkring@sha256:" + "0" * 64),
        (["model", "revision"], "0" * 40),
        (["labels", "org.sparkring.generation"], "other"),
        (["environment", "NODE_RANK"], "9"),
    ]
    for pieces, value in mutations:
        changed = json.loads(json.dumps(plan(0)))
        target = changed
        for piece in pieces[:-1]:
            target = target[piece]
        target[pieces[-1]] = value
        with pytest.raises(RuntimeError):
            module.validate_plan(changed, 0)


def test_inspect_rank_accepts_docker_lowercase_missing_object():
    module = load_control()

    def runner(*_args, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="[]\n", stderr="error: no such object: missing\n")

    controller = module.Controller(runner=runner)
    assert controller.inspect_rank(0) is None
    with pytest.raises(RuntimeError, match="cannot inspect rank 0 container"):
        controller.inspect_rank(0, required=True)


def test_start_creates_missing_worker_then_top_and_starts_in_same_order(tmp_path):
    module = load_control()
    plans = tmp_path / "plans"
    write_plans(plans)
    controller = module.Controller(plan_dir=plans, sleep=lambda _: None)
    states = {0: None, 1: None}
    actions = []

    controller.inspect_rank = lambda rank, required=False: states[rank]

    def rank_action(rank, action):
        actions.append((rank, action))
        states[rank] = container(module, rank, "created" if action == "create" else "running")

    controller.rank_action = rank_action
    controller.api_ready = lambda: all(states[r] and states[r]["State"]["Running"] for r in (0, 1))

    controller.start()

    assert actions == [(1, "create"), (0, "create"), (1, "start"), (0, "start")]


def test_start_refuses_a_partially_running_pair_without_mutation(tmp_path):
    module = load_control()
    plans = tmp_path / "plans"
    write_plans(plans)
    controller = module.Controller(plan_dir=plans)
    states = {0: container(module, 0, "running"), 1: container(module, 1, "exited")}
    controller.inspect_rank = lambda rank, required=False: states[rank]
    actions = []
    controller.rank_action = lambda rank, action: actions.append((rank, action))

    with pytest.raises(RuntimeError, match="partially running"):
        controller.start()
    assert actions == []


def test_start_failure_cleans_up_top_then_worker_for_rollback(tmp_path):
    module = load_control()
    plans = tmp_path / "plans"
    write_plans(plans)
    controller = module.Controller(plan_dir=plans, sleep=lambda _: None)
    states = {0: container(module, 0), 1: container(module, 1)}
    events = []
    controller.inspect_rank = lambda rank, required=False: states[rank]

    def rank_action(rank, action):
        events.append((action, rank))
        if action == "start" and rank == 0:
            raise RuntimeError("top failed")
        states[rank] = container(module, rank, "running")

    controller.rank_action = rank_action
    controller.stop_rank = lambda rank, data: events.append(("stop", rank))

    with pytest.raises(RuntimeError, match="top failed"):
        controller.start()
    assert events == [("start", 1), ("start", 0), ("stop", 0), ("stop", 1)]


def test_stop_is_idempotent_and_stops_top_before_worker(tmp_path):
    module = load_control()
    plans = tmp_path / "plans"
    write_plans(plans)
    controller = module.Controller(plan_dir=plans)
    states = {0: container(module, 0, "running"), 1: container(module, 1, "running")}
    events = []
    controller.inspect_rank = lambda rank, required=False: states[rank]

    def stop_rank(rank, data):
        if data is None or not data["State"]["Running"]:
            return
        events.append(rank)
        states[rank] = container(module, rank, "exited")

    controller.stop_rank = stop_rank
    controller.stop()
    assert events == [0, 1]

    states = {0: None, 1: None}
    events.clear()
    controller.stop()
    assert events == []


def test_stop_refuses_same_name_with_foreign_labels(tmp_path):
    module = load_control()
    plans = tmp_path / "plans"
    write_plans(plans)
    controller = module.Controller(plan_dir=plans)
    foreign = container(module, 0, "running")
    foreign["Config"]["Labels"]["org.sparkring.generation"] = "foreign"
    controller.inspect_rank = lambda rank, required=False: foreign if rank == 0 else None
    controller.stop_rank = lambda rank, data: pytest.fail("must not stop a foreign container")

    with pytest.raises(RuntimeError, match="label"):
        controller.stop()


def test_health_requires_both_owned_containers_running_and_served_id(tmp_path):
    module = load_control()
    plans = tmp_path / "plans"
    write_plans(plans)
    controller = module.Controller(plan_dir=plans)
    states = {0: container(module, 0, "running"), 1: container(module, 1, "running")}
    controller.inspect_rank = lambda rank, required=False: states[rank]
    controller.api_ready = lambda: True
    assert controller.health() is True

    states[1] = container(module, 1, "exited")
    assert controller.health() is False
    states[1] = container(module, 1, "running")
    controller.api_ready = lambda: False
    assert controller.health() is False


def test_status_exit_codes_distinguish_healthy_inactive_and_degraded(tmp_path, capsys):
    module = load_control()
    plans = tmp_path / "plans"
    write_plans(plans)
    controller = module.Controller(plan_dir=plans)
    states = {0: None, 1: None}
    controller.inspect_rank = lambda rank, required=False: states[rank]
    controller.api_ready = lambda: False
    assert controller.status() == module.STATUS_INACTIVE
    assert json.loads(capsys.readouterr().out)["state"] == "inactive"

    states[0] = container(module, 0, "running")
    assert controller.status() == module.STATUS_DEGRADED
    assert json.loads(capsys.readouterr().out)["state"] == "degraded"

    states[1] = container(module, 1, "running")
    controller.api_ready = lambda: True
    assert controller.status() == module.STATUS_HEALTHY
    assert json.loads(capsys.readouterr().out)["state"] == "healthy"


def test_rank_executor_delegates_exact_plan_to_tp2_execute(tmp_path, monkeypatch):
    module = load_control()
    plan_path = tmp_path / "rank1.json"
    plan_path.write_text(json.dumps(plan(1)))
    receipt_path = tmp_path / "receipt.json"
    receipt = {"schema": "test-receipt"}
    receipt_path.write_text(json.dumps(receipt))
    calls = []
    monkeypatch.setattr(module.tp2, "execute", lambda loaded, action, loaded_receipt: calls.append((loaded, action, loaded_receipt)))

    module.execute_plan(plan_path, "create", receipt_path=receipt_path)
    module.execute_plan(plan_path, "start", receipt_path=receipt_path)

    assert calls == [(plan(1), "create", receipt), (plan(1), "start", receipt)]
