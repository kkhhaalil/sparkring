# GLM-5.3 Flash R33 TP2 SparkCache

This is the canonical two-Spark GLM-5.3 Flash deployment in the `kkhhaalil/sparkring` fork.
It fixes the deployment to:

- served model ID `GLM-5.3-Flash-NVFP4-Spark`;
- API port `8888`;
- TP2, DCP1, MTP3, and SparkCache;
- model revision `a608241037e4c2565356bff7ca293f2133888f88`;
- deployment generation `port8888-a6082410`;
- the R33 public image receipt and release contract.

Status is `research-only`: this exact generation has been boot- and generation-validated on the local TP2 cluster, but historical R33 qualification does not automatically transfer to the updated checkpoint revision.

## Site inputs

Create one private environment file per rank. The local cluster uses:

```text
# rank 0
VLLM_HOST_IP=10.100.88.2
NCCL_SOCKET_IFNAME=enp1s0f1np1
GLOO_SOCKET_IFNAME=enp1s0f1np1
NCCL_IB_HCA==rocep1s0f1,roceP2p1s0f1
NCCL_IB_GID_INDEX=3
```

Rank 1 uses `VLLM_HOST_IP=10.100.88.1`; the interface, ordered HCA pair, and GID remain the same. HCA and GID overrides must be supplied together. Keep model and cache paths outside the checkout.

## Plan

From the repository root:

```bash
PYTHONPATH=. python3 runtime/common/launch.py --execute \
  glm53-flash-spark-tp2-dcp1-sparkcache -- plan \
  --rank "$RANK" --master "$MASTER" \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
  --env-file "$ENV_FILE" --image "$SPARKRING_IMAGE"
```

The catalog supplies SparkCache, port, model revision, deployment generation, and immutable runtime receipt arguments. Inspect both plans before creating containers. Start rank 1 before rank 0.

Install and verify the 2 GiB memory guard described in `runtime/profiles/glm53-flash-spark-tp2/README.md` before creating containers. The launcher binds the API without authentication; keep port 8888 on a trusted network or behind an authenticated gateway.

## Verification

Require all of the following:

- both exact generation containers are running;
- `/v1/models` reports `GLM-5.3-Flash-NVFP4-Spark`;
- a non-streaming generation request completes with content;
- logs show world size 2 and RoCEnante using `rocep1s0f1,roceP2p1s0f1`;
- both memory-guard services are active.

The metadata hashes for the pinned checkpoint are recorded in `profiles/glm53-target-variants.json`.
