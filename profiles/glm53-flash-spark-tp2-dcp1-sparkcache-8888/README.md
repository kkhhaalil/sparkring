# GLM-5.3-Flash R33 TP2 SparkCache on port 8888

This immutable deployment generation reuses the existing R33
`tp2-dcp1-sparkcache` runtime contract and public image receipt. It changes only
the API port, the model repository revision, and the deployment-generation
identity; it does not change or supersede the published profile or its evidence.

Use checkout `/home/khalil/sparkring-glm53f-tp2`. Follow the preparation,
memory-guard, rank-file, create, start, and verification procedure in the
[published TP2 guide](../glm53-flash-spark-tp2-dcp1-sparkcache/README.md), but
select this profile through `runtime/common/launch.py`. The catalog fixes:

- served ID `GLM-5.3-Flash-NVFP4-Spark`;
- API port `8888`;
- TP2, DCP1, and SparkCache;
- model revision `a608241037e4c2565356bff7ca293f2133888f88`;
- R33 public image receipt and release contract.

The rank-local environment file may additionally set the ordered private fabric
pair and its verified GID index:

```text
VLLM_HOST_IP=10.100.88.2
NCCL_SOCKET_IFNAME=enp1s0f1np1
GLOO_SOCKET_IFNAME=enp1s0f1np1
NCCL_IB_HCA==rocep1s0f1,roceP2p1s0f1
NCCL_IB_GID_INDEX=3
```

Both HCA names are required, ordered, and unique; HCA and GID overrides must be
provided together. Omit both to retain the published `f0` transport defaults.
Keep model and cache paths outside the checkout and do not reuse a live
deployment directory.

Plan only; this does not create or start containers:

```bash
cd /home/khalil/sparkring-glm53f-tp2
python3 runtime/common/launch.py \
  glm53-flash-spark-tp2-dcp1-sparkcache-8888 -- plan \
  --rank "$RANK" --master "$MASTER" \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
  --env-file "$ENV_FILE" --image "$SPARKRING_IMAGE"
```

The current model revision preserves the pinned `config.json`, index, chat
template, and generation-config hashes recorded in
`profiles/glm53-target-variants.json`. This metadata equivalence admits planning;
it does not transfer the historical R33 hardware evidence to the new generation.
