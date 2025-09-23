## FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning

We implement some key modifications base on the verl framework, including:

1. Asynchronize rollout and reward: we launch the reward model as an external service, and realize sample-wise async, with detailed implementation in `verl/experimental/agent_reward_loop`

2. We extract the most related code about FAPO algorithm in `fapo/` for reference, including `fapo/fapo_genrm` and `fapo/fapo_reasoning`. Corresponding training scripts of FAPO-GenRM and FAPO-Reasoning (and Baselines) are placed in `scripts/`.

### Step 1: Train FAPO-GenRM

Due to the file size limit, we only upload first 100 rows in `example_data/fapo-critic.jsonl` (convert to jsonl for better readability).

```bash
bash scripts/run_fapo_genrm_4b.sh
```

### Step 2: Train FAPO-Reasoning

#### Step 2.1: Launch GenRM as an External Service

```bash
# first launch multiple genrm servers
bash scripts/launch_server.sh

# launch a router to manage data_parallel genrm servers
# so that the request should be sent to the router
# then the router will distribute the request to the corresponding genrm server
bash scripts/launch_router.sh
```

#### Step 2.2: Train

```bash
# Note that you should specify the router address
# in the `fapo/fapo_reasoning/reward_fn.py`

# Train Baseline Models
bash scripts/run_baseline_reasoning_7b.sh
bash scripts/run_baseline_reasoning_32b.sh

# Train FAPO Models
bash scripts/run_fapo_reasoning_7b.sh
bash scripts/run_fapo_reasoning_32b.sh
```

