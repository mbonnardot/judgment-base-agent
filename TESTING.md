# Testing this yourself

Two ways to run the five example agents. **Path A** needs nothing but an API key
and is the right default. **Path B** runs the identical agent code against a
self-hosted DiffusionGemma-26B on an L4 GPU — that is the interesting one, and
it is what the numbers in the README were measured on.

The agents are byte-identical between the two paths. `TypeSafeBackend`
auto-delegates as soon as `DIFFUSIONGEMMA_JEV_URL` is set, so switching
backends is a comment in a `.env` file, not a code change.

---

## Common setup

```bash
git clone https://github.com/mbonnardot/judgment-base-agent.git
cd judgment-base-agent

# Recommended: uv, reproducible from the committed uv.lock
uv sync

# Or with pip (note the [eval] extra is mandatory -- see pyproject.toml)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

gcloud auth application-default login   # the example agents call Gemini too

cp examples/.env.example examples/.env
```

Then edit `examples/.env` and fill in `GOOGLE_CLOUD_PROJECT`.

Verify the library itself before touching any network:

```bash
uv run pytest -q     # 48 passed, 1 skipped, no credentials needed
```

The skipped one is the `heavy` marker: a real DiffusionGemma forward pass that
needs `torch` + `transformers>=5.8` and downloads a tiny model from Hugging
Face. Run it with `uv run pytest -m heavy` once those are installed.

---

## Path A — managed TypeSafe API

Put your key in `examples/.env`:

```bash
TYPESAFE_API_KEY=sk-...
```

```bash
uv run adk web examples --port 8008
```

Open <http://127.0.0.1:8008> and pick any of the five apps.

---

## Path B — self-hosted DiffusionGemma-26B on an L4

### What you need

| Requirement | Detail |
| :--- | :--- |
| GCP project | `medquad-assistant-capstone` (override with `DJEV_PROJECT`) |
| IAM | `roles/iap.tunnelResourceAccessor` **and** `roles/compute.viewer` |
| Why a tunnel | The VM has **no external IP** by design. IAP is the only way in. |
| Cost | `g2-standard-8` + 1× L4 ≈ **$0.71/hr** while RUNNING. Stop it when done. |

### 1. Open the tunnel

```bash
./scripts/connect_gpu.sh            # VM already running
./scripts/connect_gpu.sh --start    # also boot it if stopped (~5 min)
```

Leave it running in its own terminal. It exits non-zero with an explanation if
you lack permissions, if the port is taken, or if the model never comes up.

> [!NOTE]
> **The tunnel self-heals.** IAP tears down long-lived SSH sessions after a few
> hours (`client_loop: send disconnect: Broken pipe`), which would otherwise
> leave `adk web` pointed at a dead port with no error on the GPU side. The
> script detects the drop, releases the port, and reconnects — measured at
> 6–11 s. You will see `tunnel dropped` / `tunnel restored` in that terminal.


When it prints `==> Ready.` the endpoint is live at `http://127.0.0.1:8011`.

> [!WARNING]
> **The first judgment after a cold start is ~100× slower than steady state.**
> Measured on this VM: **30.6 s** for the first call, then **258 ms** for every
> call after it. That is CUDA graph capture plus an empty prefix cache, not the
> model's real latency.
>
> `connect_gpu.sh` fires a throwaway judgment to absorb this, so you should not
> see it. If you bypass the script, discard your first timing.


### 2. Point the agents at it

Uncomment the `Option B` block in `examples/.env`:

```bash
DIFFUSIONGEMMA_JEV_URL=http://127.0.0.1:8011
DIFFUSIONGEMMA_SYSTEM_ONE_PATH=/v1/systemone
DIFFUSIONGEMMA_MODEL_ID=RedHatAI/diffusiongemma-26B-A4B-it-NVFP4
```

### 3. Run

In a second terminal:

```bash
uv run adk web examples --port 8008
```

> [!IMPORTANT]
> `adk web` reads `examples/.env` **once at startup**. If you change which
> backend you are pointing at, restart it.

### 4. Confirm it really hit the GPU

Comparing to managed TypeSafe is the whole point, so verify rather than assume:

```bash
gcloud compute ssh djev-vllm-l4 --zone us-central1-a \
  --project medquad-assistant-capstone --tunnel-through-iap \
  --command "sudo tail -5 /var/log/djev-jev.log"
```

Each judgment logs one line:

```
systemone: q0=yes q1=yes q2=4 q3=no ... q14=3 reads=4 889ms
```

One line per *evaluation*, not per question — 15 judgments in that example
came back from a single canvas read.

---

## Reproducing the headline result

`review_triage_batch` puts 5 reviews × 3 criteria = **15 judgments through one
model call**, and is the app whose scores are directly comparable to managed Jev.

```bash
curl -sf -X POST http://127.0.0.1:8008/apps/review_triage_batch/users/demo/sessions/s1 \
  -H 'Content-Type: application/json' -d '{}'

curl -sS -X POST http://127.0.0.1:8008/run -H 'Content-Type: application/json' -d '{
  "appName":"review_triage_batch","userId":"demo","sessionId":"s1",
  "newMessage":{"role":"user","parts":[{"text":"Filter spam and rank the real bugs."}]}
}' | python3 -m json.tool | tail -40
```

Expected — spam blocked, praise skipped, three real bugs ranked by urgency:

| Review | Managed Jev | Self-hosted L4 |
| :--- | ---: | ---: |
| REV-101 Apple Pay crash | 3.00 | 3.00 |
| REV-105 Pro unlock delay | 2.00 | 2.01 |
| REV-103 dark-mode contrast | 1.19 | 1.26 |

REV-102 (crypto spam) and REV-104 (praise) are filtered out before ranking.

---

## Operating the VM

Both services run under systemd and are `enable`d, so starting the VM is
enough to bring the whole stack back.

```bash
# Start / stop (billing follows RUNNING state)
gcloud compute instances start djev-vllm-l4 --zone us-central1-a --project medquad-assistant-capstone
gcloud compute instances stop  djev-vllm-l4 --zone us-central1-a --project medquad-assistant-capstone

# Health
gcloud compute ssh djev-vllm-l4 --zone us-central1-a \
  --project medquad-assistant-capstone --tunnel-through-iap \
  --command "systemctl status djev-vllm djev-jev --no-pager | head -30"
```

| Unit | Role | Log |
| :--- | :--- | :--- |
| `djev-vllm` | vLLM engine, `127.0.0.1:8000` | `/var/log/djev-vllm.log` |
| `djev-jev` | Jev System One API, `127.0.0.1:8011` | `/var/log/djev-jev.log` |

`djev-jev` `Requires=` `djev-vllm` and blocks until the engine answers, so a
cold start takes ~5 minutes before the endpoint responds. That is weight
loading, not a hang.

Reinstall the units after rebuilding the VM:

```bash
gcloud compute scp --recurse deploy/diffusiongemma_jev/systemd \
  djev-vllm-l4:~/djev-systemd --zone us-central1-a \
  --project medquad-assistant-capstone --tunnel-through-iap
gcloud compute ssh djev-vllm-l4 --zone us-central1-a \
  --project medquad-assistant-capstone --tunnel-through-iap \
  --command "sudo bash ~/djev-systemd/install.sh"
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `connect_gpu.sh`: "cannot read djev-vllm-l4" | No project access | Ask for `roles/compute.viewer` + `roles/iap.tunnelResourceAccessor` |
| `connect_gpu.sh`: "already in use" | Old tunnel still alive | `lsof -iTCP:8011 -sTCP:LISTEN` then kill it |
| "endpoint never came up" | Engine still loading, or crashed | `sudo journalctl -u djev-vllm -n 50` on the VM |
| Agents answer but the GPU log is silent | `adk web` started before `.env` changed | Restart `adk web` |
| Everything is suspiciously fast | Still on managed TypeSafe | Check `DIFFUSIONGEMMA_JEV_URL` is actually set in the shell `adk web` sees |
| `HTTP 422 ... labels do not share one template slot` | `DIFFUSIONGEMMA_ALIAS_QUESTION_KEYS=false` against the PR's server | Leave it at the default `true` |

---

## Known limitations

Worth knowing before drawing conclusions from a demo.

- **The L4 has no native FP4.** Ada (sm_89) falls back to the Marlin kernel, so
  these latencies are a **floor, not a ceiling** — Blackwell would be faster.
- **Auto re-read dominates latency.** The server defaults to
  `mode:auto, threshold:0.1, max:4`, so a typical call does ~4 reads at ~272 ms
  rather than 1 read at ~96 ms. `samples` is not yet exposed on the backend,
  so **cost models must price expected reads, not one read.**
- **The vLLM fork is an unmerged PR** ([#57250](https://github.com/vllm-project/vllm/pull/57250)),
  pinned to commit `ceb8eebf3`. It is open, conflicted, and unreviewed. Treat
  the engine as experimental.
- **`ai_action_approval_gate` returns HTTP 500** after its judgment succeeds.
  The judgment itself is fine on the GPU; the downstream failure is undiagnosed.
- **`llm_as_a_judge_rubric` and `policy_fact_checker_loop`** have not been
  exercised against the GPU yet.
- **The tunnel is a workstation-local dev path**, not a deployment. Production
  needs an internal load balancer or Cloud Run with the GPU image.
