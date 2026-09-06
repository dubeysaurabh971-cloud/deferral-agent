# Deploying the demo

Two things live here: `space_app.py` (the demo with public-facing cost guards) and
`build_space.py` (assembles a deployable bundle and slims the index on the way out).

```bash
python -m deploy.build_space     # writes ./space_build/ — 98 MB, self-verifying
```

The build refuses to emit a bundle unless retrieval against the slimmed index is byte-identical
to the original: same chunk ids, same RRF scores to 8 decimal places.

## The memory number that decides everything

Measured on the real pipeline, RSS after loading and running one retrieval:

| stage | RSS |
|---|---:|
| baseline python | 21 MB |
| after imports (torch, chroma) | 74 MB |
| after retriever built | 647 MB |
| **after one retrieval** | **728 MB** |

The 74 → 647 MB jump is the retriever: sentence-transformer weights plus a BM25 index over
10,068 chunks. PyTorch alone is 528 MB installed. This is the constraint that rules hosts in
and out, and it cannot be reduced without changing `retrieval.py` — which would invalidate every
measured number in the root README, since retrieval determines what the gate sees.

## Hosting options, checked September 2026

| host | free RAM | verdict |
|---|---:|---|
| Fly.io shared-1x | 256 MB | ✗ |
| Render free web service | 512 MB | ✗ OOM |
| **Streamlit Community Cloud** | **1024 MB** | ✓ fits, ~25% headroom |
| Google Cloud Run | configurable | ✓ 360k GB-s/month free, needs a billing account |
| Hugging Face Spaces | — | ✗ Docker and Gradio now require PRO |

**Hugging Face no longer offers free Docker Spaces.** `create_repo(space_sdk="docker")` returns
`402 Payment Required` — "Static Spaces are free for everyone, but hosting Gradio and Docker
Spaces on free cpu-basic requires a PRO subscription." The `streamlit` SDK is no longer a valid
option at all. Only `static` is free, and it cannot run Python. The Dockerfile the build emits is
still correct; it just needs a host that will run it (Cloud Run, or a PRO Space).

## The recommended path: run it locally

```bash
streamlit run demo_app.py
```

For an interview this beats any free host — no cold start loading 728 MB while someone watches.
The public artifact that people actually click is the static explorer in `../docs/`, which needs
no API key, no backend, and cannot break.

## If you do want it hosted: Streamlit Community Cloud

It is the only free tier the measurement fits inside. It needs a **public GitHub repo** containing
the app and the index. Use a *separate* repo — 98 MB of index would make the main project
unpleasant to clone, and that repo exists to be read.

1. `python -m deploy.build_space`
2. Push `space_build/` to a new public repo. Every file is under 100 MB, so commit them directly —
   Streamlit Cloud handles Git LFS poorly, so plain files are safer than the `.gitattributes` the
   build writes for HF.
3. Connect the repo at share.streamlit.io, main file `app.py`.
4. Add secrets in the Streamlit dashboard (**not** in the repo):
   ```toml
   OPENAI_API_KEY = "sk-..."
   LLM_PROVIDER = "openai"
   OPENAI_MODEL = "gpt-5-mini"
   OPENAI_REASONING_EFFORT = "low"
   ```

Expect it to sit near the ceiling. 728 MB measured plus Streamlit's own overhead lands around
800–850 MB against a 1 GB cap. If it OOMs it restarts rather than dying permanently.

## Cost guards in `space_app.py`

Set from measured usage — 3.7–7.6K input and 0.6–1.8K output tokens per ticket, roughly $0.003 a
request on a mini-tier model.

| guard | value | why |
|---|---|---|
| `MAX_REQUESTS_PER_SESSION` | 8 | every example plus two of your own; ~$0.025/visitor |
| `MAX_TICKET_CHARS` | 600 | eval tickets are all under 200; bounds the input side |
| `MIN_SECONDS_BETWEEN` | 3 | stops a held-down key becoming a bill |
| `MAX_REQUESTS_PER_BOOT` | 300 | coarse backstop, ~$0.90 per container lifetime |

All are overridable by environment variable.

**`max_tokens` is deliberately not lowered.** `resolver.py` passes 4096, and changing it would
mean editing the pipeline this demo only observes. It would also save nothing: measured output ran
557–1808 tokens, so the cap never binds, and cutting it risks the truncation failure where a
reasoning model spends its whole budget on hidden reasoning and returns an empty `{}`.

The per-boot cap resets whenever the host sleeps and wakes, so treat it as a rate limiter, not a
budget. **The only real ceiling is a hard monthly spend limit on the provider dashboard.** Set one
before pointing anything public at your key.
