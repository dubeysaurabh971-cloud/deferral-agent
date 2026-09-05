# deferral-agent

A support-ticket RAG agent that decides whether to answer, ask, or hand off — and an honest
measurement of what that costs.

A retrieval agent that always answers is dangerous on exactly the tickets that matter: it
produces confident, well-cited prose for questions its knowledge base cannot answer. This
project builds a deferral gate on top of a naive RAG baseline and measures both sides of the
trade: what deferring buys, and what it costs.

**Short version: the gate cuts false resolutions from 58 to 4 — a 93% reduction — and takes
adversarial decision accuracy from 3.3% to 71.7%. It pays for that by deferring too readily,
refusing 51% of answerable tickets.** Whether that is a good trade depends on what a wrong
confident answer costs relative to an unnecessary handoff; the break-even is 1.19, and real
support economics are nowhere near that close. The over-deferral is a real defect with a
diagnosed cause, and this document covers both sides.

## The problem

The baseline (`NaiveResolver`) retrieves KB excerpts, answers from them, and reports `RESOLVE`
every time. Its answers are *grounded* — it does not invent facts — but it has no way to say "I
can't help with this." On the adversarial set it scores **3.3%** (2/60), and the two it gets
right are accidents: the only items whose expected decision is `RESOLVE`.

Its failure is invisible in the prose. It will write "the excerpts don't cover this" and mark the
ticket resolved in the same breath.

## Results

Both sets, same model (`gpt-5-mini`, `reasoning_effort=low`), one pass each.

| metric | baseline | gated | |
|---|---:|---:|---|
| **False resolutions** (answered when it should have deferred) | **58** | **4** | ✅ −93% |
| Adversarial decision accuracy (n=60) | 3.3% | **71.7%** | ✅ |
| Golden decision accuracy (n=100) | 100% | **49%** | ❌ the price |
| False escalation rate | 0% | **51%** | ❌ |
| Deferral precision (both sets) | — | 50.5% | |
| Raw decision accuracy, all 160 items | 63.7% | 57.5% | see below |

The baseline's 100% golden accuracy and 0% false-escalation rate are trivial, not virtuous: a
resolver that never defers cannot defer wrongly. It is a floor, not a competitor.

The last row deserves scepticism rather than a verdict. Raw accuracy counts a false resolution
and an unnecessary handoff as equally bad, which no support organisation would accept — a wrong
authoritative answer reaches the customer and generates a second ticket, while a needless handoff
costs a few minutes of staff time. Score the two error types separately and the picture inverts:

| error type (160 items) | baseline | gated |
|---|---:|---:|
| False resolutions | 58 | **4** |
| False escalations | 0 | 53 |
| Misrouted deferrals (right to defer, wrong lane) | 0 | 11 |

Letting a false resolution cost `C` times an unnecessary handoff, **the gate wins for any
C > 1.19.** At C=5 the baseline's error cost is 290 against the gate's 84. The honest caveat is
that this project never measured `C` — but it does not need to be measured precisely to clear
1.19.

By adversarial category:

| category | n | what it tests | baseline | gated |
|---|---:|---|---:|---:|
| ambiguous | 20 | underspecified; needs a question | 0% | **100%** |
| near_miss | 10 | topic covered, specific fact absent | 0% | **80%** |
| out_of_kb | 20 | KB has nothing on the subject | 0% | **65%** |
| injection | 10 | prompt injection wrapping a real request | 20% | 20% |
| **overall** | **60** | | **3.3%** | **71.7%** |

### Is this a good trade?

On error cost, yes, and not narrowly — see the break-even above. The gate eliminates 54 of the
baseline's 58 false resolutions, which is the failure this system exists to prevent.

On automation rate, no. At 51% false escalation half the answerable tickets still reach a human
and you are paying LLM inference for the privilege. The gate has learned it is allowed to decline
and has not learned when to stop. Both things are true, and shipping this would mean accepting a
deflection rate of roughly 49% in exchange for near-elimination of confidently wrong answers.

Note also what a single aggregate can hide in the other direction. Quoted alone, deferral
precision on the adversarial set is **96.4%** — when it defers there, it is nearly always right.
Across both sets it is **50.5%**, because the 51 unwarranted golden deferrals only enter the
denominator once answerable tickets are included. The flattering number is not wrong; it is just
not the whole picture.

## How it works

```
ticket → hybrid retrieval (BM25 + dense, RRF) → one structured LLM call → policy → decision
```

**Retrieval** (`src/retrieval.py`) is hybrid: BM25 keyword search and dense embeddings fused with
Reciprocal Rank Fusion over a 10,068-chunk KB built from the WixQA corpus. Hybrid beats
dense-only on support docs full of product names and error codes that embeddings blur together.

**The gate** (`src/gate.py`) is a single structured call producing the answer *and* an assessment
of it — coverage, missing detail, whether a human must act — in one pass. That matters: the
baseline's core bug was that its prose and its decision came from different code paths, so it
could contradict itself. Here they cannot diverge.

Three ordered steps, and **the order is load-bearing**:

1. **Underspecified?** → `CLARIFY`. Two tests must pass: would the right response change based on
   a detail not given, *and* would asking actually lead somewhere?
2. **Needs human authority?** → `ESCALATE`. Refunds, payouts, identity, ownership, bugs.
3. **Do the excerpts contain the specific fact?** → `RESOLVE`, else `ESCALATE`.

**The policy layer** (`apply_policy`) enforces invariants the model's decision may not violate.
Every rule is one-directional — each can only downgrade `RESOLVE` to a deferral, never upgrade a
deferral. That makes the gate safe to bolt on: it cannot introduce a false resolution the ungated
baseline would not already have made.

## Evaluation

- **Golden set** (100) — answerable tickets with reference answers, from WixQA's expert-written
  split. All expect `RESOLVE`. Measures the cost of deferring.
- **Adversarial set** (60) — hand-written, not LLM-generated, because the point is to encode
  judgement a model would not reliably produce on its own.

Decision accuracy is a pure label comparison: one call per item, no judge needed.

```bash
python -m src.eval.harness --resolver gated --no-judge   # decision metrics
python -m src.eval.sweep                                 # policy sweep, 0 API calls
```

## Findings

**1. Clarify-before-route is not clarify-before-coverage.** The first working gate put
underspecification first so "I want a refund for my recent purchase" would ask *which purchase*
rather than routing an unactionable ticket onward. Correct — but unqualified, it produced
`CLARIFY` for **45 of 60** adversarial items. Out-of-KB questions are broad, broad reads as
underspecified, and step 1 answered before step 3 could say the KB has nothing.

Adding a second test — *would the customer's reply change what happens next?* — moved adversarial
accuracy 56.7% → 71.7%: out_of_kb 25% → 65%, near_miss 60% → 80%, ambiguous held at 100%, false
resolutions flat at 4. Asking a question you cannot use the answer to is not caution; it costs a
round trip and still ends in a handoff.

**2. The same over-asking survives on answerable tickets, and that is where it costs.** Of 100
golden tickets the gate deferred 51: **38 CLARIFY, 13 ESCALATE**. The dominant mode is stark —
**22 of the 38 clarifications were on tickets the model itself labelled `kb_coverage="full"`.** It
could see the answer and asked a question anyway.

That is the mirror of finding 1. Step 1's TEST B now correctly stops it asking when the KB is
silent; TEST A is still too permissive when the KB is not. This is the single highest-value fix
remaining, and it is worth roughly 22 points of false escalation.

**3. The policy layer is very nearly inert.** `n_policy_overrides` is **1 across 160 items** — a
single `partial`-coverage downgrade. The sweep confirms it: flipping `escalate_on_partial` moves
golden accuracy 49% → 50% and adversarial not at all.

The reason is exact: every false resolution carried `kb_coverage="full"`, which the net is
structurally blind to. It can only catch a model that admits doubt, and this model's failure mode
is confident mislabeling. The gate's value comes almost entirely from the model's structured
self-assessment, not from the deterministic layer beneath it. The layer is cheap and
one-directional so it stays — but a design resting on it would be resting on nothing.

**4. Injection detection and injection disposition are different problems.** The gate flags
`injection_attempt_detected=True` on **10 of 10** injection items — perfect detection. It then
answers 8 of them with `CLARIFY` regardless of what the underlying request needed. Recognising an
attack is far easier than continuing to reason normally once you have.

**5. Retrieval scores carry no confidence signal.** An earlier approach thresholded on retrieval
score. RRF scores encode *rank*, not match quality — a chunk ranked #1 by both retrievers always
scores 0.0164, whether it answers the question or is merely the least-bad of 10,068. Measured,
near_miss tickets score *higher* than golden ones, and the best threshold tuned directly on the
test set reached only 72.5%. The judgement has to be made by something that reads the text.

## What I would do next

1. **Tighten TEST A on answerable tickets.** 22 full-coverage clarifications are the largest
   single block of error in the system.
2. **Fix injection disposition.** Detection is solved; grade the request underneath.
3. **Re-measure both sets together.** Adversarial-only iteration is what produced a gate that
   looks excellent on one set and loses to the baseline overall.

## Limitations

- **Single model, single run.** No seeds, repeats, or confidence intervals. Decision accuracy on
  60 items carries roughly ±12 points; category-level numbers (n=10–20) are directional only.
- **Golden measured for one prompt revision.** Adversarial has v1 and v3; golden has only v3, so
  the effect of the step-1 fix on false escalation is unmeasured. It may have made it worse.
- **Groundedness and correctness not measured at scale.** They ran on 6 golden items during
  development (groundedness 1.0, mean correctness 4.67/5) — a smoke signal, not a result.
- **The 160-item mix is arbitrary.** The combined 57.5% assumes a 100:60 answerable-to-adversarial
  ratio. Real traffic has its own ratio, and the comparison moves with it.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # add a key; set LLM_PROVIDER
python run.py --ingest        # download WixQA, chunk, embed (~5 min first run)
python run.py                 # interactive: paste a ticket
```

Providers: Anthropic, OpenAI, Google (Gemini), xAI (Grok). All but Anthropic share one
OpenAI-wire code path in `src/llm_client.py`; swapping is a config change, not a rewrite.

## Cost

The eval is deliberately cheap. Decision accuracy needs no LLM judge, and the baseline arm needs
no API calls at all — its decision policy is a pure function, scored offline. Every report records
its own `token_spend`. The full result above (160 gated items plus two earlier prompt revisions)
cost roughly **$0.55**.

On a reasoning model, hidden reasoning tokens share the completion budget and bill at the output
rate: at `max_tokens=1536` the structured call returned `{}` — a successful, billed, empty
response. `OPENAI_REASONING_EFFORT` and a 4096-token budget address this.

`src/eval/sweep.py` replays the policy over cached traces at zero cost, so tuning
`escalate_on_partial` never needs a second paid run.
