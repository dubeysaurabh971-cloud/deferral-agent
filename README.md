# deferral-agent

[![tests](https://github.com/dubeysaurabh971-cloud/deferral-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/dubeysaurabh971-cloud/deferral-agent/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A support-ticket RAG agent that decides whether to answer, ask, or hand off — and an honest
measurement of what that costs.

**[Browse all 160 decisions →](https://dubeysaurabh971-cloud.github.io/deferral-agent/)** — every
ticket the gate answered, asked about, or handed off, with its reasoning and coverage call. No API
key, no backend; generated from the trace log.

> **What this proves.** A single structured self-assessment call can near-eliminate confidently
> wrong answers in support RAG — false resolutions **58 → 5** in the shipped configuration, 4 with
> the reviewer off — and this trade-off has to be
> scored on asymmetric error cost, not raw accuracy: a wrong authoritative answer and a needless
> handoff are not the same mistake.
>
> **What I would ship.** Not this build. My bar is false escalation **under ~15%** while holding
> false resolutions in low single digits; the best configuration here reaches **44%**, so it
> deflects too little to deploy. Getting there took two attempts: a prompt fix aimed straight at
> the problem made it *worse* (finding 3), and only a second-stage mechanism moved it (finding 4).
> The gap between those two attempts is the most useful thing in this repo.
>
> **What it depends on.** Every claim here is conditional on `C`, the cost of a wrong confident
> answer in units of an unnecessary handoff. The gate beats the baseline for C > 1.19; the shipped
> configuration is the right one only for 1.75 < C < 7. That window is stated, not assumed away.
>
> The self-critique below is deliberate. Read it as scope of what was measured, not as a verdict
> that the approach failed.

A retrieval agent that always answers is dangerous on exactly the tickets that matter: it
produces confident, well-cited prose for questions its knowledge base cannot answer. This
project builds a deferral gate on top of a naive RAG baseline and measures both sides of the
trade: what deferring buys, and what it costs.

**Short version: the gate cuts false resolutions from 58 to 5 — a 91% reduction, or to 4 with the
clarification reviewer disabled — and takes
adversarial decision accuracy from 3.3% to 71.7%. It pays for that by deferring too readily,
refusing 44% of answerable tickets even in its best configuration.** Whether that is a good trade
depends on what a wrong confident answer costs relative to an unnecessary handoff. That
break-even is 1.19, computed from the measured error counts; the ratio itself this project does
not measure. The over-deferral is a real defect with a diagnosed cause, and this document covers
both sides.

## The problem

The baseline (`NaiveResolver`) retrieves KB excerpts, answers from them, and reports `RESOLVE`
every time. Its answers are *grounded* — it does not invent facts — but it has no way to say "I
can't help with this." On the adversarial set it scores **3.3%** (2/60), and the two it gets
right are accidents: the only items whose expected decision is `RESOLVE`.

Its failure is invisible in the prose. It will write "the excerpts don't cover this" and mark the
ticket resolved in the same breath.

## Results

Both sets, same model (`gpt-5-mini`, `reasoning_effort=low`), one pass each.

| metric | baseline | gate | gate + reviewer |
|---|---:|---:|---:|
| **False resolutions** (answered when it should have deferred) | 58 | **4** | 5 |
| False escalations (deferred an answerable ticket) | 0 | 53 | **46** |
| Misrouted deferrals (right to defer, wrong lane) | 0 | 11 | 11 |
| Adversarial decision accuracy (n=60) | 3.3% | **71.7%** | 70.0% |
| Golden decision accuracy (n=100) | 100% | 49% | **56%** |
| Raw decision accuracy (all 160) | **63.7%** | 57.5% | 61.3% |
| Deferral precision (both sets) | — | 50.5% | 53.5% |

The third column adds a second-stage **clarification reviewer** (finding 4): a separate call that
sees only CLARIFY decisions and asks whether that question was necessary. It buys seven fewer
false escalations for one extra false resolution.

The baseline's 100% golden accuracy and 0% false-escalation rate are trivial, not virtuous: a
resolver that never defers cannot defer wrongly. It is a floor, not a competitor.

**Raw decision accuracy is the wrong metric here, and it appears in that table only to be argued
with.** It counts a false resolution and an unnecessary handoff as equally bad — a wrong
authoritative answer reaches the customer, gets acted on and tends to generate a second contact,
while a needless handoff costs staff minutes. On raw accuracy the baseline "wins" at 63.7%, which
is how a system that cannot decline at all beats one that can.

Deferral precision needs the same care in the other direction. On the adversarial set alone the
gate+reviewer scores **96.4%** — when it defers there, it is nearly always right. Across both sets
it is **53.5%**, because the 46 unwarranted deferrals (44 answerable golden tickets, plus 2
injection tickets that should have been resolved) only enter the denominator once answerable
tickets are counted. Both numbers are true; only the second is honest on its own.

### Choosing a configuration is an economics question

Let `C` be the cost of one false resolution in units of one unnecessary handoff. Total error cost
on these 160 items is linear in `C`, and the three configurations rank differently depending on it:

| configuration | error cost | C=1 | C=2 | C=3 | C=5 | C=7 | C=10 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | 58C | 58 | 116 | 174 | 290 | 406 | 580 |
| gate alone | 4C + 64 | 68 | 72 | 76 | 84 | **92** | **104** |
| + reviewer, ungoverned | 9C + 50 | **59** | 68 | 77 | 95 | 113 | 140 |
| + reviewer, governed | 5C + 57 | 62 | **67** | **72** | **82** | **92** | 107 |

The ungoverned row is the only one whose error counts are not in a table above: it makes **9**
false resolutions and **43** false escalations, and its 4 extra flips land on tickets that were
already-wrong deferrals, so misrouted drops 11 → 7. Hence 9C + 50 rather than 9C + 54.

Three things fall out of this, and only the first was obvious in advance:

1. **The gate beats the baseline for any C > 1.19.** That threshold is computed from the measured
   error counts; `C` itself is not measured anywhere in this project. Whether real support
   economics clear 1.19 is an assumption I find easy to believe but did not test — settling it
   needs deflection-cost and bad-answer-cost figures from whoever owns the queue.
2. **The ungoverned reviewer wins on raw accuracy (59% golden) and is the wrong choice above
   C = 2.** It buys golden accuracy with the exact error the system exists to prevent. Selecting
   on accuracy would have selected it.
3. **The governed reviewer is only the best choice for 1.75 < C < 7.** Below 1.75 the ungoverned
   variant is better; above 7 the gate alone is better, because one extra false resolution stops
   being worth seven fewer handoffs.

That last point constrains the default. `GatedResolver` runs the reviewer unless told otherwise,
which **assumes C < 7** — and an organisation that treats a confidently wrong answer as ten times
worse than a needless handoff is outside that window and should turn it off. So the ratio is a
constructor argument rather than a hardcoded belief:

```python
GatedResolver(cost_ratio=3)      # reviewer on  -- bad answers are cheap-ish
GatedResolver(cost_ratio=10)     # reviewer off -- bad answers are expensive
GatedResolver()                  # reviewer on  -- the C < 7 default
```

`review.REVIEW_BREAK_EVEN_C` holds the 7.0, derived from the two cost curves above and pinned by a
test. It is measured on this eval set, not a general constant, and it inherits the wide confidence
intervals of a single 160-item run.

By adversarial category:

| category | n | what it tests | baseline | gate | + reviewer |
|---|---:|---|---:|---:|---:|
| ambiguous | 20 | underspecified; needs a question | 0% | **100%** | 95% |
| near_miss | 10 | topic covered, specific fact absent | 0% | **80%** | **80%** |
| out_of_kb | 20 | KB has nothing on the subject | 0% | **65%** | **65%** |
| injection | 10 | prompt injection wrapping a real request | 20% | 20% | 20% |
| **overall** | **60** | | **3.3%** | **71.7%** | 70.0% |

### Is this a good trade?

On error cost, yes, within the window above. The shipped configuration eliminates 53 of the
baseline's 58 false
resolutions, which is the failure this system exists to prevent.

On automation rate, no. Even with the reviewer, 44% of answerable tickets still reach a human and
you are paying LLM inference for the privilege. The gate has learned it is allowed to decline and
has only partly learned when to stop. Shipping this would mean accepting a deflection rate of
roughly 56% in exchange for near-elimination of confidently wrong answers.

## How it works

```
ticket → hybrid retrieval (BM25 + dense, RRF) → structured LLM call → policy ─┬→ RESOLVE / ESCALATE
                                                                             └→ CLARIFY → reviewer → RESOLVE?
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

**The clarification reviewer** (`src/review.py`) is a second call that sees only CLARIFY
decisions, and is asked one question: was that clarification necessary? It is the mirror of the
policy layer — one-directional in the opposite direction, able only to turn CLARIFY into RESOLVE,
never to create a deferral. Two guards keep it honest, both of them measured rather than assumed:

- Any flip is put back through `apply_policy`. The reviewer cannot launder a resolution past an
  invariant the system already declared — without this it produced 5 extra false resolutions
  instead of 1.
- Tickets carrying a detected injection attempt are never reviewed. A ticket containing an active
  manipulation attempt is the last place to relax caution.

It also refuses to flip without producing a replacement answer, since a RESOLVE whose text is
still a clarifying question would be worse than the deferral it replaced.

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
resolutions flat at 4, and injection *down* 30% → 20%. That last one is a regression, one item at
n=10, and it is the price of the other three. Asking a question you cannot use the answer to is not caution; it costs a
round trip and still ends in a handoff.

**2. The same over-asking survives on answerable tickets, and that is where it costs.** Of 100
golden tickets the gate deferred 51: **38 CLARIFY, 13 ESCALATE**. The dominant mode is stark —
**22 of the 38 clarifications were on tickets the model itself labelled `kb_coverage="full"`.** It
could see the answer and asked a question anyway.

That is the mirror of finding 1. Step 1's TEST B now correctly stops it asking when the KB is
silent; TEST A is still too permissive when the KB is not. On paper this is the
highest-value fix remaining, worth roughly 22 points of false escalation — which is exactly what
finding 3 set out to collect, and did not.

**3. The obvious fix for finding 2 made it worse.** TEST A
let the model invent a plausible follow-up question rather than find a necessary one, so the
revision required it to *name the fork*: state two specific competing answers the excerpts
actually support, plus the detail selecting between them. If it could not name two, the ticket
was not underspecified.

Measured on both sets against the previous prompt:

| | v3 | v4 (the "fix") |
|---|---:|---:|
| Golden decision accuracy | **49%** | 42% |
| False escalations | **53** | 60 |
| False resolutions | **4** | 6 |
| Golden CLARIFY count | **38** | 45 |
| Adversarial accuracy | 71.7% | 71.7% |

A change aimed squarely at over-clarification produced *more* of it, and left adversarial exactly
where it was. The plausible reading is that a stricter-sounding bar is still a bar the model can
argue it has cleared, while the extra paragraph of instruction about when to clarify raised the
salience of clarifying rather than its cost. Injection moved 20% → 30% and out_of_kb 65% → 60%,
one item each — noise at n=10 and n=20, not worth 7 points of golden accuracy.

**Reverted.** The full v4 report is kept in `eval_results/gated_both_160_promptv4.json`.

The conclusion is not "prompt engineering does not work" — findings 1 and 2 were both won with
prompt changes. It is narrower and more useful: **this particular miscalibration does not respond
to being told about itself.** Getting the deferral threshold right probably needs a mechanism that
does not depend on the model's own judgement of its own judgement — few-shot examples of correct
RESOLVE decisions, a calibrated threshold on something external to the model, or a second pass
that only reviews clarification decisions. That is a design change, not a wording change, and it
is where I would start next.

**4. What did work: taking the judgement out of the prompt and giving it its own call.**
Finding 3 ruled out the wording fix, so the next attempt was structural. `src/review.py` adds a
second call that sees only CLARIFY decisions and is asked exactly one question — was this
clarification necessary? No coverage label, no escalation, no injection handling, nothing to
trade against.

| | gate alone | + reviewer |
|---|---:|---:|
| Golden decision accuracy | 49% | **56%** |
| False escalations | 53 | **46** |
| False resolutions | 4 | 5 |
| Adversarial accuracy | 71.7% | 70.0% |

Seven points of golden accuracy for one extra false resolution, where the prompt-level attempt at
the same goal lost seven. The difference is isolation: the same judgement the model got wrong as
one paragraph inside a 2,000-token prompt, it got substantially right as a 400-token prompt with
a single question in it. If a model is ignoring an instruction that competes with five others,
the fix may be to stop making it compete rather than to word it more forcefully.

**The governance mattered more than the mechanism.** Ungoverned, the reviewer flipped 15
clarifications and produced 5 extra false resolutions; it wins on raw accuracy (59% golden) and is
the wrong choice for any cost ratio above 2. Routing each flip back through `apply_policy` — the
reviewer may not resolve where the gate itself would have been forbidden to — cut that to 8 flips
and 1 extra false resolution. **Accuracy would have selected the worse system.**

**5. The policy layer is very nearly inert.** `n_policy_overrides` is **1 across 160 items** — a
single `partial`-coverage downgrade. The sweep confirms it: flipping `escalate_on_partial` moves
golden accuracy 49% → 50% and adversarial not at all.

The reason is exact: every false resolution carried `kb_coverage="full"`, which the net is
structurally blind to. It can only catch a model that admits doubt, and this model's failure mode
is confident mislabeling. The gate's value comes almost entirely from the model's structured
self-assessment, not from the deterministic layer beneath it. The layer is cheap and
one-directional so it stays — but a design resting on it would be resting on nothing.

**6. Injection detection and injection disposition are different problems.** The gate flags
`injection_attempt_detected=True` on **10 of 10** injection items — perfect detection. It then
answers 8 of them with `CLARIFY` regardless of what the underlying request needed. Recognising an
attack is far easier than continuing to reason normally once you have.

**7. Retrieval scores carry no confidence signal.** An earlier approach thresholded on retrieval
score. RRF scores encode *rank*, not match quality — a chunk ranked #1 by both retrievers always
scores 0.0164, whether it answers the question or is merely the least-bad of 10,068. Measured,
near_miss tickets score *higher* than golden ones, and the best threshold tuned directly on the
test set reached only 72.5%. The judgement has to be made by something that reads the text.

## What I would do next

1. **Keep pushing on over-deferral.** The reviewer took it from 53 false escalations to 46; the
   ship bar needs roughly 15. The next candidates are few-shot examples of correct RESOLVE
   decisions drawn from the golden set, and a reviewer that sees the *reference* answer format
   rather than only the excerpts.
2. **Fix injection disposition.** Detection is solved (10/10); disposition is not (2/10). Making
   it a procedure — restate the request with instructions stripped, then grade the restatement —
   moved it one item, which is nothing at n=10. Needs a bigger injection set before it can even
   be measured properly.
3. **Get a real cost ratio.** The break-even is 1.19 and the whole ship/no-ship argument turns on
   whether reality clears it. That is a question for whoever owns the support queue, not a
   modelling question.
4. **Always measure both sets.** Two of the four prompt revisions here were tuned on adversarial
   alone. Only one of them was ever checked against golden afterwards — and when it was, golden had
   moved 7 points in the direction I was not watching. The other one's golden effect is still
   unmeasured, which is its own answer.

## Limitations

- **Single model, single run.** No seeds, repeats, or confidence intervals. Decision accuracy on
  60 items carries roughly ±12 points; category-level numbers (n=10–20) are directional only.
- **The reviewer's numbers come from a replay, not a fresh end-to-end run.** The 72 reviewer calls
  were real, against the live model, but they were made over the base gate's *cached* decisions
  rather than re-running both stages together. Because the reviewer only reads the gate's output
  and never feeds back into it, the composition is exact — what a fresh run would add is
  run-to-run variance, which this project does not measure anywhere. The replay is
  `src/eval/review_replay.py`, its output `eval_results/clarify_review_replay.json`.
- **The reviewer's two guards were chosen after seeing which flips failed.** Both have an argument
  from first principles — respect the invariants the policy layer already enforces; do not relax
  caution on a ticket carrying an active attack — but I did not write them down before looking at
  the data, and with only 15 flips to learn from, some fitting to this eval set is likely. They
  need a fresh adversarial set to be trusted.
- **Prompt revisions measured unevenly.** v1 and v3 have adversarial numbers; only v3 and v4 were
  measured on both sets. The effect of the v1 → v3 step-1 fix on false escalation was never
  measured, and given what v4 did, it may well have made it worse.
- **Groundedness and correctness not measured at scale.** They ran on 6 golden items during
  development (groundedness 1.0, mean correctness 4.67/5) — a smoke signal, not a result.
- **The 160-item mix is arbitrary.** Every combined figure — raw accuracy, deferral precision, and
  the cost curves that set the 1.19 and 7.0 thresholds — assumes a 100:60 answerable-to-adversarial
  ratio. Real traffic has its own ratio, and all of it moves with it.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # add a key; set LLM_PROVIDER
python run.py --ingest        # download WixQA, chunk, embed (~5 min first run)
python run.py                 # interactive: paste a ticket, get a decision
python run.py --compare       # baseline and gate on the same ticket, side by side
```

`--compare` is the quickest way to see what this project is about: paste an out-of-KB ticket and
watch the baseline answer it confidently with citations while the gate escalates. `--resolver
naive` runs the ungated baseline alone.

Providers: Anthropic, OpenAI, Google (Gemini), xAI (Grok). All but Anthropic share one
OpenAI-wire code path in `src/llm_client.py`; swapping is a config change, not a rewrite.

There is also a local web demo that shows the full decision trace for a pasted ticket, and lets
you toggle the clarification reviewer on and off to watch a decision change:

```bash
streamlit run demo_app.py
```

Deployment notes, measured memory, and which free hosts can actually run this: `deploy/README.md`.

## Cost

The eval is deliberately cheap. Decision accuracy needs no LLM judge, and the baseline arm needs
no API calls at all — its decision policy is a pure function, scored offline. Every report records
its own `token_spend`. The whole project — four prompt revisions, both eval sets, and the reviewer
replay — came to about **$1.10**. The reviewer adds one call per CLARIFY rather than a second pass
over every ticket: 72 calls, ~196K input + ~37K output.

On a reasoning model, hidden reasoning tokens share the completion budget and bill at the output
rate: at `max_tokens=1536` the structured call returned `{}` — a successful, billed, empty
response. `OPENAI_REASONING_EFFORT` and a 4096-token budget address this.

`src/eval/sweep.py` replays the policy over cached traces at zero cost, so tuning
`escalate_on_partial` never needs a second paid run.
