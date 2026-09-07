# deferral-agent

[![tests](https://github.com/dubeysaurabh971-cloud/deferral-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/dubeysaurabh971-cloud/deferral-agent/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A support-ticket RAG agent that decides whether to answer, ask, or hand off — and an honest
measurement of what that costs.

**[Browse all 160 decisions →](https://dubeysaurabh971-cloud.github.io/deferral-agent/)** — every
ticket the gate answered, asked about, or handed off, with its reasoning and coverage call. No API
key, no backend; generated from the trace log.

> **What this proves.** A single structured self-assessment call can near-eliminate confidently
> wrong answers in support RAG — false resolutions **58 → 12** — and this trade-off has to be
> scored on asymmetric error cost, not raw accuracy: a wrong authoritative answer and a needless
> handoff are not the same mistake.
>
> **What I would ship.** The v5 configuration, which clears the bar this README used to fail.
> False escalation is **12%** (range 9–14% over three runs) against a stated bar of ~15%, with
> false resolutions no worse than the configuration it replaces and adversarial accuracy up 6
> points. It beats the previous build on *both* error types at once rather than trading between
> them.
>
> **How it got there — the part worth reading.** Two earlier attempts had failed: a prompt fix
> aimed straight at over-deferral made it *worse* (finding 3), and a second-stage reviewer moved
> it only 7 points (finding 4). What worked was different in kind. Over-deferral turned out to be
> three separate defects wearing one number, and only one of them was in the gate's judgement:
>
> 1. Three of the gate's own definitions were wrong, not merely too cautious (finding 8).
> 2. **Retrieval, not caution, set the floor.** Recall@5 against the golden set's own reference
>    articles was **76%** — for 24 of 100 answerable tickets the answer was never in the context,
>    so no correctly-calibrated gate could have answered them. 15% was arithmetically out of
>    reach until `top_k` moved (finding 9). This is the finding I would lead with.
> 3. The reviewer from finding 4 is now **off by default**: it was compensating for a defect that
>    is now fixed, and a compensator whose input disappears does not go neutral, it inverts
>    (finding 10).
>
> **What it depends on.** Every claim here is conditional on `C`, the cost of a wrong confident
> answer in units of an unnecessary handoff. v5 beats the baseline for C > 0.36 and beats the old
> build at every C. The one genuine choice left is `top_k`, which trades the two error types
> against each other at C ≈ 3.5 — stated below, not assumed away.
>
> The self-critique below is deliberate. Read it as scope of what was measured, not as a verdict
> that the approach failed.

A retrieval agent that always answers is dangerous on exactly the tickets that matter: it
produces confident, well-cited prose for questions its knowledge base cannot answer. This
project builds a deferral gate on top of a naive RAG baseline and measures both sides of the
trade: what deferring buys, and what it costs.

**Short version: the gate cuts false resolutions from 58 to 12 — a 79% reduction — takes
adversarial decision accuracy from 3.3% to 71%, and refuses 12% of answerable tickets.** The
first two numbers are what a deferral gate is for. The third is the one that took three
iterations, because the obvious readings of it were wrong: it was not one defect but three, and
the largest was in retrieval rather than in the gate. Whether the whole trade is worth making
depends on what a wrong confident answer costs relative to an unnecessary handoff; that
break-even is 0.36, computed from the measured error counts, and the ratio itself this project
does not measure.

## The problem

The baseline (`NaiveResolver`) retrieves KB excerpts, answers from them, and reports `RESOLVE`
every time. Its answers are *grounded* — it does not invent facts — but it has no way to say "I
can't help with this." On the adversarial set it scores **3.3%** (2/60), and the two it gets
right are accidents: the only items whose expected decision is `RESOLVE`.

Its failure is invisible in the prose. It will write "the excerpts don't cover this" and mark the
ticket resolved in the same breath.

## Results

Both sets, same model (`gpt-5-mini`, `reasoning_effort=low`). The v5 columns are **means over
three complete 160-item runs**, with the per-run range in brackets; earlier columns are single
passes, which is part of why they need reading with care (see *Run-to-run variance* below).

| metric | baseline | v3 + reviewer<br>*(old ship)* | v5 gate<br>`top_k=5` | **v5 gate<br>`top_k=10`**<br>*(ships)* |
|---|---:|---:|---:|---:|
| **False resolutions** (answered when it should have deferred) | 58 | 12 | **9.3** [9–10] | 11.7 [10–14] |
| False escalations (deferred an answerable ticket) | 0 | 37 | 19.7 [18–22] | **12** [9–14] |
| Misrouted deferrals (right to defer, wrong lane) | 0 | 7 | 5.3 [5–6] | **4.7** [4–5] |
| **False escalation rate** | 0% | 37% | 20% | **12%** [9–14%] |
| Adversarial decision accuracy (n=60) | 3.3% | 65% | **74%** | 71% [68–73%] |
| Golden decision accuracy (n=100) | 100% | 63% | 80% | **88%** [86–91%] |
| Raw decision accuracy (all 160) | 63.7% | 63.7% | 78% | **82%** [79–84%] |
| Deferral precision (both sets) | — | 54% | 71% | **78%** |
| Error cost | 58C | 12C + 44 | 9.3C + 25 | **11.7C + 16.7** |
| LLM calls per ticket | 1 | 1–2 | **1** | **1** |

The second column is the configuration this README used to ship, **re-measured here rather than
quoted**: its published figures were 44% false escalation and 5 false resolutions, and a fresh run
of the identical code gave 37% and 12. Nothing changed but the run. That discrepancy is the
subject of *Run-to-run variance*, and it is why the v5 columns are averaged.

**v5 at `top_k=10` beats the old build on every error type simultaneously** — fewer false
resolutions (11.7 vs 12), 25 fewer deferral errors, 6 points more adversarial accuracy, and one
LLM call instead of up to two. That is not a trade-off being re-balanced; it is the frontier
moving. The `top_k=5` column is kept because it is a genuinely different point on that frontier
(fewer false resolutions, more handoffs) and the choice between them is an economics question,
addressed below.

The baseline's 100% golden accuracy and 0% false-escalation rate are trivial, not virtuous: a
resolver that never defers cannot defer wrongly. It is a floor, not a competitor.

**Raw decision accuracy is the wrong metric here, and it is in that table only to be argued
with.** It counts a false resolution and an unnecessary handoff as equally bad — a wrong
authoritative answer reaches the customer, gets acted on and tends to generate a second contact,
while a needless handoff costs staff minutes. Against the old build it was the metric that most
flattered the *baseline*, which tied it exactly: **63.7% either way**, whether or not the system
could decline at all. v5 reaches 82% and so no longer needs the argument made on its behalf —
which is precisely why the argument is worth keeping. It was true when it was inconvenient.

Deferral precision needs the same care in the other direction. On the adversarial set alone v5
scores **98%** — when it defers there, it is almost always right. Across both sets it is **78%**,
because the unwarranted deferrals (12 answerable golden tickets, plus the injection tickets that
should have been resolved) only enter the denominator once answerable tickets are counted. Both
numbers are true; only the second is honest on its own.

### Run-to-run variance, and why it is stated first

Every headline number here comes from a 160-item run against a hosted model at
`temperature` default, and a single run moves by several points for no reason at all. The
sharpest evidence is not an estimate but an accident: **re-measuring the unchanged v3+reviewer
build gave 37% false escalation and 12 false resolutions where this README had recorded 44% and
5.** Same code, same model, same items.

So the v5 figures are means over three complete runs with the range shown, and the observed spread
is roughly **±3 points on false escalation** and **±2 on false-resolution counts**. Two
consequences, both applied throughout:

- A change of under ~4 points on a single run is not evidence of anything. Several intermediate
  prompt revisions during v5 moved less than that and were kept or dropped on the reasoning
  behind them, not the number.
- Category figures at n=10 (`near_miss`, `injection`) move 10–20 points between identical runs.
  They are directional only, and `injection` at n=10 cannot support any claim finer than "it got
  much better" (10% → 50%).

`src/eval/aggregate.py` computes these means and **refuses to average any run that lost items**,
because every rate is a fraction over the items that survived.

### Choosing a configuration is an economics question

Let `C` be the cost of one false resolution in units of one unnecessary handoff. Total error cost
on these 160 items is linear in `C`, and the configurations rank differently depending on it:

| configuration | error cost | C=1 | C=2 | C=3.5 | C=5 | C=10 |
|---|---|---:|---:|---:|---:|---:|
| baseline | 58C | 58 | 116 | 203 | 290 | 580 |
| v3 + reviewer *(old ship)* | 12C + 44 | 56 | 68 | 86 | 104 | 164 |
| **v5 gate, `top_k=10`** *(ships)* | 11.7C + 16.7 | **28** | **40** | **58** | 75 | 134 |
| v5 gate, `top_k=5` | 9.3C + 25 | 34 | 44 | **58** | **72** | **118** |
| v5 gate + reviewer, `top_k=10` | 15C + 13 | 28 | 43 | 66 | 88 | 163 |

Four things fall out, and only the first was obvious in advance:

1. **v5 beats the baseline for any C > 0.36**, down from 1.19 for the old build. That threshold is
   computed from the measured error counts; `C` itself is not measured anywhere in this project.
   Settling it needs deflection-cost and bad-answer-cost figures from whoever owns the queue.
2. **v5 beats the old build at every C**, because it is better on both error types at once. No
   window needs stating for that comparison, which was not true of any previous version.
3. **The one real remaining choice is `top_k`, and it crosses over at C ≈ 3.5.** More context
   recovers answerable tickets and also hands the model more adjacent material to mistake for an
   answer. Below 3.5, take the deflection; above it, take the caution. This is the honest shape
   of the trade and it does not go away.
4. **The reviewer is now off by default** — a reversal, and finding 10.

So the ratio stays a constructor argument rather than a hardcoded belief:

```python
GatedResolver()                  # ships: reviewer off
GatedResolver(cost_ratio=1.2)    # reviewer on  -- bad answers are barely worse than handoffs
GatedResolver(cost_ratio=10)     # reviewer off -- bad answers are expensive
```

`review.REVIEW_BREAK_EVEN_C` holds the 1.6 at which the reviewer stops paying, and
`REVIEW_BREAK_EVEN_C_V3` keeps the old 7.0 so finding 4 stays checkable. Both are measured on this
eval set, not general constants.

By adversarial category (v5 columns are means over three runs):

| category | n | what it tests | baseline | v3+rev | v5 `k=5` | v5 `k=10` |
|---|---:|---|---:|---:|---:|---:|
| ambiguous | 20 | underspecified; needs a question | 0% | 90% | 87% | 87% |
| near_miss | 10 | topic covered, specific fact absent | 0% | 70% | **100%** | 73% |
| out_of_kb | 20 | KB has nothing on the subject | 0% | **65%** | 62% | **65%** |
| injection | 10 | prompt injection wrapping a real request | 20% | 10% | 47% | **50%** |
| **overall** | **60** | | **3.3%** | 65% | **74%** | 71% |

`near_miss` is where the `top_k` trade is most visible: at `top_k=5` it is perfect and at
`top_k=10` it is not, because the compound "what happens when X meets Y" questions get more
material that looks like an answer. At n=10 that is 3 items, inside the run-to-run spread, so it
is a direction rather than a measurement.

### Is this a good trade?

On error cost, yes, and no longer only inside a window: v5 eliminates 46 of the baseline's 58
false resolutions *and* refuses only 12% of answerable tickets, which is a better position than
the old build on both counts.

On automation rate, it now clears the bar this README set for itself — 12% against ~15% — with the
important caveat that **the remaining 12% is mostly not the gate's fault.** Retrieval recall is
88% at `top_k=10`, so roughly 12 of 100 answerable tickets still arrive without their reference
article in context. A gate that answered those would be guessing. That coincidence of numbers is
not a proof that the gate is now perfectly calibrated, but it does mean further false-escalation
gains have to come from retrieval, not from the prompt.

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

1. **Is a fact only the customer holds missing?** → `CLARIFY`. A bare ticket naming no instance of
   anything is asked about; a specific ticket is answered. Asking the customer which of *our*
   procedures or *our* interfaces applies is categorically excluded — that is our triage, not
   their missing information.
2. **Must a human act on this account?** → `ESCALATE`. Defined by the *act*, not the topic:
   performing a refund or payout, looking into their specific case, reporting a defect. Asking how
   a refund works, or where the button is, is a documentation question.
3. **Do the excerpts state what was asked?** → `RESOLVE`, else `ESCALATE`. Collecting stated facts
   from several excerpts is full coverage; *deriving* an unstated interaction from two of them is
   not, and neither is reading a conclusion out of the excerpts' silence.

Each of those three sentences is a definition that v5 changed, and each is worth 10–30 points on
one eval set or the other. Finding 8 has the details and the counts.

**The policy layer** (`apply_policy`) enforces invariants the model's decision may not violate.
Every rule is one-directional — each can only downgrade `RESOLVE` to a deferral, never upgrade a
deferral. That makes the gate safe to bolt on: it cannot introduce a false resolution the ungated
baseline would not already have made. Four rules: human authority, absent coverage, partial
coverage (the `escalate_on_partial` knob), and **a claim of full coverage that points at no
excerpt**. The last is v5's: the model must list which excerpts state the facts its answer needs,
and a `full` label with an empty list is refused. Turning an unfalsifiable claim into a checkable
one is what let `top_k` rise without false resolutions rising with it.

**The clarification reviewer** (`src/review.py`) is a second call that sees only CLARIFY
decisions, and is asked one question: was that clarification necessary? **It is off by default
since v5** — see finding 10; the mechanism is kept, tested, and one constructor argument away. It
is the mirror of the policy layer — one-directional in the opposite direction, able only to turn
CLARIFY into RESOLVE, never to create a deferral. Two guards keep it honest, both of them measured
rather than assumed:

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

- **Retrieval recall** — scored offline against each golden item's own `article_ids`, with no
  model calls. This is the measurement that reframed the project (finding 9) and it costs nothing;
  it should have been the first thing run rather than the fourth.

Decision accuracy is a pure label comparison: one call per item, no judge needed.

```bash
python -m src.eval.harness --resolver gated --no-judge --workers 8   # decision metrics, ~6 min
python -m src.eval.sweep                                            # policy sweep, 0 API calls
```

Every number in this README traces to a committed report. The v5 ones:

| report | what it is |
|---|---|
| `v5_shipped.json` | **the shipped figures** — mean + range over the three runs below |
| `v5_shipped_run{1,2,3}.json` | `top_k=10`, citation check, reviewer off. Three complete runs |
| `v5_topk5.json`, `v5_topk5_run{1,2,3}.json` | the same gate at `top_k=5` — the other end of the frontier |
| `v5_topk10_nocite_run{1,2}.json` | `top_k=10` *before* the citation check — the false-resolution rise it bought back |
| `v5_topk5_reviewer.json`, `v5_topk10_nocite_reviewer.json` | the reviewer measured at both settings (finding 10) |
| `v5_v3reviewer_remeasured.json` | the old shipped build, re-run — the 44% → 37% discrepancy |
| `gated_both_160_promptv4.json` | v4, kept as the record of finding 3 |

All were run with `--no-judge`; every one carries its own `token_spend`, `workers`,
`review_clarifications`, and a `failures` list. Runs that lost items are excluded from the
aggregates rather than averaged in.

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

*This finding stands as written — the isolation argument is sound and the measurement was real —
but the reviewer no longer ships. v5 removed the over-clarification it was compensating for, and
it now costs more than it saves. See finding 10.*

**5. The policy layer is very nearly inert.** `n_policy_overrides` is **1 across 160 items** — a
single `partial`-coverage downgrade. The sweep confirms it: flipping `escalate_on_partial` moves
golden accuracy 49% → 50% and adversarial not at all.

The reason is exact: every false resolution carried `kb_coverage="full"`, which the net is
structurally blind to. It can only catch a model that admits doubt, and this model's failure mode
is confident mislabeling. The gate's value comes almost entirely from the model's structured
self-assessment, not from the deterministic layer beneath it. The layer is cheap and
one-directional so it stays — but a design resting on it would be resting on nothing.

**v5 made this more true, not less, and the citation rule is the sharpest illustration.** Under
v5 the `escalate_on_partial` knob is now *completely* inert — the sweep gives byte-identical
metrics either way, because the v5 rubric makes the model itself defer whenever it labels
coverage `partial`, so the policy layer never gets the chance. Total overrides across the three
shipped runs: **0, 1, 0**.

And the citation rule (finding 9) added a fourth policy rule that **never once fired as an
override** in those runs. Its measured benefit — false resolutions 12.5 → 11.7, out_of_kb 57% →
65% — came entirely from *requiring the model to produce the pointer*, which changed how it
labelled its own coverage. Not from the backstop catching it out. That is worth being precise
about, because it is easy to credit the wrong half: the enforcement is real and one-directional
and I would keep it, but on this eval set it is insurance that never paid a claim. What did the
work was making an unfalsifiable claim into one the model had to attempt.

**6. Injection detection and injection disposition are different problems.** The gate flags
`injection_attempt_detected=True` on **10 of 10** injection items — perfect detection. It then
answers 8 of them with `CLARIFY` regardless of what the underlying request needed. Recognising an
attack is far easier than continuing to reason normally once you have.

**7. Retrieval scores carry no confidence signal.** An earlier approach thresholded on retrieval
score. RRF scores encode *rank*, not match quality — a chunk ranked #1 by both retrievers always
scores 0.0164, whether it answers the question or is merely the least-bad of 10,068. Measured,
near_miss tickets score *higher* than golden ones, and the best threshold tuned directly on the
test set reached only 72.5%. The judgement has to be made by something that reads the text.

**8. Over-deferral was not a calibration problem. Three of the gate's definitions were wrong.**
Findings 2–4 all treated over-clarification as the model being *too cautious* and tried to make it
less so: sterner wording (failed), then an external reviewer (worked, 7 points). Both took the
gate's categories as correct and its thresholds as miscalibrated. Reading all 57 false deferrals
from the v4 run item by item showed that was the wrong diagnosis — they fell into four mechanical
groups, and three were definitions doing the wrong job:

| what the gate did | n | why it was a definition bug |
|---|---:|---|
| Asked the customer to pick which of *our* procedures or interfaces applied | 20 | "Which editor are you in?", "dashboard or mobile app?", "shipping rule or coupon?" — the missing fact was not the customer's to supply |
| Asked the customer to run diagnostics the KB already lists | 13 | The excerpts *were* the checklist; sending it is the answer |
| Escalated because the answer had to be *assembled* from two excerpts | 9 | `full` coverage required one sentence stating the whole fact, so synthesis was labelled `partial` and blocked |
| Escalated a *question about* a privileged topic | 4 | `requires_human_authority` was defined by subject matter, so "what is the refund policy?" tripped it |

The fixes are definitions, not exhortations. `CLARIFY` now requires the missing thing to be a fact
**only the customer holds**, with procedure-choice and interface-choice categorically excluded —
and categorical exclusions are decidable in a way "would the reply change what happens" is not,
which is exactly why v4's sterner version of the same judgement failed. `requires_human_authority`
became act-based. `kb_coverage` split collection from derivation.

Measured, gate-only at unchanged `top_k=5`: **golden CLARIFY fell 25 → 1**, false escalation 37%
→ 16–19%, and adversarial accuracy *rose* 65% → 74% with `near_miss` reaching 100%. Both sets
improved, which no previous revision managed.

Two corrections were needed along the way, and both are the same shape — a rule stated too
broadly. Loosening coverage to permit assembly initially resolved **11 of 20** out_of_kb items,
because the model began answering from the excerpts' *silence* ("Wix does not support that"); the
fix is that absence is not coverage, since retrieval returns a handful of articles whose silence
means nothing. And the exclusion list, written unconditionally, overrode the bare-ticket case and
cost 8 `ambiguous` items — a one-line ticket like "the chat widget isn't showing up" got the
checklist instead of the question it needed. Ordering the bare-ticket test *first*, as a gate that
stops, fixed it. Both bugs were mine, both were caught by measuring the other set, and both are
the reason finding 4's lesson — always measure both — is now enforced by making full runs cheap.

**9. Retrieval, not caution, set the floor on false escalation — and this is the finding I would
lead with.** After finding 8 the false escalation rate sat at ~20% and would not move. Scoring
retrieval against the golden set's own `article_ids` explained why: **recall@5 was 76%**. For 24
of 100 answerable tickets the article containing the answer was never in the context. Worse, the
gap tracked the failures precisely — recall was **54%** on the 13 tickets that deferred in every
run, against **80%** on the ones that resolved.

That reframes the target completely. **A correctly-calibrated gate had to defer those 24 tickets**,
and the only way to reach 15% by tuning the gate would have been to license answering from
material that does not contain the answer — the exact failure the project exists to prevent. The
bar was arithmetically out of reach, and three iterations of prompt work had been spent against a
constraint that was never in the prompt.

`top_k` was the whole fix. Recall over all 100 golden items:

| `top_k` | 5 | 8 | 10 | 12 |
|---|---:|---:|---:|---:|
| recall | 76% | 86% | **88%** | 90% |
| context | 9.4k | 15k | 18k | 22k chars |

Raising `candidate_pool` instead makes it *worse* (77% → 80% → 84% at `top_k=10` for pools of
20/40/60): RRF rewards agreement between the two rankings, and a deeper pool adds rank-tail chunks
that dilute it. So the pool stays at 20 and only `top_k` moves.

`top_k=10` took false escalation to **12%**, and the cost was real and predictable — twice the
context is twice the adjacent material to mistake for an answer, so false resolutions rose 9.3 →
12.5, concentrated in `near_miss` and `out_of_kb`. What bought that back was the citation check:
requiring the model to *name* the excerpts stating the facts its answer needs, and refusing a
`full` label that points at nothing. False resolutions returned to **11.7** — below the old
build's 12 — with false escalation unchanged at 12%. A structural check, not a firmer instruction;
finding 3's lesson applied deliberately this time.

**10. The reviewer stopped paying once the defect it compensated for was fixed.** Finding 4's
clarification reviewer is now **off by default**, which reverses this project's own conclusion.
It is not that the reviewer degraded. It existed to catch clarifications that should have been
answers, and v5 stopped producing them — golden CLARIFY fell from 25 to 1–6. Nearly every
clarification it now sees is a genuine one, so its only remaining effect is to overturn some of
those:

| | false resolutions | deferral errors | error cost |
|---|---:|---:|---|
| v5 gate, `top_k=5` | 9.3 | 25.0 | 9.3C + 25.0 |
| … + reviewer | 13.0 | 25.0 | 13.0C + 25.0 |
| v5 gate, `top_k=10` | 12.5 | 17.0 | 12.5C + 17.0 |
| … + reviewer | 15.0 | 13.0 | 15.0C + 13.0 |

At `top_k=5` it is strictly dominated — 3.7 more false resolutions and not one fewer handoff. At
`top_k=10` it wins only below C = 1.6, against a gate that beats the baseline above C = 0.36. Its
break-even moved from **7.0 to 1.6** between v3 and v5, and both constants are in the code with
the derivation next to them.

The general lesson is worth more than the configuration change: **a compensating mechanism is
evidence about the thing it compensates for, and its value is not additive.** Fixing the cause
does not leave the compensator neutral — it inverts it. Anything bolted on to correct a defect
needs re-measuring after the defect is fixed, not assumed to still be earning its keep. This one
was measured twice and it had stopped.

**11. A positional reference into an append-only log is a bug with a delay fuse.** The explorer
selected which prompt revision to display as "the second-to-last trace per ticket", true when
written because the eval had been run a known number of times. v5 ran it eight more times, after
which that index pointed at an arbitrary intermediate prompt and the page rendered decisions no
configuration ever produced — reporting `0 skipped`, with no error. Traces now carry the full
configuration that produced them (`gate_version`, `top_k`, reviewer, knob), selection asks for
what it wants, and the export **refuses to write a page** unless every ticket matches one
configuration. Not a modelling finding, but the failure was silent, self-inflicted by ordinary
iteration, and would have quietly falsified the published artefact.

## What I would do next

1. **Push retrieval, not the prompt.** Finding 9 is the load-bearing one: 12 points of false
   escalation remain and recall@10 is 88%, so most of what is left is tickets whose answer never
   arrives. The candidates are a reranker over a deeper candidate pool (which fixes the RRF
   dilution that made bigger pools worse), chunk-level rather than article-level scoring, and
   query rewriting for the vague tickets where BM25 has nothing to match on. All of it is
   measurable offline against `article_ids` with no LLM calls — the recall sweep costs zero
   tokens, which is why it should have been the first thing run rather than the fourth.
2. **Re-derive the v5 rules on held-out data.** The three definitional changes came from reading
   the v4 run's failures on the golden set, which is the set they are scored on. See Limitations;
   this is the biggest threat to the headline number and the cheapest thing to check.
3. **Fix injection disposition.** Detection is solved (10/10); disposition went 10% → 50%, which
   is 4 items at n=10 and therefore a direction, not a result. The remaining misses are the
   documented tension between step 1 and step 2 — "I need my domain unlocked" is both vague and an
   authority case, and the precedence rule picks CLARIFY where the label wants ESCALATE. Needs a
   bigger injection set before it can be measured, let alone tuned.
4. **Get a real cost ratio.** The break-even is now 0.36 for shipping the gate at all, and C ≈ 3.5
   for the `top_k` choice. The second one is a live decision an operator has to make; the first is
   low enough that it is hard to see reality failing it. Both are questions for whoever owns the
   support queue.
5. **Always measure both sets.** Kept from the previous list because v5 proved it twice more, in
   both directions: loosening coverage cost 11 out_of_kb items, and the clarify exclusions cost 8
   ambiguous items. Neither was visible in the metric being optimised. Full runs are now ~6
   minutes (`--workers 8`), so there is no excuse left for a one-sided measurement.

## Limitations

- **The v5 rules were derived from the eval set they are scored on.** This is the most serious
  limitation here and it is not fixable by argument. The three definitional changes in finding 8
  came from reading the 57 false deferrals in the v4 golden run and grouping them by mechanism;
  the retrieval fix came from scoring recall against the golden set's `article_ids`. Some fitting
  to these 160 items is therefore certain, and the 12% should be read as an upper bound on what a
  fresh answerable set would show.

  Three things bound the damage, none of which eliminates it. The rules are *categorical* rather
  than item-shaped — "never ask which editor they are in" is a support-triage principle that
  either holds or does not, not a lookup table, and I would defend each of the three in a design
  review without reference to this eval set. The worked examples in the prompt are **invented
  rather than lifted from the golden set**, deliberately, to avoid putting the test items in the
  context that grades them. And every change was scored on *both* sets, so a rule that merely
  memorised golden answers would have shown up as adversarial regression — which is exactly how
  the two over-corrections were caught. But the item-level diagnosis was still done on the test
  set, and a held-out answerable set is the only thing that would settle it. That is next-step 2.
- **Single model; three runs, not a confidence interval.** All figures are `gpt-5-mini` at
  `reasoning_effort=low`. The v5 numbers are means over three complete runs with observed ranges
  shown, which is enough to see that ±3 points on false escalation is noise and not enough to
  compute a real interval. Category numbers at n=10 (`near_miss`, `injection`) move 10–20 points
  between identical runs and are directional only. The single-run columns for baseline and v3 are
  weaker still — see *Run-to-run variance*, where re-measuring the old build moved its headline
  7 points.
- **The `top_k` frontier is measured at two points.** `top_k` 8 and 12 have recall figures but no
  end-to-end decision runs, so the crossover at C ≈ 3.5 is drawn through two points. The API
  budget ran out before 8 could be measured, and 8 is where I would look first: it has 86% recall
  for 3k fewer characters of context than 10.
- **The results explorer still shows the v3 configuration.** `docs/data.js` is committed from the
  old build, and regenerating it needs one clean 160-item pass under a single configuration, which
  the remaining API budget did not cover. The export now refuses to write a mixed-configuration
  page rather than silently producing one (finding 11), so the published page is stale but
  internally consistent. `python -m src.eval.harness --resolver gated --no-judge --workers 8`
  followed by `python -m src.eval.export_traces` regenerates it.
- **The v3 reviewer's numbers came from a replay, not a fresh end-to-end run.** The 72 reviewer
  calls were real, against the live model, but they were made over the base gate's *cached*
  decisions rather than re-running both stages together. Because the reviewer only reads the
  gate's output and never feeds back into it, the composition is exact — what a fresh run adds is
  run-to-run variance, and when one was finally done for the v5 comparison it moved the old
  build's headline by 7 points. The replay is
  `src/eval/review_replay.py`; it writes `clarify_review_replay_governed.json` (the v3 shipped
  configuration) and `clarify_review_replay.json` (the ungoverned variant, for the comparison in
  finding 4). `--from-cache` rescores both from the saved verdicts with no API calls, so every
  number in finding 4 is reproducible offline. The v5 reviewer figures in finding 10, by contrast,
  are fresh end-to-end runs.
- **The reviewer's two guards were chosen after seeing which flips failed.** Both have an argument
  from first principles — respect the invariants the policy layer already enforces; do not relax
  caution on a ticket carrying an active attack — but I did not write them down before looking at
  the data, and with only 15 flips to learn from, some fitting to this eval set is likely. They
  need a fresh adversarial set to be trusted.
- **Prompt revisions measured unevenly, before v5.** v1 and v3 have adversarial numbers; only v3
  and v4 were measured on both sets. The effect of the v1 → v3 step-1 fix on false escalation was
  never measured, and given what v4 did, it may well have made it worse. Every v5 revision was
  scored on both sets, which is what made the two over-corrections visible.
- **Groundedness and correctness not measured at scale, and not on this model.** They ran on 6
  golden items during development (groundedness 1.0, mean correctness 4.67/5) — a smoke signal, not
  a result — and that run used `gemini-3.5-flash-lite` with a Gemini judge, before the switch to
  `gpt-5-mini`. See `eval_results/baseline_sampled.json`. Every other figure in this README is
  gpt-5-mini.
- **The 160-item mix is arbitrary.** Every combined figure — raw accuracy, deferral precision, and
  the cost curves that set the 0.36 and 3.5 thresholds — assumes a 100:60
  answerable-to-adversarial ratio. Real traffic has its own ratio, and all of it moves with it.
  This matters more for the `top_k` crossover than for anything else: it is a comparison between
  an error counted on golden and an error counted on adversarial, so it moves directly with the
  mix.
- **Retrieval recall is measured against a single reference article per item.** The golden set
  gives each question the `article_ids` it was written from, and recall asks whether any of them
  reached the top-k. An answer available in some *other* article counts as a miss, so 76%/88% are
  lower bounds on the retriever and the "24 tickets could not have been answered" claim in finding
  9 is correspondingly an upper bound. The direction of the finding is unaffected — the gap between
  deferred (54%) and resolved (80%) items is what carries it — but the exact floor is soft.

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

Evaluating it:

```bash
# both sets, 160 items, ~6 min with 8 workers. Always both sets -- see finding 8.
python -m src.eval.harness --resolver gated --no-judge --workers 8 --out my_run.json

# mean + spread over repeated runs; refuses to average a run that lost items
python -m src.eval.aggregate my_run.json my_run2.json my_run3.json     --label "my change" --out my_change.json

# side by side in the error taxonomy, with per-item decision changes
python -m src.eval.compare eval_results/v5_shipped.json eval_results/my_change.json --items

# offline, zero tokens: replay the escalate_on_partial knob over cached traces
python -m src.eval.sweep
```

`--workers` is what makes iterating at full coverage affordable; serially the same run takes ~45
minutes, which is how three of the four earlier prompt revisions ended up measured on one set
only. Failed items are retried once and any that still fail are listed in the report, because
every rate is a fraction over the items that survived.

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
its own `token_spend`. Through v4 — four prompt revisions, both eval sets, and the reviewer replay
— the whole project came to about **$1.10**.

v5 cost several times that, and the reason is worth stating: it took **eleven complete 160-item
runs** rather than one per revision, because the run-to-run spread is ±3 points and single runs
could not distinguish a real 4-point gain from noise. `top_k=10` also roughly doubles input tokens
per call (9.4k → 18k characters of context). Repeats and full coverage are the expensive part of
measuring honestly, and they are the part that was missing before. The two offline tools — the
recall sweep in finding 9 and `src/eval/sweep.py` — cost nothing at all, and the recall sweep is
the one that mattered most.

On a reasoning model, hidden reasoning tokens share the completion budget and bill at the output
rate: at `max_tokens=1536` the structured call returned `{}` — a successful, billed, empty
response. `OPENAI_REASONING_EFFORT` and a 4096-token budget address this.

`src/eval/sweep.py` replays the policy over cached traces at zero cost, so tuning
`escalate_on_partial` never needs a second paid run.
