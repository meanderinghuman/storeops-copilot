# Test set

500 scenarios, built before any training and frozen in git. Every scenario is at a **held-out store
(CA_3 or CA_4)** that never appears in the training data.

| File | Scenarios | Built by |
| --- | --- | --- |
| `testset.jsonl` | 400: 150 seen tools, 100 held-out tools, 75 out-of-scope, 75 ambiguous / write-safety | `python -m eval.build_testset` (seed 42) |
| `testset_hard.jsonl` | 100: misspelled names, two products per message, follow-up turns, tricky requests | `python -m eval.build_hard` (seed 43) |

## How expected answers are made

Each scenario has a structured `expected` block computed from the store database, not written by
hand: the tool calls that must happen (with exact arguments), the write that must happen (or must
not), numbers the answer must contain, and the outcome (`answer`, `order`, `no_order`, `transfer`,
`clarify`, `decline`). `note` restates it in plain words. The scorer (`eval/metrics.py`) gives 100%
to a scripted perfect agent on all 500 scenarios (`tests/test_eval.py`).

| Split | Categories |
| --- | --- |
| seen (150) | stock / on-order / lead time / price / sales lookups, N-day forecasts, run-out-before-weekend, reorder quantity, orders with a quantity, orders "for whatever we need" (5 where the right answer is *don't order*) |
| unseen (100) | store-wide promotions (20), "is X going on sale" (30), stock transfers in (25) and out (25) |
| out_of_scope (75) | weather, HR, IT, customer service, competitors, trivia, writing tasks; decline with no tool call |
| ambiguous_safety (75) | vague product categories (25), "order the usual" (10), "should I order X?" (20), hypothetical orders (10), "don't place anything yet" (10); none may stage an order |
| hard (100) | typos (25), two products (25), follow-ups after a recommendation: confirm / change quantity / decline / "what about B?" (25), negated orders, nonexistent stores, products not carried (25) |

Product requests use a different product each time, and only products whose full name the search
tool resolves unambiguously (three near-duplicate names are excluded). Wording for the main 400 was
rewritten into natural manager language by GLM-5.3; a rewrite was accepted only if it kept the exact
product name, every number and store ID, and the intent (an order stays an order).

## Review

The plan called for the project owner to hand-check every scenario. At the owner's request the review
was done by the AI coding assistant (Claude) instead, so treat it as a careful automated-plus-read
review, not an independent human audit. All 500 rows were read; automated checks covered dropped or
added numbers, product names in out-of-scope rows, and rewrites that turned questions into orders.

Changes made (recorded per row in the `review` field and in `testset_review.csv`):

- `t258`, `t261`: vague categories "hand" and "microwaves" weren't natural or weren't sold; now
  "hand soap" and "microwave popcorn", each matching several products.
- `t380`: an inventory-turnover email request was arguably in scope; replaced with a clearly
  out-of-scope request.
- `t226`, `t228`, `t239`, `t241`, `t244`: the rewrite added contradictory context ("we're low, so
  send stock to the other store"); reverted to the template wording.

Kept on purpose: `h100` ("How many lottery tickets do we have?") is borderline out-of-scope. It stays
a *not carried* case like the other seven (check the catalog before saying the store doesn't stock
it). The large API model failed it; the scenario was not changed after seeing that result.
