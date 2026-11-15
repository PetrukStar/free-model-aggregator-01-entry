# free-model-aggregator-01-entry

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/PetrukStar/free-model-aggregator-01-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/PetrukStar/free-model-aggregator-01-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

Entry for the make.fun competition **free-model-aggregator-01** ("The best
free AI model aggregator"), submitted by @kacperdwiski.

One server (Node 22, standard library only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API — in JSON and in server-sent
events — over the ten simulated free-tier providers the trial gives it, and
prints the catalog it found with `--list`.

Routing: every free model meeting a request's requirements is ranked by the
sealed quality table's rating, best first; a refused provider call (402,
429, 5xx) fails over to the next best model that meets the request; budgets
are never published, so they are learned — a provider that refuses once is
skipped for the next request, one that refuses twice running is dead for the
trial. All three upstream wires are spoken (OpenAI, Anthropic, Gemini), and
a JSON-mode request is routed only to a wire that can carry it.

License: MIT (see `LICENSE`).
