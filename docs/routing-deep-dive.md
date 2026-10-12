# Deep dive: routing one agent swarm across multiple small models

This document explains the technical mechanism behind the demo: how the **vLLM Semantic Router** lets a multi-agent OpenClaw workload use **several small models on one GPU** instead of one large model for everything.

![Routing](diagrams/diagram2-routing.png)

## The idea

Most agent turns are easy. A persona saying "summarize this in one line" or "say hello to the user" does not need a 27B model. But a persona asked to "prove this theorem step by step" does. If you serve a single large model, every easy turn pays the large model's latency and GPU cost. If you serve a single small model, hard turns fail.

The Semantic Router breaks that tradeoff: it classifies each request and dispatches it to the **cheapest model that can handle it**, while all candidate models stay **co-resident on one MI300X**.

## The components

```
OpenClaw agent  --OpenAI /v1/chat/completions-->  Envoy (:8899)
                                                     |  ext-proc
                                                     v
                                          vLLM Semantic Router  (CPU decision)
                                                     |
                     +-------------------------------+-------------------------------+
                     v                               v                               v
             Qwen3.5-2B  (:8001)          Qwen3.5-9B  (:8002)            Qwen3.8-27B (:8003)
                     \_______________________________ one MI300X _______________________________/
```

- **Envoy** exposes an OpenAI-compatible endpoint on `:8899`. Any OpenAI client (OpenClaw, `curl`, the OpenAI SDK) talks to it unchanged.
- **The router** runs as an Envoy `ext-proc` external processor. It never generates tokens; it only decides where a request goes. The classifier (keyword/regex, or an optional ModernBERT model) runs on the **CPU**, so the decision tier costs **zero GPU VRAM**.
- **The backends** are three plain vLLM servers, one per model, each launched with a distinct `--served-model-name` and port, all on the same GPU.

## The routing contract (what actually makes it work)

These are the constraints learned bringing the router up on real hardware. They are the difference between a router that starts and one that fails silently at the booth:

1. **Config model names must equal each backend's `--served-model-name`.** The router rewrites the request's `model` field to the decided name and forwards it; if that string does not match what the backend advertises, the backend returns a model-not-found error. The fleet script and the router config must agree on the exact names.

2. **Keyword `method` must be `regex`, `bm25`, or `ngram`.** `substring` is rejected at startup (schema validation). For deterministic, booth-safe matching use `regex` with word boundaries, e.g. `\b(prove|theorem|derive|optimize|refactor)\b`. This avoids matching "proven" inside unrelated prose.

3. **Decisions are evaluated in priority order, lowest number first.** The catch-all default must have the **highest** number so specific rules win. In this demo: `10` hard -> 27B, `20` trivial -> 2B, `100` default -> 9B.

4. **Verify routing from the router log, not from vibes.** Every request emits a `routing_decision` line naming the matched signal and the chosen backend. During bring-up, 4/4 crafted prompts (one trivial, one default, two reasoning) routed to the expected model. Treat an unverified route as unproven.

## Walkthrough of one request

A persona sends: `"Prove there are infinitely many primes."`

1. Envoy receives it on `:8899` and calls the router via ext-proc.
2. The router runs the keyword rules. `prove` matches the `hard_reasoning` signal.
3. Decisions evaluate in priority order. Priority `10` (hard_reasoning) matches first and wins.
4. The router rewrites `model` to `Qwen3.8-27B` and forwards to `:8003`.
5. The 27B generates the proof; the response includes the served model name.
6. The router logs the decision. The dashboard shows the request landing on the 27B lane.

A later `"hello"` from another persona matches `trivial` (priority `20`) and lands on the 2B at ~61 tok/s, so the swarm's easy turns never block on the big model.

## Why this is a good fit for a multi-agent swarm

A swarm multiplies request volume: 12 personas each taking several turns is dozens of requests per round. Their difficulty is **heterogeneous**: some personas (Scout, Translator) do light work, others (Analyst, Coder) do heavy work. Static model selection would either underserve the hard personas or overpay for the easy ones. Per-request routing matches each turn to the right model automatically, so the swarm gets 27B-quality reasoning where it matters and 2B latency where it does not, all from one GPU.

## See also

- [`../router/config.yaml`](../router/config.yaml) the full router configuration
- [`../swarm/swarm.py`](../swarm/swarm.py) the continuous rolling swarm orchestrator
- [GPU efficiency](diagrams/diagram3-gpu-efficiency.png) keeping the GPU fed
