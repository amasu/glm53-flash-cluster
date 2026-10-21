# Downstream consumer notes — glm-5.3-flash

Interactions observed when OpenAI-compatible clients use this endpoint for
**vision / auxiliary image-analysis workloads**. Server-side behavior is
correct; the pitfalls are on the consumer side. (2026-10-20, standing
`nvfp4-dflash2` stack on the supervisor endpoint.)

## Vision input works on the standing stack

`/v1/chat/completions` accepts `image_url` content parts (data-URI or URL)
on the standing nvfp4-dflash2 stack — no `lab-vision` takeover needed for
occasional image inputs. A 256 KB screenshot embedded as a data URI prefilled
~840 prompt tokens and completed normally.

## Pitfall: thinking is ON by default — budget for it

`glm-5.3-flash` is a reasoning model: by default it emits a `reasoning`
field (and only then `content`). Measured against this endpoint:

| Request | thinking | wall time | reasoning tokens |
|---|---|---|---|
| tiny 8×8 PNG, `max_tokens=20` | on | ~2 s | 20 (all tokens; `content: null`, `finish_reason: length`) |
| 256 KB screenshot, 3-sentence description, `max_tokens=600` | on | 24.2 s | 212 |
| same request | **off** | **12.7 s** | 0 |

Consequence: image-analysis prompts that ask for detailed output can spend
minutes in reasoning before any content — enough to trip short client-side
budgets. Real case: Hermes Agent's `vision_analyze` aux task inherits the
main model (`provider: auto`) with a 120 s default timeout and timed out
repeatedly with "Request timed out" against this endpoint.

## Consumer-side fix (Hermes Agent, 2026-10-20)

```
hermes config set auxiliary.vision.timeout 300
hermes config set auxiliary.vision.extra_body \
  '{"chat_template_kwargs": {"enable_thinking": false}}'
```

Notes:
- `enable_thinking: false` goes through `chat_template_kwargs` (GLM chat
  template on vLLM). Generic OpenAI `reasoning_effort` fields are NOT honored
  by this template.
- Hermes caches `config.yaml` at session start — the change applies to NEW
  sessions only; a running session keeps the old timeout.
- Same pattern applies to any aux/summary/classifier task pointed at this
  endpoint: either disable thinking for short-form tasks or raise the client
  timeout well above 120 s.

## Server-side alternative

If a consumer cannot change its client, the equivalent knob server-side is
serving with the non-thinking template default, or documenting
`chat_template_kwargs` support in the endpoint's consumer docs. Do not change
the standing stack's defaults for one consumer — the 95/100 hardmode score
depends on the current recipe (benchmarks.md §14).
