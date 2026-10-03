# Qwen3-4B (thinking): Qwen-reported vs. reproduced

| Eval | Qwen reported | My result (avg@k) | Context length | Output length |
|---|---|---|---|---|
| BFCL-v3 | 65.9 | 65.8 (avg@1) | 65,536 (YaRN x2, for multi-turn) | 32,768 |
| TAU1-Retail | 33.9 (avg@1) | 28.2 (avg@8) | 40,960 | 32,768/turn |
| TAU1-Airline | 32.0 (avg@1) | 31.2 (avg@8) | 40,960 | 32,768/turn |
| TAU2-Retail | 38.6 (avg@1) | 35.8 (avg@5) | 40,960 | 32,768/turn |
| TAU2-Airline | 28.0 (avg@1) | 22.8 (avg@5) | 40,960 | 32,768/turn |
| TAU2-Telecom | 17.5 (avg@1) | 19.1 (avg@5) | 40,960 | 32,768/turn |

# Qwen3-4B-Thinking-2507 on TAU2, by customer model (Qwen's numbers used gpt-4.1)

| Eval | Qwen reported | gpt-4.1 customer | Qwen3-30B-A3B-Thinking-2507 customer | Qwen3-235B-A22B-Instruct-2507-FP8 customer | Context length | Output length |
|---|---|---|---|---|---|---|
| TAU2-Retail | 53.5 (avg@1) | 52.6 (avg@1) | 39.3 (avg@5) | 53.5 (avg@5) | 65,536 | 32,768/turn |
| TAU2-Airline | 58.0 (avg@1) | 54.0 (avg@1) | 43.2 (avg@5) | 50.4 (avg@5) | 65,536 | 32,768/turn |
| TAU2-Telecom | 27.2 (avg@1) | 28.9 (avg@1) | 27.2 (avg@5) | 27.9 (avg@5) | 65,536 | 32,768/turn |
