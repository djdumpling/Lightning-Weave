"""Small CPU audit probe; random models, no benchmark accuracy claims."""
import json
import sys
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from self_distillation.attention.sparse import SparseAttentionPatch, SparseAttentionSpec, SparseRowContext, pattern_mask

torch.set_num_threads(1)
output = {"torch": torch.__version__, "device": "cpu", "benchmark_accuracy_claimed": False}
relay = []
for layers in (1, 4):
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=96, hidden_size=64, intermediate_size=96, num_hidden_layers=layers,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, attention_dropout=0,
                         initializer_range=0.12)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    original = torch.arange(32)[None]
    changed = original.clone()
    changed[0, 24] = 73
    spec = SparseAttentionSpec(window=4, sink_tokens=2, query_chunk_size=8)
    with torch.inference_mode(), SparseAttentionPatch(model, spec, require_half_precision=False):
        a = model(original, use_cache=False).logits[:, -1].float()
        b = model(changed, use_cache=False).logits[:, -1].float()
    direct = bool(pattern_mask(spec, torch.tensor([31]), torch.tensor([24]))[0,0])
    relay.append({"layers": layers, "query_position":31, "changed_key_position":24,
                  "direct_attention_permitted":direct, "max_abs_final_logit_change": (a-b).abs().max().item()})
output["hidden_state_relay"] = relay
assert relay[0]["max_abs_final_logit_change"] == 0
assert relay[1]["max_abs_final_logit_change"] > 0

geometry = []
n = 4096
for name, context in [
    ("sink4_w512", None),
    ("task_observation_w512", SparseRowContext(256, ((1024,1536),(2560,3072)))),
    ("task_observation_w512", SparseRowContext(2048, ((2560,3072),))),
]:
    spec = SparseAttentionSpec.from_name(name)
    patch = SparseAttentionPatch(model, spec, require_half_precision=False)
    if context:
        patch.set_context(**context.__dict__)
    blocks = patch._get_geometry(torch.zeros(1,1,n,1), n)
    edges = sum(item[4] for item in blocks)
    gathered = sum((item[1]-item[0])*item[2].numel() for item in blocks)
    retained_mask_bytes = sum(item[3].numel()*item[3].element_size() for item in blocks)
    geometry.append({"name":name, "length":n, "prefix_end":context.prefix_end if context else 0,
                     "causal_edges":n*(n+1)//2, "allowed_edges":edges,
                     "allowed_fraction_of_causal":edges/(n*(n+1)//2),
                     "gathered_query_key_pairs":gathered,
                     "gathered_to_allowed_ratio":gathered/edges,
                     "retained_mask_bytes":retained_mask_bytes})
output["geometry"] = geometry
path = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name("probe_results.json")
path.write_text(json.dumps(output, indent=2)+"\n")
print(json.dumps(output, indent=2))
