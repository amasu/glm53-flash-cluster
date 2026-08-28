from pathlib import Path

# 512K-context guard for the SM121 (CC 12.1) sparse-MLA indexer.
#
# Why: the persistent-topk / cooperative workspace paths request 62 CTAs +
# 128 KB smem at 512K-context shapes; GB10 has 48 SMs and ~99 KB opt-in smem
# -> hard abort. Disabling both on compute-capability family 12.x lets the
# non-persistent fallback (which fits at <=262K AND 512K on 48 SMs) serve
# full-length requests.
#
# Anchors verified against vllm 0.1.dev20051+g487ecf187 in glm53:v8
# (vllm/vllm-openai:glm53-flash-arm64-cu130 day-0 image + v1..v8 chain).
# Mirrors the upstream FujitsuPolycom/glm53-flash-tp2-spark
# sparse_attn_indexer*.patch pair (Apache-2.0, vendored by the 92/100
# 512K-context recipe, forum #381350 post 55).

p = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm"
    "/model_executor/layers/sparse_attn_indexer.py"
)
s = p.read_text()
old = (
    "        use_persistent_topk = current_platform.is_cuda() "
    "and topk_tokens in (\n"
    "            512,\n"
    "            1024,\n"
    "            2048,\n"
    "        )\n"
)
new = (
    "        use_persistent_topk = (\n"
    "            current_platform.is_cuda()\n"
    "            and topk_tokens in (512, 1024, 2048)\n"
    "            and not current_platform.is_device_capability_family(120)\n"
    "        )\n"
)
if s.count(old) != 1:
    raise SystemExit("indexer use_persistent_topk match count: %d" % s.count(old))
p.write_text(s.replace(old, new))

p2 = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm"
    "/model_executor/layers/sparse_attn_indexer_kpool.py"
)
s2 = p2.read_text()
old2 = "        if current_platform.is_cuda() and select_k in (512, 1024, 2048):\n"
new2 = (
    "        if (\n"
    "            current_platform.is_cuda()\n"
    "            and select_k in (512, 1024, 2048)\n"
    "            and not current_platform.is_device_capability_family(120)\n"
    "        ):\n"
)
if s2.count(old2) != 1:
    raise SystemExit("kpool select_k guard match count: %d" % s2.count(old2))
p2.write_text(s2.replace(old2, new2))

# Self-verify: each file carries exactly one occurrence of the NEW guarded
# expression, and both files byte-compile.
import py_compile

assert p.read_text().count("use_persistent_topk = (\n") == 1
assert p2.read_text().count(
    "            and not current_platform.is_device_capability_family(120)\n"
    "        ):\n"
) == 1
for f in (p, p2):
    py_compile.compile(str(f), doraise=True)
print("512K-context indexer guard (CC 12.x family) applied + verified")
