# Kev model engine

Kev is a Jev-like family of System 1 decision models on a Qwen3.5 backbone, proposed
for support in [#27](https://github.com/ThinkFlowLab/system1-omni/issues/27). This
directory holds the two parts of the engine that are not the backbone.

| File | What it does |
| --- | --- |
| `kev_prompt.py` | Builds the prompt from a state and its questions. Kev has no chat template, so this is the only thing that decides the layout. |
| `kev_head.py` | The trained pointer head and the decision readout: `scale * dot(k(h_opt), q(h_decide))`, temperature, softmax over a question's candidates. |
| `checkpoint.py` | Reads `head.pt` without torch. Loading the head is what the engine does at startup, and needing no torch for it is the point. |

Both are CPU-only: `re`, `math` and `numpy`, no torch and no CUDA. The backbone
runs elsewhere and hands over vectors; the head is 1.3M parameters against a 4.2B
backbone, so it is not worth a kernel.

## Status

Not yet wired to anything, and no PR: the shape of a Kev engine depends on the
answers in #27, and the backbone it consumes does not exist in this repository
yet. Placement follows the layout convention (`src/models/laya/`,
`src/models/cua_s1/`) rather than a decision that Kev is accepted.

## What is verified, and how

**Verified, no GPU needed.** Everything here is CPU code, so it is checked
against the real thing rather than against itself.

- **`kev_prompt.py` against upstream's own `encode()`.** 300 random records
  across four configurations, plus forged delimiters, truncation, option
  isolation and the `strict=True` overflow path, compared field by field. Zero
  divergence, and the raised messages match character for character.
  `tests/differential/` fetches upstream at a pinned revision and imports it with
  torch and transformers replaced by stand-ins; run
  `python3 tests/differential/fetch_upstream.py` first, and the tests skip
  without it.
- **`kev_head.py` against the shipped `head.pt`.** The four tensors load as
  `[256,2560]`/`[256]`, float32, 1,311,232 parameters. Probabilities sum to one
  within each question's candidates and not across a batch, identical candidates
  get exactly `1/K`, the fitted temperature 2.406050072164233 flattens the
  distribution without ever changing the argmax, and the batched path agrees with
  per-question scoring to 2e-9. Set `KEV_HEAD_PT` to a copy of `head.pt`; those
  tests skip without it.

**Not verified.** No backbone was run, no real tokenizer was exercised, and no
end-to-end decision was compared against the reference implementation. That needs
a GPU and the 9.3 GB checkpoint, and it is a separate step from this code.

## Two things this code depends on that are easy to get wrong

- **The delimiter ids come from the base tokenizer, not the adapter repository.**
  `jaredpalmer/kev-4b/added_tokens.json` maps the five delimiters to 151648-151661,
  which is the Qwen2.5 range; `Qwen/Qwen3.5-4B-Base` has no token between 151600
  and 151700. The ids that `AutoTokenizer.from_pretrained` actually resolves are
  in `kev_prompt.SPECIAL_IDS`, taken from two agreeing files. Wrong ids here index
  unrelated embedding rows and produce plausible-looking wrong decisions rather
  than an error.
- **The adapter targets the Gated DeltaNet layers.** Kev's LoRA covers twelve
  modules including all five GDN projections across all 24 linear layers, so the
  merge is mandatory there. The note that applies to Cua-S1, that its GDN layers
  run on base weights, does not transfer.
