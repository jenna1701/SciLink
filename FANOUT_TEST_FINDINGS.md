# Meta fan-out — live testing findings (branch `feature-meta-fanout`)

Autonomous test/fix session on Bedrock opus-4-8. Summary of what was validated,
bugs found, and what was fixed vs. flagged.

## Validated (live, real LLM + real codegen)

### Complementarity gate — battery (one LLM call each)
| Scenario | Verdict | Outcome |
|---|---|---|
| Image + co-registered EELS map (same region) | complementary 0.93 | run ✅ |
| STEM image + XRD of a different material | unrelated 0.95 | decline ✅ |
| **Cross-modal: BSE image + XPS, same coupon** | complementary 0.83–0.86 | run ✅ |
| **Cross-modal: Ti image + Si Raman** | unrelated 0.96 | decline ✅ |
| **Redundant: two XPS of the same coupon** | redundant 0.95 (clustered) | decline ✅ |
| **3-way: image + XPS + unrelated Raman** | partially_complementary | prune Raman, run image+XPS ✅ |
| **Image + hyperspectral EELS datacube (co-registered)** | complementary 0.97 | run ✅ (join: pixel co-registration on 32×32 R1 grid) |

### End-to-end (parallel branches + cross-modal fusion)
- **Cross-modal image + XPS:** BSE→`ImageAnalysisAgent` (63%/37% two-phase),
  XPS→`CurveFittingAgent` (Ti⁰/TiO₂ doublets). Fusion found the *real*
  two-component agreement across modalities **and** correctly refused to
  over-assign Z→phase without EDS. Honest, non-fabricated synthesis. ✅
- **Image + hyperspectral datacube:** gate complementary 0.97 (pixel
  co-registration); HAADF→`ImageAnalysisAgent`, EELS cube→`HyperspectralAnalysisAgent`
  (correct heterogeneous routing); both branches produced output, fusion
  succeeded. The run was noisy (see Bug C + the EELS metadata note) but the
  fan-out orchestration was resilient: branches retried and recovered, and the
  agents' own QC correctly rejected degenerate fits on the synthetic cube. A
  clean re-run with proper energy metadata confirms the EELS branch. ✅

### Offline robustness (24/24 checks, deterministic)
branch-error handling, empty-but-successful flagging, fuse edge cases
(<2 / nonexistent / good indices), N=4 concurrency + ledger integrity,
single-branch rejection, gate fail-closed, soft/hard size caps.

## Bugs found & FIXED (in the fan-out feature — mine)
1. **Gate empty completions** — meta chat model carried delegation tool
   schemas → structured prompt returned an attempted tool-call w/ empty text.
   Fixed: dedicated tool-free `_structured_model` + retry-on-empty.
2. **Gate under-credited bulk-vs-local reconciliation** — image+spectrum
   (non-pixel-co-registered) flip-flopped complementary↔uncertain. Fixed the
   rubric to credit "spatially-resolved vs bulk/area-averaged of the SAME
   sample" as a valid join (the canonical multimodal case the fusion template
   targets). Stabilized to 5/5 complementary.
3. **Empty/errored branches waved through** — `run_fanout` now flags any
   branch with no usable output (errored OR empty-success), warns, and does
   not recommend fusing them.
4. **`_mesh_task` primary ambiguity** — now states the PRIMARY path explicitly
   so a child can't analyze a companion as the primary on near-identical data.
5. **Negative-verdict guard** — an `unrelated`/`redundant` verdict can never
   carry a runnable `fanout_set`, even if the model inconsistently lists one.
6. **(MOST IMPORTANT) Branch metadata was never forwarded to the analysis
   child.** `metadata` in a branch spec fed only the gate; `_mesh_task`/
   `_run_one_branch` never passed it on, so every branch SYNTHESIZED metadata
   from the task prose ("I created XPS metadata… none was provided") — losing
   technique-specific fields (e.g. the EELS `energy_range` → repeated
   `missing physical range` errors). On real data this risks wrong technique/
   axis assumptions and thus wrong analysis. Fixed: `_mesh_task` now forwards
   the primary's metadata — a `.json` path → instructs the child to
   `load_metadata` on it (don't synthesize); inline text → embedded directly.

## Bugs found — PRE-EXISTING in the shared curve/image stack (NOT fan-out; flagged for your nod)

These surfaced when the hard XPS fit hit a synthesis-JSON-parse failure. The
fan-out is resilient to them (the branch caught the error, retried, produced a
`_002` result, fusion succeeded), but they are real and worth fixing. They live
in sensitive shared agent code, so I did NOT change them — patches ready below.

### Bug A — `_salvage_synthesis_fields` AttributeError on the salvage path
- **Where:** `controllers/curve_fitting_controllers.py:6565,6734` and
  `controllers/image_analysis_controllers.py:6642,6821` call
  `self._salvage_synthesis_fields(response)`.
- **Root cause:** that method lives on `BaseAnalysisAgent` (`base_agent.py:149`),
  but `UnifiedCurveSynthesisController` (6347) / `UnifiedImageSynthesisController`
  (6452) are standalone classes that don't inherit it → `AttributeError:
  '…Controller' object has no attribute '_salvage_synthesis_fields'`.
- **Trigger:** only when the synthesis JSON fails strict parsing (long
  `detailed_analysis` with an unescaped quote/newline) — exactly the case the
  salvage path exists for, so the safety net itself crashes.
- **Proposed fix:** have the controllers call the shared util directly (which
  `base_agent` already wraps):
  ```python
  from ...utils.synthesis_parse import salvage_synthesis_fields
  raw = response.text if hasattr(response, "text") else str(response)
  salvaged = salvage_synthesis_fields(raw)
  ```
  (or give the controller a `_salvage_synthesis_fields` helper mirroring the
  base agent's extract-text-then-salvage). Apply at all four call sites.

### Bug B — `NoneType` not subscriptable in `run_analysis`
- **Where:** `analysis_orchestrator_tools.py:2686`
  `"detailed_analysis": result.get("detailed_analysis", "")[:2000]`.
- **Root cause:** when synthesis fails (Bug A), `result["detailed_analysis"]`
  is present but `None`; `.get(key, "")` returns `None` (key exists) → `None[:2000]`.
- **Proposed fix:** `(result.get("detailed_analysis") or "")[:2000]` and guard
  `scientific_claims` similarly. Defensive; safe across all analysis paths.

Note: Bug A → synthesis returns None → Bug B crash → caught by `run_analysis`'s
outer try → agent retries (the `_002` run). So the loop recovers, at the cost of
a wasted attempt + error-level logs.

### Bug C — no image-dimension cap → API rejects >8000 px images (NEW, shared infra)
- **Where:** `wrappers/litellm_wrapper.py:415-434` (`_convert_parts`) encodes both
  PIL images and `{mime_type,data}` byte dicts to base64 with **no dimension
  guard**.
- **Symptom (seen live, 8×):** `BedrockException … image dimensions exceed max
  allowed size: 8000 pixels` on the image-synthesis montage → synthesis returns
  `None` → Bug B. The model APIs (Bedrock AND direct Anthropic) hard-cap at
  8000 px per dimension; a tall multi-panel synthesis figure (best-of-N
  candidates stacked) exceeds it.
- **Impact:** affects ANY image-analysis run with large composite figures, not
  just fan-out — likely common on Bedrock.
- **Proposed fix (central, provider-agnostic):** downscale any encoded image
  whose largest dimension > ~7990 px before base64, in both image branches of
  `_convert_parts`:
  ```python
  _MAX_IMAGE_DIM = 7990
  def _cap(raw: bytes) -> bytes:
      im = Image.open(io.BytesIO(raw))
      if max(im.size) > _MAX_IMAGE_DIM:
          im.thumbnail((_MAX_IMAGE_DIM, _MAX_IMAGE_DIM))
          out = io.BytesIO(); im.save(out, format="PNG"); return out.getvalue()
      return raw
  ```
  (PIL path: thumbnail before save; dict path: decode→cap→re-encode only for
  image mime types.) High blast radius (every image to every model) — verify
  small images are byte-unchanged; hence flagged, not applied.

## Known characteristics (not bugs)
- The gate needs metadata to confidently call "complementary" — with bare
  array probes and no metadata it conservatively returns `uncertain` (correct:
  shape alone can't establish same-sample/join-axis). Fan-out effectively
  expects per-dataset metadata, which the meta already gathers before delegating.
- Synthetic test fixtures get flagged as non-physical by the analysis agents'
  own data-validity gates — agent honesty, not a fan-out issue. Tests use
  realistic fixtures (smoothed random fields, XPS doublets, EELS datacubes).
- The EELS hyperspectral skill *requires* a numeric energy axis
  (`energy_range`/`axis_spec.axis_2` with start/end) and raises `ValueError`
  without it — correct strictness, but a fan-out caller must ensure the
  datacube branch's metadata carries it (the meta gathers metadata pre-delegation).
