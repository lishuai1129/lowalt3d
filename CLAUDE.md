# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`lowalt3d` is a research (毕设) pipeline that turns **drone aerial photos** into a
**material-aware 3D scene** (ITU-R P.2040 electromagnetic properties per surface) for
**low-altitude wireless ray-tracing / path-loss simulation**. Output feeds three
interchangeable simulators: a built-in Python path-loss model, **Wireless InSite** (native
`.city`), and **Cesium for Unreal** (`.glb` + C++ LineTrace tracer in `ue/`).

Code comments, docs, and commit messages are in Chinese; domain terms and identifiers below
are kept verbatim.

## Layout & what's tracked

- `scripts/` — all pipeline stages (standalone argparse CLIs). **This is the whole codebase.**
- `configs/` — `classes.txt` (**6** material classes: `background/vegetation/asphalt/concrete/
  glass/metal`, order is load-bearing) and `radio_params.yaml` (P.2040 EM params per class).
- `ue/` — UE5 `RFRayTracer.{h,cpp}`; copy into a UE project's `Source/`, not built here.
- `data/`, `outputs/`, `experiments/`, `repos/`, `tools/`, `docs/`, `materials.csv` are
  **`.gitignore`d** — runtime products, third-party code, and local reference material. Do not
  expect them in git; do not add them.

Third party (fetch manually, see README): `repos/pgsr` = [PGSR](https://github.com/zju3dv/PGSR)
(a 3DGS variant, with CUDA submodules to compile), `tools/` = COLMAP Windows-CUDA build.

## Running

There is **no build system, test suite, or linter** — stages are Python scripts run one by one.

- **Platform is Windows-native.** README examples use `cmd` syntax (`set SCENE=`, `^` line
  continuation, `\` paths). The interactive shell here is bash — forward slashes and `$SCENE`
  work; help strings in scripts show `\` paths.
- Activate the conda env first (Python 3.10 + PyTorch 2.8 cu128; verified on RTX 5070 Ti /
  Blackwell sm_120). Deps: `numpy scipy opencv-python open3d pillow pycollada trimesh matplotlib pyyaml`.
- **`scripts/*.py` are run from the repo root `E:\lowalt3d`** (not from `scripts/`). Several
  inject `repos/pgsr` and `scripts/` onto `sys.path` at import time to reach PGSR's
  `scene` / `gaussian_renderer` / `arguments` modules and `material_common`.
- **PGSR stages (COLMAP convert, `train.py`, `render.py`) are run from `repos\pgsr`**, not root.

The full pipeline is in `README.md` ("快速开始"); every stage is parameterized by one `SCENE`
variable. Canonical order: COLMAP pose → PGSR train + TSDF mesh → georeference → auto material
labels → gaussian material learning → transfer to mesh → path loss + viz → export InSite/UE.

## Architecture — the big picture

**Data flows through files, not a shared runtime.** Each stage reads the previous stage's
output files and writes its own. The two interface contracts that hold the pipeline together:

1. **`face_materials.csv`** (`face_id,material_id,material_name,eps_r,sigma,penetration_loss_db`)
   — the universal per-face material table. *Every* downstream consumer (path loss, InSite
   export, UE export) reads exactly this, so the two material-assignment routes below are
   drop-in swappable and their outputs directly comparable.
2. **`material_common.py`** — single source of truth for the class list, the P.2040 EM table
   (`load_radio_params`, with a built-in fallback if `radio_params.yaml` is missing), and the
   RGB `_PALETTE`. The palette does double duty: it also encodes class → color in
   `material_mesh.ply`, and `export_gltf_ue.py` reverse-maps vertex colors back to classes via
   nearest-palette lookup. **Class order must stay identical across `configs/classes.txt`,
   `radio_params.yaml`, `_DEFAULT_CLASS_ORDER`, and `autolabel.py::_KEYWORDS`** — 2D mask pixel
   values are class indices into this order. `smooth_face_materials.py` /
   `label_windows_geom.py` assert `--classes` matches the yaml; other scripts do not — a desync
   silently shifts every `material_id`.

**Two material-assignment routes produce the same `face_materials.csv`:**
- *Main line (方向A, differentiable):* `train_gaussian_material.py` attaches a learnable K-dim
  material vector to each 3D Gaussian and supervises it against 2D masks by abusing PGSR's
  `override_color` hook — material logits are alpha-blended through the *existing* rasterizer
  (3 channels at a time), so **no CUDA changes are needed** (Feature-3DGS / LangSplat trick).
  Geometry/color/opacity are frozen; only the material vector trains. Optional multi-view
  consistency loss `L_cons` reprojects a neighbor view's material via rendered depth (KL).
  Then `gaussian_to_mesh_material.py` transfers per-Gaussian material to mesh faces by
  opacity×distance-weighted kNN vote.
- *Baseline (直接投票):* **removed.** `assign_material.py` (camera-ray voting onto first-hit
  face) was deleted — the differentiable route superseded it. Recover from git history if an
  A/B baseline is ever needed again.

2D masks come from the SegFormer route (`autolabel.py` / `train_seg.py` / `infer_seg.py` /
`labelme2mask.py`). `autolabel_color.py` (color/ExG heuristic) was **deleted** — pure colour
rules label shadow/dark ground as glass (recall ≈ 0 on aerial); do not resurrect it.

**Glass/label routing is two-level: detect = candidate, geometry = verdict.** Never feed low-confidence
2D detections straight into training as hard CE targets — that makes a model learn false glass.
- *Candidate layer*: `detect_windows.py` (Grounding DINO + optional SAM) emits 2D window candidates.
  Keep `--box_threshold` ≥ 0.12 (leak>precision) and use `--min_win_px` to drop GDINO-confidence
  speckles. It only *suggests* where windows are. Also `--note` prints the intended chain.
- *Verdict layer* (scene-independent geometry, does the final call): windows must sit on vertical
  faces (`--geo_gate --horiz_forbid glass,metal,vegetation`), be recessed pits with neighbor support
  (`label_windows_geom.py --recess_*`), and clustered above `--min_win_area` connected faces
  (`smooth_face_materials.py --min_win_area`, drops detection speckles back to neighbor mode).
- So the robust chain is: `detect(0.15+min_win_px)` → `train_gaussian_material --lambda_gate` →
  `gaussian_to_mesh_material` → `smooth --geo_gate --min_win_area` → `label_windows_geom` → export.

**Large scenes are split into blocks, trained separately, then merged back into one Gaussian
set.** `split_scene.py` → per-block PGSR train → `merge_blocks.py` → one `point_cloud.ply` →
normal TSDF/material/export chain, unchanged. The point is *not* "it OOMs otherwise" — it is
that a single run forces `--max_all_points 3000000 --max_abs_split_points 0
--opacity_cull_threshold 0.05`, which trade vegetation/thin-structure completeness for VRAM.
N blocks = N × the 6M Gaussian budget at PGSR defaults.
- **Iron rule: exactly one global SfM.** Per-block COLMAP yields mutually incompatible
  coordinate frames and cannot be merged.
- Partitioning happens in **ENU meters** (reuses `georeference.py`'s `get_gps` /
  `geodetic_to_enu` / `umeyama`), so up is +U and block sizes are given in meters. VastGaussian's
  four steps: camera-**count**-balanced grid → `core`/`expanded` boxes → cameras selected by
  center-in-expanded ∪ visibility ≥ `--vis_thresh` → points in expanded. Merge keeps only
  Gaussians whose center is in that block's **core**, so the overlap is cut exactly once.
- **PGSR needs no changes**: each block's `cameras_extent` is recomputed by `getNerfppNorm` from
  that block's own cameras. Blocks are COLMAP *text* sub-models (`sparse/*.txt`) with the 2D
  observation table left empty — PGSR reads only extrinsics/intrinsics + point coords. `-i`
  takes a relative `../../images_2` so the image set is shared, not copied per block.
- Don't reach for `colmap model_cropper`/`model_splitter`: measured on scene02, cropping to 55%
  of points still retains **674/678 images** (it keeps any image observing ≥1 retained point).

**Coordinate systems are the main footgun.** COLMAP world coordinates are arbitrary
units/orientation. `georeference.py` solves a **Umeyama similarity transform** (scale+R+t)
between COLMAP camera centers and GPS-EXIF ENU meters, emitting `geo_transform.json`. Downstream:
- `simple_pathloss.py` takes `--scale` (meters/unit) so absolute dB is meaningful; without it,
  only the *relative* distribution (NLoS regions higher) is valid. It infers the vertical axis
  and sky side from camera positions.
- `export_dae.py` / `export_gltf_ue.py` take `--geo_transform` to bake ENU meters + orientation.
  UE export additionally rotates COLMAP→ENU(Z-up)→glTF(Y-up) and back-computes a Cesium
  lat/lon/alt anchor at the mesh centroid.

**`cameras.json`** (PGSR/3DGS convention) is the shared camera format: `position` = camera
center, `rotation` = 3×3 camera-to-world, principal point at image center.

**Three simulators, cross-validated** on the same material mesh: `simple_pathloss.py` (Python,
FSPL + P.2040 penetration, LoS via Open3D raycasting), `ue/RFRayTracer.cpp` (UE LineTrace,
same physics), and Wireless InSite (`.city`). `compare_engines.py` resamples UE `coverage_ue.csv`
onto the Python grid and reports RMSE / correlation.

## Non-obvious conventions & gotchas

- **InSite: load `scene.city`, never `scene.dae`.** InSite's COLLADA importer collapses multiple
  materials into one; `export_dae.py` also hand-writes a native `.city` with per-face material
  *indices* precisely to bypass this. The `.city` writer format was reverse-engineered from
  InSite output — preserve its exact keyword structure if editing.
- Model lying flat / wrong size on export → pass `--geo_transform`. Ground floating at z≈-80
  (ENU origin is at drone altitude) → add `--ground_z0` (pure translation, path loss unaffected).
- **PGSR VRAM must be judged at iteration 7000+, not before.** Both geometric regularizers
  (`single_view_weight_from_iter` / `multi_view_weight_from_iter`) switch on at 7000; before
  that a run looks cheap and then falls off a cliff. Once over budget it does not merely slow —
  it spills to shared memory and thrashes over PCIe (100×+, never recovers).
- Budget ≈ **pixels-per-image × geometric-regularizer buffers + Gaussian count**; both terms
  matter and either can blow it alone. Measured on 16 GB: scene03 `-r 2` block (11.36 Mpx/img)
  died at only **2.85M** Gaussians; scene03 `-r 8` whole-scene (0.71 Mpx/img) sailed through
  7000 at 3.4 it/s but degraded to 1.97 s/it once densification reached **5.3M**. So blocking
  (which cuts Gaussians per block) and lowering `-r` (which cuts pixels) fix *different* halves —
  diagnose which term is dominant before picking.
- `--ncc_scale` is a **divisor** (`cameras.py:39`: `ncc_resolution = resolution / ncc_scale`),
  so `0.5` *upsamples* the NCC image 4× in pixels. Leave it at `1.0`.
- `-i` is joined onto `source_path`, so its depth must match: `../../images_4` from a block dir
  (`data/<scene>/blocks/bXX`), plain `images_4` when `-s` is the scene root.
- OOM during PGSR training: the two causes need different fixes. Too many *Gaussians* (or a
  scene too big to densify at defaults) → `split_scene.py`, never the
  `--max_all_points`/`--max_abs_split_points 0`/`--opacity_cull_threshold 0.05` trio, which buys
  VRAM with vegetation and thin-structure completeness. Too many *pixels per image* → lower
  `-r`; blocking cannot help there.
- PGSR's `convert.py` shells out via `os.system` → cmd.exe: pass `--data_path` /
  `--colmap_executable` as **absolute paths with backslashes**, or it dies with
  `'..' 不是内部或外部命令`.
- Georeference needs ≥3 GPS-tagged images; residual >5 m means consumer-GPS noise (use RTK/GCP).
- Heavy geometry libs (torch, open3d, trimesh, collada, cv2) are imported *inside* `main()`, not
  at module top, so `--help` stays fast and import errors are localized — follow this pattern.
- `simple_pathloss.py` P0 model is direct + penetration only (no reflection/diffraction yet).
