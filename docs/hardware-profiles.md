# Hardware profiles (`HW_PROFILE`)

A hardware profile is a per-machine bundle of speed settings: device, precision,
OpenVINO on/off, TrackNet on/off, and Pass 1 subsampling. Profiles tune each
machine for its own throughput. They are not meant to make two machines produce
identical output.

Profiles are separate from `LearnedParams` (`backend/app/learning/params.py`),
which are detection-quality parameters shared across all machines. Hardware
profiles never write to that table.

## Quick start

In `.env`:
```
HW_PROFILE=cuda
```
If `HW_PROFILE` is unset, the plain `CV_DEVICE` / `TRACKNET_ENABLED` /
`PASS2_BALL` settings apply unchanged. An unknown profile name is an error (see
[Precedence rules](#precedence-rules)).

## Where it lives

- **`backend/app/hw_profile.py`** — the schema (`PROFILE_DEFAULTS`) and the
  loader (`active_profile()`). This is the only file that knows the shape of
  a profile.
- **`backend/app/hw_profiles.json`** — the profiles themselves, tracked in git so
  every checkout has them. Adding a machine means adding a JSON block here; no
  code change is needed.

Current profiles:

| Name | Machine | device | precision | openvino | tracknet | pass2_ball |
|---|---|---|---|---|---|---|
| `cuda` | NVIDIA GPU | `cuda` | `fp16` | off | on | on |
| `arc140v` | Intel Arc 140V laptop | `xpu` | `fp16` | on | on | off |

**Deployment note:** the EC2 worker does not set `HW_PROFILE`. `deploy/restart-ec2.sh`
sets `CV_DEVICE=cuda` and `TRACKNET_ENABLED=1` directly, so the `cuda` profile's
`precision` and `openvino` values do not apply there. TrackNet weights reach the
worker through S3 (`storage_s3.upload_tracknet_weights`,
`cloud_ingest._sync_tracknet_weights`), the same way calibration does.

## Adding a new profile

Edit `backend/app/hw_profiles.json`, add a block:
```json
"my-gpu-name": {
  "device": "cuda",        // xpu | cuda | cpu
  "precision": "fp16",     // fp16 | fp32
  "openvino": false,       // Intel-only; leave off on non-Intel GPUs
  "tracknet": true,
  "pass2_ball": true
}
```
Then set `HW_PROFILE=my-gpu-name` in that machine's own `.env`. That's it —
nothing else to touch.

Optional keys (omit unless you actually need them):
```json
  "p1_scale": 0.3,        // overrides LearnedParams for THIS machine only
  "p1_frame_step": 4      // (see "p1_scale / p1_frame_step" below)
```

## Every knob and what it plugs into

| Profile key | Consumer | Behavior |
|---|---|---|
| `device` | `cv/device.py` `torch_device()` | Checked before `CV_DEVICE`; `None`/no profile → old `CV_DEVICE` path. |
| `precision` | `cv/players.py` (pose OpenVINO export + torch `.track(half=...)`), `cv/ball_tracknet.py` (`.model.half()`) | `"fp16"` (default) matches today's behavior (pose OV export was always `half=True`); `"fp32"` opts out. Skipped on CPU regardless (fp16 on CPU is unsupported/slow). |
| `openvino` | `cv/players.py` `_openvino_pose_dir`, `cv/ball_tracknet.py` `_openvino_infer` | `False` short-circuits the OpenVINO export/compile attempt entirely — set this on non-Intel GPUs to avoid a wasted export. |
| `tracknet` | `config.py` → `TRACKNET_ENABLED` | Only used if the `TRACKNET_ENABLED` env var is **unset**; an explicit env var always wins. |
| `pass2_ball` | `config.py` → `PASS2_BALL_ENABLED` | Same precedence as `tracknet`. |
| `p1_scale`, `p1_frame_step` | `pipeline.py` `_with_hw_overrides()` | See [below](#p1_scale-and-p1_frame_step). |
| `pose_batch` | *(nothing yet)* | Schema placeholder. Pose inference is frame-serial today (`PoseTracker.update()` calls `model.track()` one frame at a time); there's no batch path to plug a batch size into without restructuring the Pass 2 loop. Add real wiring only once someone measures a GPU-bound pose stage. |

## `p1_scale` and `p1_frame_step`

These two names are also **LearnedParams**: versioned, stored in the database, and
tuned globally by the discrepancy report. A hardware profile must not write to
that table, or one machine's speed settings would leak into every other machine's
baseline.

So instead: `pipeline.py`'s `_with_hw_overrides()` takes the LearnedParams
dict `get_latest()` just fetched and, **only if the active profile sets
these keys**, overlays them on an in-memory copy before Pass 1/Pass 2 run.
The DB row is never touched. Leave them unset (the default) to use whatever
the learned baseline says.

Practical effect: a weak GPU can set a smaller `p1_scale` / bigger
`p1_frame_step` so Pass 1 leans harder on subsampling and Pass 2 runs
lighter — without that override ever polluting the shared learned baseline
the other machine (or the report) sees.

## Precedence rules

- **Explicit env var always beats the profile.** `TRACKNET_ENABLED=0` in
  `.env` wins even if `HW_PROFILE`'s `tracknet` is `true`.
- **Unset `HW_PROFILE` → zero behavior change.** Every consumer falls back to
  exactly what it did before this feature existed.
- **Unknown `HW_PROFILE` name → hard error** (`SystemExit` listing the valid
  names) rather than silently falling back — a typo'd profile name should
  never quietly run on the wrong settings.

## Known interactions

- **autolearn / model_registry** don't reference device/precision at all —
  safe to run under any profile.
- **Learned values were measured under specific hardware** (e.g.
  `pose_imgsz=1920` benchmarks, ball thresholds) — switching device/precision
  can shift detection results slightly. This is expected, since cross-machine
  reproducibility is not a goal.
- **Device/precision resolve once per process** (`lru_cache`d in
  `device.py`, module-level in `config.py`) — changing `HW_PROFILE` needs a
  **backend restart**. This is also why there's no frontend toggle: a
  Setup-tab dropdown would imply live switching that doesn't actually work.

## Verifying

```powershell
cd backend
./.venv/Scripts/python.exe -c "from app.hw_profile import active_profile; print(active_profile())"
```
`None` with `HW_PROFILE` unset; the resolved dict otherwise.
