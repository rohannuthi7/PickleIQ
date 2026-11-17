# Rally boundary logic

How rally start and end boundaries are decided: how Pass 1 opens and closes a
rally window, and how the refine module decides whether to keep a Pass 1 window,
move its start later, or end it sooner. For the broader detection pipeline, see
[`detection-spec.md`](detection-spec.md).

Every threshold named here is a parameter key. Pass 1 parameters live in
`backend/app/learning/params.py`; refine parameters live in
`backend/app/refine/config.py` (`RefineConfig`). Values below are the code
defaults; `data/refine/config.override.yaml` can override refine values at runtime.

## Two meanings of "Pass 2"

- **`cv/pass2.py`** does detection and visualization only. It writes no
  `rally_labels` and makes no boundary decisions. Pass 1 is the only source of
  boundaries.
- **`backend/app/refine/`** is the rally refinement module. It decides keep /
  start-later / end-sooner for each Pass 1 window and runs after Pass 2
  (`pipeline._refine_after_pass2` calls `refine.run_refine_json`).
  - It is **advisory**. It writes `data/refine/<video_id>.json` (and a
    `.segments.jsonl` confidence log), not `rally_labels`. Its verdicts inform the
    keep/drop review UI; Pass 1 windows remain the boundaries in use.

The "Refine" section below describes the refine module.

## Pass 1: how a rally starts and ends

File: `backend/app/cv/pass1.py` (`run_pass1`, `close_rally`). Sampling uses
`p1_frame_step = 3` and `p1_scale = 0.4`, so `sampled_fps = fps / 3`.

### Per-sample gate signal

Two mutually exclusive regimes:

- **Calibrated (primary):**
  `signal = (f - last_cross) <= p1_cross_window_s * fps`.
  The gate is live while a net crossing was seen within `p1_cross_window_s`
  seconds. Live play means the ball keeps crossing the net.
- **Uncalibrated (fallback):**
  `signal = motion > max(p1_motion_floor, baseline * p1_motion_ratio)`
  `AND (f - last_ball_frame) <= p1_ball_recent_s * fps`.
  Defaults: `p1_motion_floor = 0.12`, `p1_motion_ratio = 1.6`. `baseline` is an
  EMA that adapts only during dead time (α = 0.05 down, 0.01 up), so a rally never
  raises it. Motion is measured inside the court mask only.

**Net crossing** (`_match_fast` plus court projection): a blob that is the mutual
nearest neighbour between consecutive sampled frames, consistent in size
(`|Δdiameter| ≤ 0.6 · max(diameter)`), and moving at ball-like speed
(`p1_ball_speed_px = 3.5` to 45 px/frame at low resolution). Both endpoints are
projected to court coordinates, and a crossing fires when
`(cy_now − NET_Y) · (cy_prev − NET_Y) < 0`. This sets `last_cross = f`.

### Start and end via hysteresis

- Go active after `on_n = p1_on_s · sampled_fps` sustained live samples.
- Go dead after `off_n = p1_off_s · sampled_fps` sustained dead samples.
- Transitions are back-dated: `flip_at = f − streak · step`.
- A rising transition sets `rally_start`; a falling one calls
  `close_rally(rally_start, flip_at)`.

### `close_rally`: the emitted boundaries

1. **Drop blips:** discard if the raw `(end_f − start_f) / fps < p1_min_rally_s`.
2. **Tight end:** the hysteresis falling edge lands about
   `p1_cross_window_s + p1_off_s` (around 4 s) after the real end, so instead
   `end_tight = min(end_f, last_cross + p1_end_settle_s · fps)`, where
   `p1_end_settle_s = 0.6 s` approximates ball flight to landing. Without a tracked
   crossing (uncalibrated), the raw transition is used.
3. **Padded start:** `s = max(0, start_f − p1_pad_s · fps)`. The start is padded
   for lead-in; the end stays tight.
4. **Merge:** if `s ≤ last_row.end + p1_merge_gap_s · fps`, extend the previous
   window instead of emitting a new one. Two close rallies can merge; this is an
   accepted recall trade-off.

The result is generous, recall-favouring windows: a padded and merged start and
an end tied to ball flight.

## Refine: keep, start later, end sooner

For each Pass 1 window, refine loads tracking for `window ± window_margin_s`
(2.0 s) and runs four stages (`refine/refine.py`, `refine_window`).

### Tracking coverage check

If the Pass 2 overlay covers less than `min_tracking_coverage` (0.5) of the
window's own frame range (`TrackingData.present`), the window is rejected with
`no_tracking_data` and the classifier does not run. Without this check, a window
that Pass 2 never reached (for example an interrupted job) would produce the same
constant P(live) at every timestep and be rejected as `low_live_ratio`,
indistinguishable from a window that tracking confirmed was dead. The training
side (`train.py`, `build_clip_dataset`) applies the same threshold.

### Stage B: P(live) per timestep

At `feature_hz = 15`, the default `rules` classifier (`refine/classifiers.py`)
computes `P = σ(bias + Σ wᵢ · featureᵢ)` with `rules_bias = −2.0`.

- Positive weights: `cross_recency` 1.0 (exponentially decayed,
  τ = `cross_recency_tau_s`), `crossings_3s` 0.8, `ball_speed_n` 3.0,
  `ball_in_court` 1.5, `frac_visible_1s` 0.6, `reactivity` 0.8,
  `mean_pspeed_n` 1.5, `max_pspeed_n` 0.7, `ready_frac` 0.2.
- Negative weights: `near_player_time_n` −1.4 (ball resting on a player means
  dead), `out_of_court_time_n` −1.4, `facing_away_retreat` −1.6.
- An XGBoost backend exists (F1 about 0.52 against about 0.34 for the rules model
  at the time it was measured). `rules` remains the default and the fallback when
  the model file is missing.

**`facing_away_retreat`** (`features.py`, maximum across players) captures the
clearest "play has stopped" cue: a player moving back toward their own baseline.
Motion must be sustained, at least `baseline_retreat_ratio` (0.7) of the trailing
`baseline_retreat_hold_s` (1.0 s) above `baseline_retreat_speed_mps` (0.5 m/s), so
a quick recovery step mid-rally does not fire it. A pose term (the shoulder-line
normal projected through the homography) raises the score when the body also
faces away from the net. Two keypoints cannot fully resolve a 180° facing
ambiguity, so it acts as a multiplier (`0.5 + 0.5 · facing`), not a gate.

### Decision 1: keep the window?

Stage C (`refine/postprocess.py`) turns P(live) into segments:

- Hysteresis: enter at `enter_thresh` (0.7), exit at `exit_thresh` (0.25).
- Trailing live-ratio filter: a timestep counts as in-rally only if at least
  `live_ratio` (0.7) of the last `ratio_window_s` (2.0 s) were live.
- Merge segments separated by at most `merge_gap_s` (2.0 s).
- Drop segments shorter than `min_rally_s` (2.5 s).

Each surviving segment must then pass every accept gate, or it is recorded with
`verdict = "rejected"` and a reason:

- No segments at all: `low_live_ratio` if `max P < enter_thresh`, otherwise
  `too_short`.
- `mean_p_live < mean_p_live_min` (0.25): `low_live_ratio`.
- `serve_gate_score < serve_gate_min` (0.25 by default): `no_serve`.
- Refined duration below `min_rally_s` after snapping: `too_short`.

One Pass 1 window can split into several rally records, one per surviving
segment, which undoes Pass 1's merging.

**Serve gate score** (`refine/serve_gate.py`, `score_serve`) is a weighted mean of
five components (`pre_serve` 0.25, `contact` 0.35, `flight` 0.15, `landing` 0.10,
`rally_evidence` 0.15):

- **pre_serve:** serve formation held for at least `formation_hold_s` (0.8 s) near
  the segment start.
- **contact:** the ball appears or accelerates (at least
  `serve_contact_speed_mps`, 4.0) within `serve_contact_near_m` (2.0 m) of a
  stationary player at or behind the baseline. This is also the start anchor.
- **flight:** a net crossing `serve_flight_min_s` (0.3 s) to `serve_flight_max_s`
  (2.0 s) after contact, with full credit when the ball travels away from the
  server.
- **landing:** the first bounce lands beyond the kitchen on the receiver's
  diagonal.
- **rally_evidence:** crossing rate normalised to `rally_evidence_x10s` (5.0 per
  10 s), multiplied by the fraction of visible ball beyond the kitchen (divided by
  0.3). This discounts kitchen dinking drills.

### Decision 2: move the start later

```
anchor = gate.contact_t                         # detected serve contact, if found
       else min(t_start, first_crossing − 0.9)  # back off a crossing by serve-flight time
       else t_start                             # Stage C rising edge
refined_start = max(span_lo, anchor − pre_roll_s)   # pre_roll_s = 0.5
```

Because Pass 1 pads the start by `p1_pad_s`, anchoring to the serve contact
usually moves the start later, onto the serve.

### Decision 3: end sooner

`snap_end` walks back from the segment's falling edge to the last meaningful ball
event inside `[start, seg_end]`: the latest bounce, net crossing, or fast-ball
moment (`speed ≥ end_event_speed_mps`, 3.0). If none exists, the falling edge is
kept. Then `refined_end = min(span_hi, event_t + post_roll_s)` with
`post_roll_s = 1.0`.

### Ball position in 3D (ballistic fit)

Projecting the ball's pixel position through the homography assumes it is on the
ground. That is wrong for an airborne ball: a serve toss at about 3 m projects
several metres behind the baseline, so position-based serve checks fail on real
serves even though the pixel track is accurate.

`refine/ballistic.py` fixes this by fitting a gravity parabola
`P0 + V0 t + ½ g t²` to each flight so that it reprojects onto the observed pixels
(reprojection RMS around 3 px). On well-fitting segments, `io_loaders.load_tracking`
replaces the ground-plane `ball_xy` / `ball_height` with the fit
(`use_ballistic_3d`, on by default; `ballistic_max_rms_px`). It costs about 5 s per
clip.

The ball is usually not trackable at the moment of the serve strike, so its first
detection is already mid-flight. `serve_gate._ballistic_contact` extrapolates the
fitted serve arc backwards; when the origin is low and at or behind a baseline,
that point is taken as the strike.

An earlier image-space contact check (`serve_contact_use_pixels`, off by default)
is kept for reference but superseded by the ballistic fit.

**Measured effect** on a labelled keep/drop benchmark of 7 clips: F1 went from
0.561 to 0.643 with the ballistic fit, and to 0.673 with arc contact recovery,
with both precision and recall improving.

### Runtime overrides and the grading loop

- Any `RefineConfig` field, such as `serve_gate_min` or `mean_p_live_min`, can be
  overridden in `data/refine/config.override.yaml`. The refiner loads it on every
  run (`pipeline._refine_after_pass2`, `annotate.py` `_refine_records`). These
  values are not in `learned_params`.
- In the testing UI, each Pass 1 window is listed with its verdict and can be
  graded keep or drop (`window_grades` table, matched to windows by frame-span IoU
  ≥ 0.5 because window IDs change on each run). Grades drive a keep/drop accuracy
  readout and a `serve_gate_min` recommendation (`refine/recommend.py`). Applying a
  recommendation writes the override file; nothing is applied automatically. The
  endpoints are in `api/annotate.py`, and the UI is `PassTab` and
  `Recommendation` in `frontend/src/Workspace.jsx`.

### Output record

Each segment produces
`{pass1_window_id, pass1_start/end(_frame), verdict, rejection_reason,
refined_start/end(_frame), confidence{mean_p_live, serve_gate_score,
net_crossings_per_10s}, splits[]}` in `data/refine/<video_id>.json`.
`rejection_reason` is one of `no_tracking_data`, `low_live_ratio`, `no_serve`,
`too_short`.

## Summary

Pass 1 is a net-crossing recency gate with hysteresis. It emits windows with a
padded start and a ball-flight end, merged when close together. Refine computes
P(live) per timestep, smooths it into segments, applies four accept gates (live
ratio, mean P, serve gate, minimum duration), then re-anchors the start to the
serve contact minus 0.5 s and the end to the last real ball event plus 1 s. Its
output is advisory and is not written to `rally_labels`.

## Key files and symbols

| File | Symbols |
|---|---|
| `backend/app/cv/pass1.py` | `run_pass1`, `close_rally`, `_match_fast`, `_court_mask` |
| `backend/app/cv/pass2.py` | Pass 2 detection overlay (no boundaries) |
| `backend/app/refine/refine.py` | `refine_window`, `refine_video`, `run_refine_json` |
| `backend/app/refine/classifiers.py` | `RulesClassifier`, `XGBoostClassifier` |
| `backend/app/refine/postprocess.py` | Stage C: hysteresis, live-ratio filter, segments |
| `backend/app/refine/serve_gate.py` | `score_serve`, `snap_end`, `_ballistic_contact` |
| `backend/app/refine/ballistic.py` | Gravity-arc fit for 3D ball position |
| `backend/app/refine/config.py` | `RefineConfig` (all refine thresholds and weights) |
| `backend/app/pipeline.py` | `_refine_after_pass2` |
| `backend/app/learning/params.py` | `LearnedParams` (Pass 1 `p1_*` parameters) |

Last checked against the code: 2026-11-16.
