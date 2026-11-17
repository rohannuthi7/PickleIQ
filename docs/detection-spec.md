# Detection spec: Pass 1 and Pass 2

How the two-pass computer-vision pipeline detects rallies, players and the ball.
For the end-to-end flow (including the human Review stage and coaching), see
[`pipeline-overview.md`](pipeline-overview.md). For the exact rally start/end
logic, see [`rally-boundaries.md`](rally-boundaries.md).

> **Design note.** Pass 1 is the only source of rally start/end boundaries. It
> emits generous, recall-favouring windows (padded and merged). Pass 2 does not
> compute boundaries; it produces an accurate per-frame player and ball overlay
> inside the windows the user selects. The older rally-decision code
> (`rally_logic`, terminal-event end detection, learned start/end offsets,
> serve-vs-pass rejection, and the `rally_end` / `serve_start` model hooks) is
> still in the tree but is not called. The boundary-offset learning loop is
> paused for the same reason: the offsets are not applied anywhere.

## Orchestration

- `backend/app/pipeline.py` runs everything. On upload it loads the latest global
  `LearnedParams` (`learning/params.py`) and the effective court calibration
  (clip-local override, otherwise the newest global one), then runs Pass 1.
- Pass 2 does not run automatically. After the user reviews Pass 1's candidate
  windows, the selected windows are submitted as one Pass 2 job
  (`api/rally_analysis.py`, `process_selections`).
- Both passes take the same `params` dict. All thresholds below are keys in it and
  can be tuned globally (see [Learnable state](#learnable-state)).
- Calibration is a homography (image pixels to court metres) plus an estimated
  focal length. Pass 2 requires it; Pass 1 uses it for its primary signal and
  falls back to motion without it.

## Pass 1: coarse gate

Entry point: `backend/app/cv/pass1.py`, `run_pass1`.

**Goal:** cheaply answer "is a rally roughly happening here?" over 100% of the
footage, at low resolution (`p1_scale` = 0.4) and a reduced frame rate
(`p1_frame_step` = 3). The output is the authoritative rally `(start, end)`
windows, stored as `rally_labels` with `source='pass1'`.

**Boundaries.** `close_rally` pads each raw window by `p1_pad_s` per side and
merges it into the previous window when the padded windows touch (within
`p1_merge_gap_s`). Windows whose raw length is under `p1_min_rally_s` are dropped.
The result fully contains each rally, at the cost of loose edges and occasionally
two close rallies merged into one window.

**Signals**

1. **Net crossings (primary, needs calibration).** A fast, ball-sized blob of
   consistent size is matched frame to frame (`_match_fast`), projected to court
   coordinates, and flagged when its court-Y sign flips across the net line. The
   gate stays open while `(frame - last_cross) <= p1_cross_window_s`.
2. **Motion energy against an adaptive baseline (fallback, no calibration).** Mean
   frame difference inside the court mask is compared with a slow EMA baseline.
   The gate fires when `motion > max(p1_motion_floor, baseline * p1_motion_ratio)`
   and a fast blob was seen within `p1_ball_recent_s`. The baseline adapts only
   during dead time.
3. **Sparse serve-formation pose checks** every `p1_spot_check_s`
   (`is_serve_formation`). These are logged but do not drive the gate.

**Court mask** (`_court_mask`): all Pass 1 signals are restricted to the court and
a margin, so neighbouring courts and spectators cannot open the gate. Pass 1's
ball detector does not receive player boxes; it is a gate, not a tracker.

**Windowing:** active/dead transitions use hysteresis (`p1_on_s` / `p1_off_s`
sustained sample counts).

**Trace:** per-sample features (`me` motion, `mt` threshold, `ball`, `cross`,
`sf`, players) are written to the `pass1` overlay. They are the training data for
a learned gate classifier (`backend/scripts/train_gate.py`).

## Pass 2: detection overlay

Entry point: `backend/app/cv/pass2.py`, `run_pass2`.

**Goal:** inside the selected windows only (about 20% of frames), at full frame
rate, produce an accurate per-frame overlay of player and ball positions. Pass 2
writes no `rally_labels`; it clears any stale `source='pass2'` rows and writes
only the `pass2` overlay.

**Processing extent.** Pass 2 merges consecutive windows (raw gap at most
`p1_merge_gap_s`) and pads by `p1_pad_s`. This only decides where detection runs.

**Per frame, per window**

- **Players:** `PoseTracker` (YOLO11-pose and ByteTrack) gives court positions and
  mean player speed per frame. Serve-formation checks produce `serve_frames`.
- **Ball observation, in priority order:**
  1. User `ball_corrections` for that frame (confidence 1.0).
  2. An external track posted to `/api/videos/{id}/ball_track`
     (see [`external-ball-tracker.md`](external-ball-tracker.md)).
  3. **TrackNetV2**, when `TRACKNET_ENABLED` is set and weights are present. It is
     off by default in code and enabled on the EC2 worker
     (`deploy/restart-ec2.sh` sets `TRACKNET_ENABLED=1`), which uses a checkpoint
     fine-tuned on hand-labelled frames from this camera setup. Pretrained
     tennis/badminton weights did not transfer (about 5x slower, no gain). The
     fine-tuned checkpoint did: on a real 7,413-frame clip where the classical
     detector observed the ball in about 0.5% of frames, TrackNet observed it in
     2,525 frames (34%), arcs formed in every window, and 4 of the clip's 13 Pass 1
     windows changed from rejected to kept on ball evidence alone. Weights sync
     from S3 (`storage_s3.upload_tracknet_weights` / `download_tracknet_weights`,
     `cloud_ingest._sync_tracknet_weights`); `backend/scripts/upload_tracknet_weights.py`
     ships a new checkpoint, and the worker picks it up on its next poll.
  4. The classical `MotionBallDetector`, run permissively (no player-box
     exclusion), followed by `TrajectoryBallSelector` (`ball_track.py`). The
     selector scores each blob by its fit to the recent trajectory and suppresses
     blobs on a player's feet unless they move at least `ball_hit_speed_px`. A
     court-region filter (`_in_court_region`) rejects projections more than about
     2.5–3 m outside the court.
- **Arc-fit noise filter** (`ball_filter.refine_track`, per window): splits the
  selected detections at time gaps (`ball_arc_gap_frames`) and sharp turns, fits a
  quadratic to each segment of at least `ball_arc_min_pts` points with RANSAC, and
  drops points that lie on no fitted arc. When a window fits no arcs at all (for
  example a sparse far-baseline track), the raw detections are kept, because there
  is no trajectory to judge them against.
- **Kalman smoothing** (`smooth_track`, `max_gap` about 0.4 s, `ball_kalman_gate2`
  outlier rejection) bridges short gaps and rejects physically implausible jumps.
- **3D reconstruction** (`reconstruct_world`): apparent ball diameter gives height
  above the court; the homography gives court x/y. This size-to-depth step is the
  largest noise source for the far-baseline ball, which is only a few pixels wide.
- **Player numbering:** `assign_numbers_at_serve` gives players stable numbers 1–4
  at serve frames. This is for display only.

**Bounce and hit markers** (`bounce.py`, `detect_events` with `model=None`): local
height minima near the court plane mark bounces; sharp reversals in court-depth
velocity mark hits. An out-of-court bounce is emitted only for a real detection
with known height within `out_max_dist_m` of a line. These markers are display
annotations and do not end rallies.

**Output:** the per-frame `pass2` overlay (players, ball, world x/y/height,
`pspeed`, bounce/hit events). Coaching, refine, and training scripts read it.

## Learnable state

- **`LearnedParams`** (`learning/params.py`): global, versioned parameters, each
  clamped to bounds defined in `DEFAULTS` / `BOUNDS`. They are tuned by the
  discrepancy-report loop and validated before commit. Groups include:
  - Pass 1 gate: `p1_frame_step`, `p1_scale`, `p1_motion_ratio`,
    `p1_motion_floor`, `p1_ball_recent_s`, `p1_ball_speed_px`, `p1_on_s`,
    `p1_off_s`, `p1_pad_s`, `p1_merge_gap_s`, `p1_cross_window_s`,
    `p1_min_rally_s`, `p1_spot_check_s`.
  - Pass 2 ball and bounce: `bounce_max_height_m`, `hit_min_height_m`,
    `player_radius_m`, `min_event_gap_s`, `net_fail_height_m`, `out_max_dist_m`.
  - Rally start/end (inactive while the rally-decision code is not called):
    `serve_speed_mps`, `serve_behind_margin_m`, `serve_formation_lookback_s`,
    `confirm_window_s`, `player_active_mps`, `p2_reject_passes`,
    `start_offset_frames`, `end_offset_frames`.
  - The parameters that currently shape Pass 1 windows most are `p1_pad_s`,
    `p1_merge_gap_s`, `p1_min_rally_s` and `p1_cross_window_s`.
- **`rally_end` model** (BounceGRU): registered in `learning/model_registry.py`
  with a shadow/live promotion step and trained by
  `backend/scripts/train_bounce.py`. It needs at least three labelled clips to
  validate. It is not loaded while `detect_events` runs with `model=None`.
- **`serve_start` model:** a hook exists in `detect_start`; no model is built.
- **TrackNet:** fine-tuned with `backend/scripts/export_tracknet_dataset.py` and
  `backend/scripts/finetune_tracknet.py` on hand-labelled `ball_corrections`.

## Known limitations

- **Far-side ball.** Size-to-depth noise makes far-baseline height unreliable. The
  fine-tuned TrackNet improves observation coverage but not this.
- **Classical-detector noise** can still produce a false "return", so some
  between-point ball movement survives as a false-positive window.
- **Faults without a ball signature** (foot faults, kitchen and service-motion
  faults) are out of scope.

## Key files

| File | Role |
|---|---|
| `backend/app/pipeline.py` | Orchestration |
| `backend/app/cv/pass1.py` | Pass 1 gate |
| `backend/app/cv/pass2.py` | Pass 2 orchestration, window building, ball filters |
| `backend/app/cv/ball_coarse.py` | Classical ball detector |
| `backend/app/cv/ball_tracknet.py` | TrackNetV2 inference |
| `backend/app/cv/ball_track.py` | Trajectory selector |
| `backend/app/cv/ball_filter.py` | Velocity gate, arc RANSAC, hit/bounce classification |
| `backend/app/cv/bounce.py` | World reconstruction, `detect_events`, BounceGRU |
| `backend/app/cv/rally_logic.py` | Rally-decision code (not called) |
| `backend/app/cv/court.py` | Court geometry, homography, `in_court`, `side_of_net` |
| `backend/app/cv/players.py` | YOLO11-pose tracking, `is_serve_formation` |
| `backend/app/learning/params.py` | Tunable parameters |
