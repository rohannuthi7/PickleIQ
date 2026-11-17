# Connecting an external ball tracker

An external tracker can run independently (any language or framework, offline is
fine) and produce a per-frame ball track. POST that track to the API and Pass 2
uses it ahead of its built-in detectors. Nothing else in the pipeline needs to
change: better ball positions flow into height reconstruction, bounce and hit
events, net-crossing counts, and the refine stage's serve gate.

## Ball-source priority in Pass 2

`manual correction` (per frame) > `external track` > `TrackNet` (when enabled) >
`classical MotionBallDetector` (fallback).

Frames missing from the external track fall back to the next source, so a partial
track (for example, only the frames the tracker is confident about) is fine.

## Endpoints

```
POST /api/videos/{video_id}/ball_track
Content-Type: application/json

{
  "source": "my_tracker_v1",                // free-text label for the producer
  "track": [
    { "frame": 401, "x": 962.4, "y": 540.1, "diameter": 7.8, "confidence": 0.94 },
    { "frame": 402, "x": 968.0, "y": 536.7, "diameter": 7.5, "confidence": 0.91 }
  ]
}
```

- `GET /api/videos/{video_id}/ball_track` returns `{ source, n_frames, created_at }`.
- `DELETE /api/videos/{video_id}/ball_track` removes the track.

Video id, dimensions, fps and frame count are available from `GET /api/videos` and
`GET /api/videos/{video_id}`.

## Coordinate and frame conventions

- **`frame`:** 0-based frame index, decoded in order (the same indexing Pass 2
  uses). Decode the exact stored file (`GET /api/videos/{video_id}/stream`) so frame
  numbers line up.
- **`x`, `y`:** ball centre in full-resolution image pixels, origin top-left (the
  video's native width and height, not a resized or letterboxed frame).
- **`diameter`** (optional but important): apparent ball diameter in pixels. It
  drives the size-to-depth height reconstruction, which bounce detection depends
  on. Without it, height is unknown, but good `x, y` still improves net-crossing
  counts and the serve gate, which only need position.
- **`confidence`** (optional, default 0.9): 0 to 1. Informational only; to fall
  back to the built-in detectors for a frame, omit that frame.

## After uploading

Re-run Pass 2 so it uses the track: `POST /api/videos/{video_id}/reprocess?which=pass2`
(also available as "Regenerate video" in the testing UI). The server logs
`pass2 <id>: using external ball track (N frames)` when it picks the track up.

## Minimal Python example

```python
import requests

BASE = "http://127.0.0.1:8420/api"
video_id = requests.get(f"{BASE}/videos").json()[0]["id"]

# Your tracker returns a list of {frame, x, y, diameter?, confidence?}.
track = run_my_tracker(video_id)

requests.post(f"{BASE}/videos/{video_id}/ball_track",
              json={"source": "my_tracker_v1", "track": track}).raise_for_status()
requests.post(f"{BASE}/videos/{video_id}/reprocess", params={"which": "pass2"})
```

## In-process alternative

To call a model live inside Pass 2 instead of uploading a precomputed track,
implement the same interface as `TrackNetBall` in
`backend/app/cv/ball_tracknet.py` (an `available` flag and
`infer_triplet(frames) -> (x, y, diameter, confidence) | None`) and add it to the
priority chain in `backend/app/cv/pass2.py`. The upload path is simpler: it keeps
the tracker out of the CV process and works from any language.
