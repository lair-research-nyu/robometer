#!/usr/bin/env python3
"""LeRobot v3.0 dataset loader for Robometer model training.

Reads a LeRobot v3.0 root (``meta/`` + ``data/`` parquet + per-camera
``videos/`` mp4) and emits one RBM trajectory per episode. Designed for the
datasets produced by ultra-tools' rrd_to_lerobot pipeline (towerstack), which
add a per-episode ``exit_type`` column (success/failure) on top of the stock
LeRobot layout; stock datasets without it are treated as all-successful demos.

Episodes live inside *concatenated* chunk videos (one mp4 holds many episodes,
located by chunk/file index + ``from_timestamp``), so there is no per-episode
video file to hand the converter. Instead each trajectory's ``frames`` is a
lazy callable (see create_trajectory_video_optimized) that seeks into the
chunk mp4 with torchcodec and decodes <= max_frames uniformly-spaced frames.
The callable carries only plain picklable state so it survives the converter's
spawn-based worker pool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq

# exit_type -> RBM quality_label
QUALITY_LABEL_MAP = {"success": "successful", "failure": "failure"}


class LeRobotEpisodeFrameLoader:
    """Lazy per-episode frame decoder for one camera of a LeRobot v3.0 dataset.

    Picklable (plain paths/ints/floats only); the torchcodec decoder is created
    inside __call__ so spawn workers each build their own.
    """

    def __init__(self, video_path: str, from_ts: float, fps: float,
                 n_frames: int, max_frames: int) -> None:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self.video_path = video_path
        self.from_ts = float(from_ts)
        self.fps = float(fps)
        self.n_frames = int(n_frames)
        self.max_frames = int(max_frames)

    def __call__(self) -> np.ndarray:
        """Decode <= max_frames uniformly-spaced frames as (N, H, W, 3) uint8."""
        from torchcodec.decoders import VideoDecoder

        n = min(self.n_frames, self.max_frames) if self.max_frames > 0 else self.n_frames
        idx = np.unique(np.round(np.linspace(0, self.n_frames - 1, n)).astype(int))
        # Mid display-window timestamps so float boundaries never land on the
        # neighbouring frame (same convention as the value-estimation reader).
        ts = self.from_ts + (idx + 0.5) / self.fps

        dec = VideoDecoder(self.video_path)
        batch = dec.get_frames_played_at(seconds=ts.tolist())
        return batch.data.permute(0, 2, 3, 1).contiguous().numpy()  # NCHW -> NHWC


def _load_episode_meta(root: Path):
    """All per-episode rows from meta/episodes/*, sorted by episode_index."""
    import pandas as pd

    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {root}/meta/episodes")
    frames = []
    for f in files:
        tbl = pq.read_table(f)
        keep = [c for c in tbl.column_names if not c.startswith("stats/")]
        frames.append(tbl.select(keep).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def _episode_task(tasks_cell) -> str:
    """The episode's single task string (rrd_to_lerobot writes one task/episode)."""
    if tasks_cell is None:
        return ""
    seq = list(tasks_cell)
    return str(seq[0]) if seq else ""


def load_lerobot_dataset(
    base_path: str,
    data_source: str = "towerstack",
    camera: str = "head",
    subtask: str = "",
    max_frames: int = 64,
) -> Dict[str, List[Dict]]:
    """Load a LeRobot v3.0 dataset and organize trajectories by task.

    Args:
        base_path: LeRobot v3.0 root (contains meta/, data/, videos/).
        data_source: RBM data_source tag — must match the entry added to
            robometer/data/dataset_success_cutoff.txt.
        camera: which observation.images.<camera> video stream to use.
        subtask: optional exact-match filter on the episode task string.
        max_frames: frames decoded per episode (uniform subsample).

    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries.
    """
    root = Path(base_path)
    if not root.exists():
        raise FileNotFoundError(f"LeRobot dataset path not found: {root}")

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    meta = _load_episode_meta(root)

    key = f"videos/observation.images.{camera}"
    for col in (f"{key}/chunk_index", f"{key}/file_index", f"{key}/from_timestamp"):
        if col not in meta:
            raise KeyError(f"camera {camera!r}: missing column {col!r} in episode metadata")

    print(f"Loading LeRobot dataset from: {root}")
    print(f"  camera={camera} fps={fps} episodes={len(meta)} data_source={data_source}")

    task_data: Dict[str, List[Dict]] = {}
    n_skipped = 0
    # dict records, not itertuples: the slash-named locator columns are not
    # valid identifiers and itertuples would rename them positionally.
    for row in meta.to_dict(orient="records"):
        task = _episode_task(row.get("tasks"))
        if subtask and task != subtask:
            continue

        exit_type = str(row.get("exit_type", "success")).lower()
        quality_label = QUALITY_LABEL_MAP.get(exit_type, "failure")

        chunk = int(row[f"{key}/chunk_index"])
        file = int(row[f"{key}/file_index"])
        video_path = (root / "videos" / f"observation.images.{camera}" /
                      f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4")
        try:
            frames = LeRobotEpisodeFrameLoader(
                video_path=str(video_path),
                from_ts=float(row[f"{key}/from_timestamp"]),
                fps=fps,
                n_frames=int(row["length"]),
                max_frames=max_frames,
            )
        except FileNotFoundError as e:
            print(f"Skipping episode {row['episode_index']}: {e}")
            n_skipped += 1
            continue

        trajectory = {
            "id": f"{data_source}_ep{int(row['episode_index']):05d}",
            "frames": frames,
            "task": task,
            "is_robot": True,
            "quality_label": quality_label,
            "data_source": data_source,
        }
        task_data.setdefault(task, []).append(trajectory)

    n_total = sum(len(v) for v in task_data.values())
    print(f"Loaded {n_total} trajectories across {len(task_data)} tasks"
          + (f" ({n_skipped} skipped)" if n_skipped else ""))
    return task_data
