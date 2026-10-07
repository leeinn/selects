"""Read existing index metadata; never decode or open originals for review."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from selects.config import FolderConfig
from selects.db import init_db, session_scope
from selects.db.models import (
    AestheticScore,
    ClassicalScore,
    Embedding,
    Moment,
    MomentMember,
    Photo,
    PhotoCategory,
    PhotoPerson,
    PipelineState,
    Story,
    StoryItem,
    Swipe,
)
from selects.ml.aesthetic import rank_score
from selects.ml.curation import CuratedPhoto, _apply_taste_blend
from selects.util import KEEP_DECISIONS

from .schema import SCHEMA_VERSION


def check_state(cfg: FolderConfig) -> None:
    """State and preview roots must be real directories inside this library."""
    for path in (cfg.state_dir, cfg.previews_dir, cfg.thumbs_dir, cfg.state_dir / "creative"):
        if path.is_symlink() or not path.resolve().is_relative_to(cfg.folder):
            raise ValueError(f"unsafe state directory: {path}")
    if cfg.db_path.is_symlink():
        raise ValueError("unsafe index.db symlink")


def _finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def load_photos(cfg: FolderConfig) -> dict[int, dict]:
    check_state(cfg)
    if not cfg.db_path.is_file():
        raise ValueError(f"library is not indexed; run: selects index '{cfg.folder}'")
    if cfg.db_path.is_symlink():
        raise ValueError("index.db must not be a symlink")
    Session = init_db(cfg.db_path)
    with session_scope(Session) as s:
        rows = (
            s.query(
                Photo,
                ClassicalScore,
                Embedding.aesthetic_iqa,
                AestheticScore,
                Swipe.decision,
                PhotoCategory.primary_category,
                PipelineState,
            )
            .outerjoin(ClassicalScore, ClassicalScore.photo_id == Photo.id)
            .outerjoin(Embedding, Embedding.photo_id == Photo.id)
            .outerjoin(AestheticScore, AestheticScore.photo_id == Photo.id)
            .outerjoin(Swipe, Swipe.photo_id == Photo.id)
            .outerjoin(PhotoCategory, PhotoCategory.photo_id == Photo.id)
            .outerjoin(PipelineState, PipelineState.photo_id == Photo.id)
            .order_by(Photo.id)
            .all()
        )
        photos = {}
        ranked = []
        for p, tech, iqa, aest, swipe, category, stage in rows:
            preview = cfg.state_dir / p.preview_path if p.preview_path else None
            safe = preview is not None and preview.resolve().is_relative_to(cfg.previews_dir)
            score = _finite(
                rank_score(
                    _finite(aest.ap25_score) if aest else None,
                    _finite(aest.nima_score) if aest else None,
                    _finite(iqa),
                    ap_w=cfg.ap_weight,
                    nima_w=cfg.nima_weight,
                )
            )
            photos[p.id] = {
                "photo_id": p.id,
                "filename": Path(p.path).name,
                "path": p.path,
                "sha256": p.sha256,
                "format": p.format,
                "preview_path": str(preview.resolve()) if safe and preview.is_file() else None,
                "preview_status": "ready" if safe and preview.is_file() else "missing_or_unsafe",
                "taken_at": p.taken_at.isoformat() if p.taken_at else None,
                "gps_lat": p.gps_lat,
                "gps_lon": p.gps_lon,
                "aesthetic": score,
                "selects_score": score,
                "taste": None,
                "hyperiqa": _finite(aest.ap25_score) if aest else None,
                "iqa": _finite(iqa),
                # Laplacian variance, not a normalized artistic quality score.
                "sharpness": _finite(tech.blur) if tech else None,
                "exposure": _finite(tech.exposure) if tech else None,
                "luma_mean": _finite(tech.luma_mean) if tech else None,
                "clipped_ratio": _finite(tech.clipped_high) if tech else None,
                "auto_reject": bool(tech.auto_reject) if tech else False,
                "reject_reason": tech.reject_reason if tech else None,
                "faces_count": tech.faces_count if tech else None,
                "eyes_open_ratio": _finite(tech.eyes_open_ratio) if tech else None,
                "user_decision": swipe,
                "category": category,
                "moment_id": None,
                "moment_rank": None,
                "moment_primary": False,
                "story_id": None,
                "story_ids": [],
                "story_scenes": [],
                "person_ids": [],
                "classical_done": bool(stage and stage.classical_done),
                "embedding_done": bool(stage and stage.embedding_done),
            }
            ranked.append(
                CuratedPhoto(
                    p.id, p.sha256 or "", photos[p.id]["taken_at"], score, None, None, None
                )
            )
        _apply_taste_blend(s, ranked)
        for r in ranked:
            photos[r.photo_id]["taste"] = r.taste
            if r.final is not None:
                photos[r.photo_id]["selects_score"] = r.final

        for mm, m in s.query(MomentMember, Moment).join(Moment).order_by(Moment.id).all():
            p = photos.get(mm.photo_id)
            if p and p["moment_id"] is None:
                p.update(
                    moment_id=m.id,
                    moment_rank=mm.rank,
                    moment_primary=mm.photo_id == m.primary_photo_id,
                )
        stories = s.query(Story).order_by(Story.day, Story.id).all()
        by_day = {st.day: st.id for st in stories if len(st.day) == 10 and st.day[4] == "-"}
        for p in photos.values():
            day = (p["taken_at"] or "")[:10]
            if day in by_day:
                p["story_ids"].append(by_day[day])
        for it in s.query(StoryItem).order_by(StoryItem.story_id, StoryItem.rank).all():
            p = photos.get(it.photo_id)
            if not p:
                continue
            if it.story_id not in p["story_ids"]:
                p["story_ids"].append(it.story_id)
            if it.scene_label:
                p["story_scenes"].append([it.story_id, it.scene_label])
        for pid, person_id in s.query(PhotoPerson.photo_id, PhotoPerson.person_id).all():
            if pid in photos:
                photos[pid]["person_ids"].append(person_id)
        for p in photos.values():
            p["story_id"] = p["story_ids"][0] if p["story_ids"] else None
    return photos


def _quality(p: dict) -> tuple:
    return (
        p["user_decision"] in KEEP_DECISIONS,
        p["moment_primary"],
        p["selects_score"] if p["selects_score"] is not None else -1,
        p["sharpness"] if p["sharpness"] is not None else -1,
        -p["photo_id"],
    )


def make_batches(candidates: list[dict], batch_size: int = 50) -> list[dict]:
    """Keep Moments together; split oversized bursts into adjacent bounded batches."""
    if not 30 <= batch_size <= 80:
        raise ValueError("batch_size must be between 30 and 80")
    groups = defaultdict(list)
    for p in candidates:
        groups[
            ("moment", p["moment_id"]) if p["moment_id"] is not None else ("photo", p["photo_id"])
        ].append(p)
    ordered = sorted(
        groups.values(),
        key=lambda g: (
            g[0]["story_id"] or 0,
            min(p["taken_at"] or "" for p in g),
            g[0]["photo_id"],
        ),
    )
    batches, pending = [], []

    def flush(split=False):
        if pending:
            batches.append(
                {
                    "batch_id": len(batches) + 1,
                    "photo_ids": [p["photo_id"] for p in pending],
                    "split_moment": split,
                }
            )
            pending.clear()

    for group in ordered:
        group.sort(key=lambda p: (not p["moment_primary"], p["moment_rank"] or 0, p["photo_id"]))
        if len(pending) + len(group) > batch_size:
            flush()
        if len(group) > batch_size:
            for start in range(0, len(group), batch_size):
                pending.extend(group[start : start + batch_size])
                flush(split=True)
        else:
            pending.extend(group)
    flush()
    return batches


def generate_candidates(
    cfg: FolderConfig,
    *,
    percentile: float = 50,
    max_candidates: int | None = None,
    include_story: bool = True,
    include_maybe: bool = False,
    batch_size: int = 50,
    moment_id: int | None = None,
    story_id: int | None = None,
) -> dict:
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100 (50 keeps the top half)")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    photos = load_photos(cfg)
    eligible = {
        pid: p
        for pid, p in photos.items()
        if p["user_decision"] in KEEP_DECISIONS
        or (not p["auto_reject"] and p["user_decision"] != "reject")
    }
    if moment_id is not None:
        eligible = {pid: p for pid, p in eligible.items() if p["moment_id"] == moment_id}
    if story_id is not None:
        eligible = {pid: p for pid, p in eligible.items() if story_id in p["story_ids"]}
    scores = [p["aesthetic"] for p in eligible.values() if p["aesthetic"] is not None]
    sharp = [p["sharpness"] for p in eligible.values() if p["sharpness"] is not None]
    threshold = float(np.percentile(scores, percentile)) if scores else None
    high_aesthetic = float(np.percentile(scores, 90)) if scores else None
    high_sharpness = float(np.percentile(sharp, 90)) if sharp else None
    reasons = defaultdict(set)
    protected = set()

    def add(pid, reason, protect=False):
        reasons[pid].add(reason)
        if protect:
            protected.add(pid)

    moments, stories, scenes = defaultdict(list), defaultdict(list), defaultdict(list)
    for pid, p in eligible.items():
        if p["user_decision"] in KEEP_DECISIONS:
            add(pid, "user_keep", True)
        if p["aesthetic"] is None:
            add(pid, "unscored")
        elif threshold is not None and p["aesthetic"] >= threshold:
            add(pid, "aesthetic_percentile")
        if (
            high_aesthetic is not None
            and p["aesthetic"] is not None
            and p["aesthetic"] >= high_aesthetic
        ):
            add(pid, "high_aesthetic", True)
        if (
            high_sharpness is not None
            and p["sharpness"] is not None
            and p["sharpness"] > 0
            and p["sharpness"] >= high_sharpness
        ):
            add(pid, "high_sharpness", True)
        mean = p["luma_mean"]
        if mean is not None and (mean < 0.32 or mean > 0.78) and not p["auto_reject"]:
            add(pid, "extreme_light", True)
        if include_maybe or moment_id is not None:
            add(pid, "expanded_review")
        if p["moment_id"] is not None:
            moments[p["moment_id"]].append(p)
        for sid in p["story_ids"]:
            stories[sid].append(p)
        for sid, label in p["story_scenes"]:
            scenes[(sid, label)].append(p)
    for members in moments.values():
        primary = next((p for p in members if p["moment_primary"]), max(members, key=_quality))
        add(primary["photo_id"], "moment_primary", True)
    if include_story:
        for members in stories.values():
            represented = [p for p in members if p["photo_id"] in reasons]
            add(max(represented or members, key=_quality)["photo_id"], "story_representative", True)
        for members in scenes.values():
            add(max(members, key=_quality)["photo_id"], "scene_representative", True)

    by_sha = defaultdict(list)
    for p in eligible.values():
        if p["sha256"]:
            by_sha[p["sha256"]].append(p["photo_id"])
    # Existing content hashes suffice; no new duplicate detector is run.
    for ids in by_sha.values():
        matched = [pid for pid in ids if pid in reasons]
        if len(matched) > 1:
            keep = max(matched, key=lambda pid: _quality(eligible[pid]))
            for pid in matched:
                if pid != keep and pid not in protected:
                    reasons.pop(pid)

    candidates = [
        dict(
            eligible[pid],
            candidate_reasons=sorted(why),
            exact_duplicate_ids=sorted(by_sha.get(eligible[pid]["sha256"], [])),
        )
        for pid, why in reasons.items()
    ]
    pool_size = len(candidates)
    if max_candidates is not None and pool_size > max_candidates:
        if len(protected) > max_candidates:
            raise ValueError(
                f"max_candidates={max_candidates} cannot preserve {len(protected)} protected photos; raise the cap"
            )
        # Fill discretionary slots round-robin across stories/days, then quality.
        buckets = defaultdict(list)
        for p in sorted(candidates, key=_quality, reverse=True):
            if p["photo_id"] not in protected:
                buckets[p["story_id"] or (p["taken_at"] or "")[:10]].append(p)
        kept = [p for p in candidates if p["photo_id"] in protected]
        depth = 0
        while len(kept) < max_candidates:
            for key in sorted(buckets, key=str):
                if depth < len(buckets[key]) and len(kept) < max_candidates:
                    kept.append(buckets[key][depth])
            depth += 1
        candidates = kept
    candidates.sort(key=lambda p: (p["story_id"] or 0, p["taken_at"] or "", p["photo_id"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "library": str(cfg.folder),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "options": {
            "percentile": percentile,
            "max_candidates": max_candidates,
            "include_story": include_story,
            "include_maybe": include_maybe,
            "batch_size": batch_size,
            "moment_id": moment_id,
            "story_id": story_id,
        },
        "summary": {
            "indexed": len(photos),
            "eligible": len(eligible),
            "pool_size": pool_size,
            "candidates": len(candidates),
            "protected": len(protected),
            "omitted_by_cap": pool_size - len(candidates),
            "missing_previews": sum(p["preview_path"] is None for p in candidates),
        },
        "candidates": candidates,
        "batches": make_batches(candidates, batch_size),
    }
