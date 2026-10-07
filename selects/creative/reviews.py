"""Atomic review imports and human/machine reports, entirely under .selects."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from selects.config import FolderConfig

from .candidates import check_state, load_photos
from .schema import DECISIONS, SCHEMA_VERSION, SELECTED, CreativeReview, ReviewInput


def creative_path(cfg: FolderConfig, filename: str) -> Path:
    check_state(cfg)
    if Path(filename).name != filename:
        raise ValueError("creative filenames must be basenames")
    path = cfg.state_dir / "creative" / filename
    if path.is_symlink() or not path.resolve().is_relative_to(cfg.state_dir / "creative"):
        raise ValueError(f"unsafe creative output: {path}")
    return path


def write_text(cfg: FolderConfig, filename: str, content: str) -> Path:
    path = creative_path(cfg, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only our state file; a failed import never truncates saved reviews.
    fd, tmp = tempfile.mkstemp(prefix=".creative-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return path


def write_json(cfg: FolderConfig, filename: str, data: dict) -> Path:
    return write_text(
        cfg, filename, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def _check_identity(review: CreativeReview, photos: dict[int, dict]) -> None:
    photo = photos.get(review.photo_id)
    if photo is None:
        raise ValueError(f"unknown photo_id: {review.photo_id}")
    if photo["filename"] != review.filename:
        raise ValueError(f"filename mismatch for photo_id {review.photo_id}")
    if review.sha256 is not None and review.sha256 != photo["sha256"]:
        raise ValueError(f"photo {review.photo_id} changed since review (sha256 mismatch)")


def validate_selection(reviews: list[dict], photos: dict[int, dict]) -> None:
    moments, hashes = defaultdict(list), defaultdict(list)
    for row in reviews:
        if row["decision"] not in SELECTED:
            continue
        p = photos[row["photo_id"]]
        if p["sha256"]:
            hashes[p["sha256"]].append(row)
        if p["moment_id"] is not None:
            moments[p["moment_id"]].append(row)
    for rows in hashes.values():
        if len(rows) > 1:
            raise ValueError(
                "exact duplicates cannot both be hero/grade/story: "
                + ", ".join(str(r["photo_id"]) for r in rows)
            )
    for mid, rows in moments.items():
        if len(rows) > 2:
            raise ValueError(
                f"Moment {mid} has more than two selected photos; demote siblings to maybe/reject"
            )
        if len(rows) == 2 and not any(
            r.get("moment_exception") and r.get("moment_exception_reason") for r in rows
        ):
            raise ValueError(
                f"Moment {mid} needs a justified moment_exception to retain two photos"
            )


def load_reviews(
    cfg: FolderConfig, photos: dict[int, dict], *, replaced_ids: set[int] | None = None
) -> list[dict]:
    path = creative_path(cfg, "reviews.json")
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION or data.get("library") != str(cfg.folder):
        raise ValueError("saved review schema/library mismatch")
    result, seen = [], set()
    for row in data["reviews"]:
        payload = {k: v for k, v in row.items() if k not in ("created_at", "updated_at")}
        review = CreativeReview.model_validate(payload)
        if review.photo_id not in (replaced_ids or set()):
            _check_identity(review, photos)
        if review.photo_id in seen:
            raise ValueError(f"duplicate saved photo_id: {review.photo_id}")
        seen.add(review.photo_id)
        result.append(
            dict(review.model_dump(), created_at=row["created_at"], updated_at=row["updated_at"])
        )
    return result


def import_reviews(cfg: FolderConfig, source: Path, *, replace: bool = False) -> dict:
    """Validate the entire batch and merged Moment decisions before a single write."""
    data = json.loads(source.read_text(encoding="utf-8"))
    batch = ReviewInput.model_validate({"reviews": data} if isinstance(data, list) else data)
    if batch.library is not None and Path(batch.library).expanduser().resolve() != cfg.folder:
        raise ValueError("review file belongs to another library")
    photos = load_photos(cfg)
    updated_ids = {r.photo_id for r in batch.reviews}
    existing = (
        {}
        if replace
        else {r["photo_id"]: r for r in load_reviews(cfg, photos, replaced_ids=updated_ids)}
    )
    now, seen = datetime.now(timezone.utc).isoformat(), set()
    for review in batch.reviews:
        if review.photo_id in seen:
            raise ValueError(f"duplicate photo_id in import: {review.photo_id}")
        seen.add(review.photo_id)
        _check_identity(review, photos)
        prior = existing.get(review.photo_id)
        existing[review.photo_id] = dict(
            review.model_dump(),
            sha256=photos[review.photo_id]["sha256"],
            created_at=prior["created_at"] if prior else now,
            updated_at=now,
        )
    merged = [existing[pid] for pid in sorted(existing)]
    validate_selection(merged, photos)
    path = write_json(
        cfg,
        "reviews.json",
        {
            "schema_version": SCHEMA_VERSION,
            "library": str(cfg.folder),
            "reviews": merged,
        },
    )
    return {"imported": len(seen), "total_reviewed": len(merged), "path": str(path)}


def review_digest(reviews: list[dict], photos: dict[int, dict]) -> str:
    identities = [photos[r["photo_id"]] for r in reviews]
    data = json.dumps([reviews, identities], sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def effective_reviews(cfg: FolderConfig, photos: dict[int, dict]) -> tuple[list[dict], list[str]]:
    reviews = load_reviews(cfg, photos)
    warnings = []
    path = creative_path(cfg, "shortlist.json")
    if path.exists():
        shortlist = json.loads(path.read_text(encoding="utf-8"))
        if shortlist.get("source_digest") == review_digest(reviews, photos):
            selected = {r["photo_id"]: r for r in shortlist["selection"]}
            reviews = [
                dict(
                    r,
                    decision=selected[r["photo_id"]]["decision"],
                    review_decision=r["decision"],
                    selection_note=selected[r["photo_id"]]["selection_note"],
                )
                for r in reviews
            ]
        else:
            warnings.append(
                "Shortlist is stale; showing review nominations. Run creative shortlist again."
            )
    validate_selection(reviews, photos)
    return reviews, warnings


def report(cfg: FolderConfig) -> dict:
    photos = load_photos(cfg)
    reviews, warnings = effective_reviews(cfg, photos)
    results = []
    for row in reviews:
        p = photos[row["photo_id"]]
        results.append(
            dict(
                row,
                path=p["path"],
                preview_path=p["preview_path"],
                selects_score=p["selects_score"],
                moment_id=p["moment_id"],
                story_id=p["story_id"],
                taken_at=p["taken_at"],
            )
        )
    order = {d: i for i, d in enumerate(DECISIONS)}
    results.sort(key=lambda r: (order[r["decision"]], -r["creative_score"], r["photo_id"]))
    summary = {d: sum(r["decision"] == d for r in results) for d in DECISIONS}
    summary.update(
        indexed=len(photos), reviewed=len(results), unreviewed=len(photos) - len(results)
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "library": str(cfg.folder),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "warnings": warnings,
        "reviews": results,
    }
    paths = [write_json(cfg, "creative_results.json", document)]
    fields = list(CreativeReview.model_fields) + [
        "selects_score",
        "path",
        "preview_path",
        "moment_id",
        "story_id",
        "taken_at",
        "created_at",
        "updated_at",
        "review_decision",
        "selection_note",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fields, extrasaction="ignore")
    writer.writeheader()
    for row in results:
        # Text cells must not become spreadsheet formulas when opened in Excel.
        safe = {
            k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
            for k, v in row.items()
        }
        writer.writerow(safe)
    paths.append(write_text(cfg, "creative_results.csv", buffer.getvalue()))
    md = [
        "# Trip Selection",
        "",
        f"Reviewed: {len(results)} / {len(photos)}",
        "",
        "Scores describe visual editing potential in previews, not measured RAW dynamic range.",
        "",
    ]
    for warning in warnings:
        md.extend([warning, ""])
    for decision in DECISIONS:
        md.extend([f"## {decision.title()}", ""])
        for row in results:
            if row["decision"] != decision:
                continue
            md.extend(
                [
                    f"### {row['filename']} (photo {row['photo_id']})",
                    "",
                    f"Score: {row['creative_score']:.2f}",
                    "",
                    f"Path: {row['path']}",
                    "",
                    "Why:",
                    row["reason"],
                    "",
                    "Grade:",
                    row["grade_direction"] or "No grading recommended.",
                    "",
                ]
            )
            if row.get("selection_note"):
                md.extend([f"Selection: {row['selection_note']}", ""])
            md.extend(["---", ""])
    paths.append(write_text(cfg, "selection.md", "\n".join(md)))
    return {"summary": summary, "warnings": warnings, "paths": [str(p) for p in paths]}
