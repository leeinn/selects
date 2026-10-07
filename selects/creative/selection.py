"""Budgeted, diversity-aware shortlist of agent nominations; no new vision model."""

from __future__ import annotations

from collections import Counter

from selects.config import FolderConfig

from .candidates import load_photos
from .reviews import load_reviews, review_digest, validate_selection, write_json
from .schema import SCHEMA_VERSION


def _dimensions(row: dict, p: dict) -> list[tuple[str, str]]:
    dims = []
    for key in ("scene_type", "location", "subject", "composition_key"):
        value = row.get(key) or (p.get("category") if key == "scene_type" else None)
        if value:
            dims.append((key, str(value).casefold()))
    for sid in p["story_ids"]:
        dims.append(("story", str(sid)))
    if not p["story_ids"] and p["taken_at"]:
        dims.append(("day", p["taken_at"][:10]))
    if not row.get("location") and p["gps_lat"] is not None and p["gps_lon"] is not None:
        dims.append(("location", f"{p['gps_lat']:.2f},{p['gps_lon']:.2f}"))
    for person_id in p["person_ids"]:
        dims.append(("person", str(person_id)))
    return dims


def shortlist(cfg: FolderConfig, *, heroes: int = 12, grades: int = 40, story: int = 80) -> dict:
    """Nested caps: grades includes heroes; story includes both. Never fill with maybes."""
    if not 0 <= heroes <= grades <= story:
        raise ValueError("caps must satisfy 0 <= heroes <= grades <= story")
    photos = load_photos(cfg)
    reviews = load_reviews(cfg, photos)
    validate_selection(reviews, photos)
    dimensions = {r["photo_id"]: _dimensions(r, photos[r["photo_id"]]) for r in reviews}
    counts = Counter()
    selected = {}
    by_id = {r["photo_id"]: r for r in reviews}

    def choose(cap: int, nominations: set[str], decision: str):
        pool = [
            r for r in reviews if r["decision"] in nominations and r["photo_id"] not in selected
        ]
        while pool and len(selected) < cap:

            def utility(r):
                dims = dimensions[r["photo_id"]]
                # A small coverage bonus/repetition penalty preserves quality while
                # giving the trip rhythm across subjects, places and compositions.
                coverage = sum(0.6 for d in dims if not counts[d])
                repetition = sum(0.5 * min(counts[d], 4) for d in dims)
                narrative = 0.12 * r["story_value"] if decision == "story" else 0
                return r["creative_score"] + coverage - repetition + narrative, -r["photo_id"]

            best = max(pool, key=utility)
            pool.remove(best)
            pid = best["photo_id"]
            selected[pid] = decision
            counts.update(dimensions[pid])

    choose(heroes, {"hero"}, "hero")
    choose(grades, {"hero", "grade"}, "grade")
    choose(story, {"hero", "grade", "story"}, "story")
    selection = []
    for pid in sorted(by_id):
        original = by_id[pid]["decision"]
        final = selected.get(pid, "maybe" if original in {"hero", "grade", "story"} else original)
        selection.append(
            {
                "photo_id": pid,
                "decision": final,
                "selection_note": "Quality, diversity and story budget"
                if final != original
                else "",
            }
        )
    path = write_json(
        cfg,
        "shortlist.json",
        {
            "schema_version": SCHEMA_VERSION,
            "library": str(cfg.folder),
            "source_digest": review_digest(reviews, photos),
            "caps": {"heroes": heroes, "grades_including_heroes": grades, "story_total": story},
            "selection": selection,
        },
    )
    return {
        "path": str(path),
        "heroes": sum(d == "hero" for d in selected.values()),
        "grading_total": sum(d in {"hero", "grade"} for d in selected.values()),
        "story_total": len(selected),
    }
