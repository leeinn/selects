"""Stable CLI boundary for multimodal review agents."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click

from selects.config import FolderConfig
from selects.export import preview_xmp_writes, rating_sidecar_path, write_xmp_ratings

from .candidates import check_state, generate_candidates, load_photos
from .reviews import effective_reviews, import_reviews, report, write_json
from .schema import CreativeReview
from .selection import shortlist

LIBRARY = click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


def _echo(data):
    click.echo(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


@click.group()
def creative():
    """Photography editor: preview candidates, review imports, reports and XMP."""


@creative.command("schema")
def schema_command():
    """Print the JSON schema for a single creative review."""
    _echo(CreativeReview.model_json_schema())


@creative.command("status")
@LIBRARY
def status_command(folder):
    """Read indexing/review coverage; never start ML or decode originals."""
    cfg = FolderConfig(folder=folder)
    _run(check_state, cfg)
    if not cfg.db_path.is_file():
        _echo(
            {
                "library": str(cfg.folder),
                "indexed": False,
                "next_passes": ["index", "classical", "embed", "aesthetic", "moment", "story"],
            }
        )
        return
    photos = _run(load_photos, cfg)
    rows, warnings = _run(effective_reviews, cfg, photos)
    _echo(
        {
            "library": str(cfg.folder),
            "indexed": True,
            "photos": len(photos),
            "classical_pending": sum(not p["classical_done"] for p in photos.values()),
            "embedding_pending": sum(not p["embedding_done"] for p in photos.values()),
            "aesthetic_missing": sum(p["aesthetic"] is None for p in photos.values()),
            "hyperiqa_missing": sum(p["hyperiqa"] is None for p in photos.values()),
            "preview_missing": sum(p["preview_path"] is None for p in photos.values()),
            "photos_in_moments": sum(p["moment_id"] is not None for p in photos.values()),
            "photos_in_stories": sum(bool(p["story_ids"]) for p in photos.values()),
            "reviewed": len(rows),
            "warnings": warnings,
        }
    )


@creative.command("candidates")
@LIBRARY
@click.option("--max-candidates", type=click.IntRange(min=1))
@click.option(
    "--percentile",
    type=click.FloatRange(0, 100),
    default=50.0,
    show_default=True,
    help="Aesthetic percentile floor; 50 keeps the top half, plus protected photos.",
)
@click.option("--include-story/--no-include-story", default=True, show_default=True)
@click.option(
    "--include-maybe", is_flag=True, help="Include all technically eligible lower-ranked photos."
)
@click.option("--batch-size", type=click.IntRange(30, 80), default=50, show_default=True)
@click.option(
    "--batch", type=click.IntRange(min=1), help="Print just one batch from the generated manifest."
)
@click.option(
    "--moment",
    "moment_id",
    type=click.IntRange(min=1),
    help="Compare all eligible members of one Moment.",
)
@click.option("--story", "story_id", type=click.IntRange(min=1))
def candidates_command(folder, batch, **options):
    """Print JSON and save a manifest under <library>/.selects/creative/."""
    cfg = FolderConfig(folder=folder)
    manifest = _run(generate_candidates, cfg, **options)
    if batch is not None:
        if batch > len(manifest["batches"]):
            raise click.ClickException(
                f"batch {batch} does not exist ({len(manifest['batches'])} batches)"
            )
        selected_batch = manifest["batches"][batch - 1]
    suffix = ""
    if options["moment_id"] is not None:
        suffix += f"_moment_{options['moment_id']}"
    if options["story_id"] is not None:
        suffix += f"_story_{options['story_id']}"
    path = _run(write_json, cfg, f"candidates{suffix}.json", manifest)
    click.echo(f"Manifest: {path}", err=True)
    if batch is not None:
        ids = set(selected_batch["photo_ids"])
        manifest = dict(
            manifest,
            candidates=[p for p in manifest["candidates"] if p["photo_id"] in ids],
            batches=[selected_batch],
        )
    _echo(manifest)


@creative.command("import")
@LIBRARY
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--replace", is_flag=True, help="Replace all saved reviews with this complete set.")
def import_command(folder, results, replace):
    """Atomically merge a validated review array or {library, reviews} document."""
    _echo(_run(import_reviews, FolderConfig(folder=folder), results, replace=replace))


@creative.command("report")
@LIBRARY
def report_command(folder):
    """Write creative_results.json/.csv and selection.md; unreviewed stays unreviewed."""
    _echo(_run(report, FolderConfig(folder=folder)))


@creative.command("shortlist")
@LIBRARY
@click.option("--heroes", type=click.IntRange(min=0), default=12, show_default=True)
@click.option(
    "--grades",
    type=click.IntRange(min=0),
    default=40,
    show_default=True,
    help="Grading cap including heroes.",
)
@click.option(
    "--story",
    type=click.IntRange(min=0),
    default=80,
    show_default=True,
    help="Final story cap including heroes and grades.",
)
def shortlist_command(folder, **caps):
    """Choose a diverse set within nested caps, using reviewed nominations."""
    cfg = FolderConfig(folder=folder)
    result = _run(shortlist, cfg, **caps)
    result["report"] = _run(report, cfg)
    _echo(result)


@creative.command("xmp")
@LIBRARY
@click.option(
    "--apply/--dry-run",
    default=False,
    show_default=True,
    help="Default is dry-run. Apply writes only .xmp sidecars.",
)
@click.option("--force", is_flag=True, help="Allow replacing a higher existing sidecar rating.")
def xmp_command(folder, apply, force):
    """Map hero/grade/story/maybe/reject to 5/4/3/2/1 stars in sidecars."""
    cfg = FolderConfig(folder=folder)
    photos = _run(load_photos, cfg)
    reviews, warnings = _run(effective_reviews, cfg, photos)
    if apply and warnings:
        raise click.ClickException(
            "Saved shortlist is stale; run creative shortlist before applying ratings."
        )
    triples = [(r["photo_id"], Path(photos[r["photo_id"]]["path"]), r["decision"]) for r in reviews]
    targets = {}
    for pid, path, _ in triples:
        target = rating_sidecar_path(path).resolve()
        if target in targets:
            raise click.ClickException(
                f"sidecar collision between photos {targets[target]} and {pid}: {target}"
            )
        targets[target] = pid
    fn = write_xmp_ratings if apply else preview_xmp_writes
    plans = _run(fn, triples, force=force, library_root=cfg.folder, sidecar_only=True)
    _echo({"dry_run": not apply, "warnings": warnings, "plans": [asdict(p) for p in plans]})
    if apply and any(p.action == "no_op" for p in plans):
        raise click.ClickException(
            "Some sidecars could not be written; inspect the returned plans."
        )
