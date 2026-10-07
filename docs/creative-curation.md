# Creative curation

Selects has two stages for a photography workflow: existing indexing and technical/ML curation narrow the library safely, then a photography editor reviews cached previews for serious grading and storytelling. Visual grading potential is independent of how attractive the current preview looks. No new local vision model is used.

## Architecture and reused capabilities

The implementation was checked against source and tests, rather than the README alone:

| Existing code/data | Reuse |
|---|---|
| `decode/raw.py`, `indexer/preview.py` | Embedded RAW preview with existing demosaic fallback during indexing; cached 256px thumbnails and 1024px previews |
| `ClassicalScore`, `pipeline.py` | Severe technical rejects, sharpness, exposure, face/eye and cached luminance signals |
| `Embedding`, `AestheticScore`, `ml/aesthetic.py` | HyperIQA on historical `ap25_score` / 10, else SigLIP prompt IQA; existing ensemble/ranking logic |
| `ml/curation.py`, `ml/taste.py` | Existing taste blend can reorder candidates; original machine scores remain separate from creative scores |
| `Moment`, `MomentMember`, `ml/moments.py` | Burst membership, primary member, ranks; no duplicate/burst model is reimplemented |
| `Story`, `StoryItem`, `PhotoCategory`, `PhotoPerson` | Day/Story/scene coverage and metadata for diversity and batch context |
| `Photo.sha256` | Existing exact-duplicate identity; no new file hashing during creative review |
| `export.py` | Existing rating planning, higher-rating protection, metadata preservation, plus opt-in sidecar-only behavior |
| CLI/server routes | Existing index stages and API/UI behavior stay available; the new workflow is exposed through CLI |

The historical `curate()` top-quartile per-scope behavior remains intact for existing callers. The creative workflow has its own default percentile floor of 50, plus retention rules, to avoid prematurely excluding promising RAWs.

No database migration is needed. Existing ORM/Alembic tables supply context; there is no `CreativeReview` table. Authoritative creative reviews are versioned JSON stored at `<library>/.selects/creative/reviews.json`. Writes replace state files atomically. Sequential batch import is the supported write workflow; no agent edits SQLite.

New implementation is isolated in `selects/creative/{schema,candidates,reviews,selection,cli}.py`, with registration in `selects/cli.py`, optional export support in `selects/export.py`, and contract/safety/scale tests in `tests/test_creative.py`. Indexing also repairs absent still-image preview/thumbnail caches when originals have unchanged hashes, using its existing decoder and resetting processing flags.

## CLI workflow

```bash
selects creative status /path/to/trip
selects index /path/to/trip --pass index
selects index /path/to/trip --pass classical
selects index /path/to/trip --pass embed
selects index /path/to/trip --pass aesthetic
selects index /path/to/trip --pass moment
selects index /path/to/trip --pass story
selects creative candidates /path/to/trip --percentile 50 --include-story --batch-size 50
selects creative candidates /path/to/trip --moment 12
selects creative import /path/to/trip /path/to/trip/.selects/creative/batch_0001.json
selects creative shortlist /path/to/trip --heroes 12 --grades 40 --story 80
selects creative report /path/to/trip
selects creative xmp /path/to/trip --dry-run
```

Run only the needed indexing passes; `selects index <library>` runs the existing full pipeline. `creative status` reports coverage without running ML. Missing model stages are allowed: unscored photographs remain candidates. Missing/unsafe previews are explicit gaps with null `preview_path`; the creative commands never decode the original as a fallback.

All command stdout is JSON. Candidate-manifest location messages go to stderr. Use `python -m selects` as an equivalent entry point from a development checkout.

## Candidate manifest

`creative candidates` prints and saves an envelope:

```json
{
  "schema_version": 1,
  "library": "/path/to/trip",
  "options": {"percentile": 50, "batch_size": 50},
  "summary": {"indexed": 3000, "eligible": 2400, "pool_size": 1500, "candidates": 1500},
  "candidates": [
    {
      "photo_id": 123,
      "filename": "DSC_4821.ARW",
      "sha256": "actual source hash",
      "path": "/path/to/trip/DSC_4821.ARW",
      "preview_path": "/path/to/trip/.selects/previews/hash.jpg",
      "moment_id": 12,
      "moment_rank": 0,
      "story_id": 5,
      "aesthetic": 0.84,
      "selects_score": 0.84,
      "sharpness": 910.0,
      "candidate_reasons": ["aesthetic_percentile", "moment_primary"]
    }
  ],
  "batches": [{"batch_id": 1, "photo_ids": [123], "split_moment": false}]
}
```

Additional fields include original format, capture time/GPS, HyperIQA/IQA/taste, classical diagnostics, user decision, all Story/scene memberships, persons, exact-duplicate IDs, and processing flags. Examples show abbreviated structures; counts and paths come from the actual library.

Candidate rules:

- Manual keep/silver always enters, including a machine auto-reject override. Ordinary auto-rejects and manual rejects are excluded.
- Keep at least the eligible Moment primary; if it is rejected, choose the best eligible sibling.
- Keep the top half of scored eligible photographs by default. Missing aesthetic is a non-gate.
- Protect the top decile of aesthetic and positive sharpness scores and non-rejected extreme-light frames (`luma_mean < .32` or `> .78`). These are review opportunities, not RAW-recovery measurements.
- With Story coverage enabled, protect a representative from each available Story and existing Story scene, including lower-aesthetic narrative coverage. Unique scene opportunities come from existing scene metadata, not a new uniqueness model.
- Collapse discretionary exact-hash duplicates. Protected/manual entries remain visible for editorial comparison, with their duplicate IDs; final nominations cannot include identical copies twice.

`--percentile` means a floor, so 50 retains roughly the highest 50%, 60 the highest 40%, before additions. Ties and retention rules can make the actual pool larger. There is no promise of a fixed candidate count.

`--max-candidates` caps the pool while preserving protected photographs. An impossible cap fails with the minimum protected count. Remaining slots are distributed across Stories/days by quality. Capped omissions remain unreviewed. `--include-maybe` broadens the pool to every technically eligible photograph, independent of earlier creative decisions. `--no-include-story` disables Story/scene retention.

`--batch-size` accepts 30–80, default 50. Normal Moments are kept together. Larger bursts split into adjacent batches marked `split_moment`. Small final batches are allowed. `--batch N` prints one batch while keeping the complete manifest on disk. Prefer reading the saved manifest for a long review so reindexing cannot change batch boundaries between calls. `--moment ID` includes all eligible Moment alternatives; `--story ID` scopes by Story membership. Scoped manifests use separate filenames.

## Reviews and import

`selects creative schema` emits the Pydantic JSON schema. Import an array of review objects or `{ "schema_version": 1, "library": "...", "reviews": [...] }`.

Required fields are `photo_id`, `filename`, `technical_score`, `composition`, `light`, `moment`, `story_value`, `uniqueness`, `grading_potential`, `decision`, `reason`, and `grade_direction`. All scores use 0–10; NaN, infinity, booleans, and numeric strings are rejected. Decisions are exactly hero/grade/story/maybe/reject. Reasons must be nonempty. Selected photos need a visual grading direction. Omit `creative_score` to calculate it automatically:

```text
composition 22% + light 18% + moment 15% + grading_potential 20%
    + story_value 10% + uniqueness 15%
```

Technical score is independent. An observed `technical_failure: true` gates out hero/grade/story; a low score or high ISO alone does not. A supplied creative score must agree with the weighted score rounded to two decimals. Creative scores never update aesthetic, embeddings, swipes, taste training labels, or existing database ratings.

Optional fields include source `sha256` (recommended), `reviewer`, scene type, location, subject, composition key, and justified Moment exceptions. Import checks IDs, filename/hash identity, duplicate IDs, the whole merged Moment selection, and exact duplicate nominations before writing. One nominee per Moment is the default. Two require an explicit exception and justification; more than two fail.

Imports merge by ID; updates preserve creation time and record a new update time. Re-reviewing a replaced source can update that ID with its new hash. Unknown or changed saved sources cause reports to fail rather than attaching old judgments to a different photograph. `creative import <library> complete.json --replace` explicitly replaces the complete review set, useful after pruning old sources. No original is changed by import or creative reject. Reports are derived output and are not import files; edit the batch review format or the authoritative review fields through CLI import.

## Final selection and reports

Do a cross-batch visual editing pass before finalizing. `shortlist` is a deterministic aid based on reviewed nominations and metadata, not a substitute for image judgment. It combines creative quality with coverage/repetition signals for Story/day, scene type/category, location/GPS, subject/person, and composition. Supply consistent optional labels to improve its ability to avoid repeated framing. It does not run another embedding or duplicate model.

Caps are nested: `--grades 40` includes heroes; `--story 80` includes heroes and grades. Must satisfy `0 <= heroes <= grades <= story`. Shortlist never promotes maybe/reject or fills an undersupplied tier with weak nominations. It saves effective decisions separately from original reviews. New imports or indexed metadata invalidate an old shortlist; reports warn and show review nominations until shortlist is rerun.

`report` writes these files under `.selects/creative/`:

```text
reviews.json              authoritative, versioned merged reviews
candidates.json           broad candidate snapshot with batches
shortlist.json            optional budgeted selection and provenance
creative_results.json     enriched reviews, machine scores, coverage and warnings
creative_results.csv      scores/decisions/reasons/paths for human filtering
selection.md              Hero / Grade / Story / Maybe / Reject, reasons and grading direction
```

Unreviewed photos stay unreviewed. CSV text is quoted for correct newlines/commas and spreadsheet-formula safety. Scores express visual grading potential from previews; they do not claim sensor or RAW latitude measurements.

## XMP and original-file safety

`creative xmp` defaults to dry-run; `--dry-run` is also explicit. `--apply` writes `stem.xmp` for RAW and `stem.ext.xmp` for other still formats using the existing rating engine: hero 5, grade 4, story 3, maybe 2, reject 1. This keeps ratings for same-stem RAW+JPEG pairs separate. It does not modify RAW, JPEG, HEIC, or other original image bytes. For JPEG/HEIC the new command uses sidecars as well; the existing export API/UI keeps its established embedded-rating behavior.

Existing higher sidecar ratings are preserved unless `--force` is specified. Updating an existing readable sidecar preserves its other metadata. Unsupported paths, missing sources, linked sidecars, and targets outside the library are refused. If two reviewed RAW originals share a sidecar target in one directory, the entire creative XMP command fails before writing to avoid collisions. A stale shortlist also prevents apply until refreshed. `--apply` reports individual write failures and returns a nonzero status.

All creative state/report writes are under the library's `.selects/`; sidecars are the explicit export exception beside photos. State directories and output/preview symlinks cannot redirect the new workflow into originals. New code never deletes, renames, moves, or overwrites original images.

The orchestration Skill is [travel-photo-curator](../skills/travel-photo-curator/SKILL.md). Load it from this repository when asking Codex to review a library. It performs preview-based editorial judgment and drives CLI imports/reports/XMP planning without implementing ML.
