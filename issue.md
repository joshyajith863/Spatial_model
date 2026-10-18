# Issue: DXF coordinate extraction stopped working on `experiment` branch

**Labels:** bug
**Branches affected:** `experiment`, `main`
**Branches not affected:** `feature/normalized-floorplan`

---

## Summary

The model stopped extracting coordinates from uploaded `.dxf` files. The upload
endpoint (`POST /api/blueprints/upload`) no longer parses walls/rooms/bounds —
it just saves the file and returns `status: "processing"`.

## Root cause

The DXF parser (`app/parsers/dxf_parser.py`) and the `NormalizedFloorplan`
model (`app/models.py`) were committed **only** on the
`feature/normalized-floorplan` branch (commit `9c8c744`) and were **never
merged** into `main` or `experiment`.

The current branch (`experiment`) was built from `main`, which never received
that code. When switching branches, git removed the `.py` sources — but a
leftover compiled artifact (`app/parsers/__pycache__/dxf_parser.cpython-312.pyc`,
dated Oct 12) remained, making it look like the code was still present.

## Evidence

- `app/parsers/` on `experiment` contained only `__pycache__/` (no `.py` files)
  before the fix.
- `app/api/blueprints.py` had a `TODO` where the `parse_dxf` call should be.
- `STATUS.md` documented this as a known gap:
  > "DXF path lives on another branch. `app/parsers/dxf_parser.py` exists only
  > on `feature/normalized-floorplan`, not merged into `experiment`/`main`."
- An untracked `normalized-floorplan.patch` in the repo root indicated a prior
  (incomplete) attempt to bring the work across branches via a patch.

## Impact

- No coordinate extraction from DXF uploads (walls, rooms, bounds, units).
- `GET /api/blueprints/{id}/floorplan` returned 404 (no floorplan parsed).
- Blocked the downstream pipeline: SKG graph population, query engine, and
  HUD overlay all depend on parsed coordinates.

## Resolution (fixed on `experiment`)

Merged `feature/normalized-floorplan` into `experiment` (merge commit `9484f5d`).

- One trivial conflict in `.gitignore` (added `weights/` ignore section on
  `experiment` vs. nothing on the feature branch) — resolved by keeping the
  `weights/` section.
- `requirements.txt` auto-merged cleanly; both branches' dependencies kept
  (including `ezdxf`).
- Verified `ezdxf 1.4.4` is installed in the venv.

### Verification

Ran `parse_dxf` directly on a real uploaded file
(`data/uploads/4d98f90d-..._floorplan.dxf`):

- 257 walls extracted with real X/Y coordinates
- Bounds: `(-0.65, -0.22, 0.15, 0.35)` in meters

## Follow-up / remaining gap

Rooms = 0 on the test file. This is not a parser bug — the file puts walls on
`xref-...$0$A-WALL` (matched by the parser) but has **no room-boundary layer**;
the parser's configured `room_layers` do not exist in this DXF. Options:

1. Add a room-layer hint / config to the parser.
2. Derive rooms by polygonizing the extracted wall network.

This should be tracked as a separate issue once prioritized.

## Prevention

- Merge feature branches into `main`/`experiment` promptly, or keep a
  long-lived branch checked out only when actively working on it.
- Remove stale `__pycache__/` after branch switches if it causes confusion
  (already covered by `.gitignore` — do not commit `.pyc` files).
