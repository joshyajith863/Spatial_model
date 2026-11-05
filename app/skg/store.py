"""In-memory blueprint_id -> SpatialKnowledgeGraph store (single-process demo).

Not persistent and not multi-worker safe; swap for sqlite/pickle if needed.
Graphs are created on `POST /api/blueprints/{id}/confirm` and read by
`POST /api/query/`.
"""

from app.skg.engine import SpatialKnowledgeGraph

_GRAPHS: dict[str, SpatialKnowledgeGraph] = {}


def save(blueprint_id: str, graph: SpatialKnowledgeGraph) -> None:
    _GRAPHS[blueprint_id] = graph


def get(blueprint_id: str) -> SpatialKnowledgeGraph | None:
    return _GRAPHS.get(blueprint_id)
