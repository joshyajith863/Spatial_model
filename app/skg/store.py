"""
In-memory store mapping blueprint_id -> SpatialKnowledgeGraph.

Shared between the DXF/PDF mesh path (app/api/mesh.py) and the raster CV
path (app/api/blueprints.py), so either input route can populate a graph
that the query endpoint later reads from, by the same blueprint_id.

Fine for a local-network demo (single process, no persistence needed across
restarts). Swap for sqlite/pickle-to-disk if you need it to survive restarts.
"""

from app.skg.engine import SpatialKnowledgeGraph

_GRAPHS: dict[str, SpatialKnowledgeGraph] = {}


def save(blueprint_id: str, graph: SpatialKnowledgeGraph) -> None:
    _GRAPHS[blueprint_id] = graph


def get(blueprint_id: str) -> SpatialKnowledgeGraph | None:
    return _GRAPHS.get(blueprint_id)


def exists(blueprint_id: str) -> bool:
    return blueprint_id in _GRAPHS
