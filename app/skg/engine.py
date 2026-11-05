"""
Spatial Knowledge Graph engine.

Ground truth store + deterministic math over a blueprint graph.
NetworkX handles topology (rooms, doors, adjacency); Shapely handles geometry
(polygons, areas). No LLM calls happen in this module — every function here
is pure, testable graph/geometry logic.
"""

import networkx as nx


class SpatialKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.room_geometry: dict[str, object] = {}  # room_id -> shapely Polygon
        self.exits: set[str] = set()  # rooms explicitly marked as exits

    # ---- construction -------------------------------------------------
    def add_room(self, room_id: str, polygon, label: str | None = None):
        self.graph.add_node(room_id, label=label or room_id)
        self.room_geometry[room_id] = polygon

    def add_door(self, room_a: str, room_b: str, width_m: float):
        self.graph.add_edge(room_a, room_b, width_m=width_m)

    def mark_exit(self, room_id: str):
        if room_id not in self.graph:
            raise ValueError(f"Unknown room: {room_id}")
        self.exits.add(room_id)

    # ---- lookup -------------------------------------------------------
    def resolve_room(self, name: str) -> str:
        """Map a user-typed name ('the kitchen') to a room id.
        Matches id or label, case-insensitive, then falls back to a unique
        substring match. Raises ValueError if missing or ambiguous."""
        needle = name.strip().lower()
        if needle.startswith("the "):
            needle = needle[4:]

        for node, data in self.graph.nodes(data=True):
            if needle in (node.lower(), str(data.get("label", "")).lower()):
                return node

        matches = [
            node
            for node, data in self.graph.nodes(data=True)
            if needle in node.lower() or needle in str(data.get("label", "")).lower()
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            known = ", ".join(sorted(self.graph.nodes)) or "none"
            raise ValueError(f"Unknown room '{name}'. Known rooms: {known}")
        raise ValueError(f"Ambiguous room '{name}': could be {', '.join(sorted(matches))}")

    # ---- queries ------------------------------------------------------
    def compute_area(self, room: str) -> float:
        return float(self.room_geometry[self.resolve_room(room)].area)

    def find_path(self, room_from: str, room_to: str) -> list[str]:
        a, b = self.resolve_room(room_from), self.resolve_room(room_to)
        try:
            return nx.shortest_path(self.graph, a, b)
        except nx.NetworkXNoPath:
            raise ValueError(f"No connection between '{a}' and '{b}'")

    def check_egress(self, room: str) -> dict:
        """Shortest path to the nearest exit. Uses explicitly marked exits if
        any exist; otherwise falls back to leaf rooms (degree <= 1) and flags
        the result as provisional."""
        start = self.resolve_room(room)
        provisional = not self.exits
        candidates = self.exits or {n for n in self.graph if self.graph.degree[n] <= 1}
        candidates = candidates - {start} if len(candidates) > 1 else candidates

        best: list[str] | None = None
        for target in candidates:
            try:
                path = nx.shortest_path(self.graph, start, target)
            except nx.NetworkXNoPath:
                continue
            if best is None or len(path) < len(best):
                best = path
        if best is None:
            raise ValueError(f"No egress path found from '{start}'")
        return {"path": best, "hops": len(best) - 1, "provisional_exits": provisional}


def execute(skg: SpatialKnowledgeGraph, query) -> dict:
    """Dispatch a StructuredQuery to the matching SKG method.
    Result shapes match API.md; this is where every number comes from —
    never from the translator."""
    if query.intent == "find_path":
        path = skg.find_path(query.room_from, query.room_to)
        return {"path": path, "hops": len(path) - 1}
    if query.intent == "compute_area":
        return {"room": skg.resolve_room(query.room), "area_m2": skg.compute_area(query.room)}
    if query.intent == "check_egress":
        return skg.check_egress(query.room)
    raise ValueError(f"Unhandled intent: {query.intent}")
