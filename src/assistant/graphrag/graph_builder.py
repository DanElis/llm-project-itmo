"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}

    def add_node(entity, chunk_id: str) -> None:
        node = nodes.setdefault(entity.node_id, {"type": entity.type, "name": entity.name, "sources": set()})
        node["sources"].add(chunk_id)

    for result in results:
        for entity in result.entities:
            add_node(entity, result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            for end in (relation.subject, relation.object):
                add_node(end, result.chunk_id)
            key = (relation.subject.node_id, relation.object.node_id)
            # Между двумя узлами схема допускает только один тип связи, поэтому пара узлов – ключ ребра.
            edge = edges.setdefault(key, {"type": relation.type, "sources": set()})
            edge["sources"].add(result.chunk_id)

    conflicts = 0
    for (source, target), edge in list(edges.items()):
        back = edges.get((target, source))
        if edge["type"] != "REPLACED_BY" or back is None or back["type"] != "REPLACED_BY" or source > target:
            continue
        conflicts += 1
        forward_weight, back_weight = len(edge["sources"]), len(back["sources"])
        if forward_weight <= back_weight:
            del edges[(source, target)]
        if back_weight <= forward_weight:
            del edges[(target, source)]

    graph = nx.DiGraph(conflicts=conflicts)
    for node_id, attrs in nodes.items():
        graph.add_node(node_id, type=attrs["type"], name=attrs["name"], sources=sorted(attrs["sources"]))
    for (source, target), edge in edges.items():
        sources = sorted(edge["sources"])
        graph.add_edge(source, target, type=edge["type"], weight=len(sources), sources=sources)
    return graph
