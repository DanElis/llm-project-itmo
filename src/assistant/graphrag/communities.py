"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

SEED = 42
MAX_WORDS = 150
MAX_FACTS = 60

SYSTEM = (
    "Ты пишешь краткую сводку фрагмента графа знаний сервисной службы холодильного оборудования. "
    f"Пиши по-русски, не длиннее {MAX_WORDS} слов, одним абзацем. Используй только переданные факты и числа, "
    "ничего не добавляй и не пересчитывай. Коды моделей, деталей, ошибок и регламентов пиши точно как в данных. "
    "Обязательно назови: какие линейки и модели входят, какие узлы отказывают чаще всего и сколько раз, "
    "какие детали сняты с производства и чем заменены."
)


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    if graph.number_of_nodes() == 0:
        return []
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))
    for source, target, attrs in sorted(graph.edges(data=True)):
        weight = attrs.get("weight", 1)
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)
    parts = nx.community.louvain_communities(undirected, weight="weight", seed=SEED)
    # Порядок сообществ: от больших к меньшим, при равенстве – по первому узлу, чтобы номера не плавали.
    ordered = sorted((sorted(part) for part in parts), key=lambda nodes: (-len(nodes), nodes[0]))
    return [Community(number, nodes) for number, nodes in enumerate(ordered)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    reply = llm.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": community_digest(community, graph)}],
        max_tokens=400,
    )
    words = reply.content.split()
    return " ".join(words[:MAX_WORDS])


def community_digest(community: Community, graph: nx.DiGraph) -> str:
    """Готовые факты и подсчёты сообщества для запроса к модели."""
    members = set(community.nodes)
    edges = [(s, t, attrs) for s, t, attrs in graph.edges(data=True) if s in members or t in members]
    line_of = {s: _name(graph, t) for s, t, attrs in graph.edges(data=True) if attrs.get("type") == "BELONGS_TO"}
    by_type: dict[str, list[str]] = {}
    for node in sorted(members):
        by_type.setdefault(graph.nodes[node].get("type", node.split(":")[0]), []).append(_name(graph, node))
    lines = []
    lines.append("Сущности сообщества:")
    lines += [f"- {kind}: {', '.join(names)}" for kind, names in sorted(by_type.items())]

    failures, failures_by_model = Counter(), Counter()
    for source, target, attrs in edges:
        if attrs.get("type") == "FAILED":
            failures[_name(graph, target)] += attrs.get("weight", 1)
            line = line_of.get(source)
            failures_by_model[f"{_name(graph, source)} ({line})" if line else _name(graph, source)] += attrs.get(
                "weight", 1
            )
    if failures:
        lines.append("Отказы узлов в сообществе (число отчётов о выезде):")
        lines += [f"- {name}: {count}" for name, count in failures.most_common()]
        lines.append("Отказы по моделям в сообществе (число отчётов):")
        lines += [f"- {name}: {count}" for name, count in failures_by_model.most_common()]

    replaced = sorted((s, t) for s, t, attrs in edges if attrs.get("type") == "REPLACED_BY")
    if replaced:
        lines.append("Детали, снятые с производства, и их замены:")
        for source, target in replaced:
            models = [_name(graph, model) for model in _models_with_part(graph, source)]
            where = ", ".join(
                f"{model} ({line_of[f'Model:{model}']})" if f"Model:{model}" in line_of else model for model in models
            )
            used = f" (стоит в моделях {where})" if where else ""
            lines.append(f"- {_name(graph, source)}{used} заменена на {_name(graph, target)}")

    other = sorted(
        (
            (attrs.get("weight", 1), _fact(graph, s, t, attrs))
            for s, t, attrs in edges
            if attrs.get("type") not in ("FAILED", "REPLACED_BY")
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if other:
        lines.append("Другие связи (в скобках – в скольких фрагментах документов найдена связь):")
        lines += [f"- {fact} ({weight})" for weight, fact in other[:MAX_FACTS]]
    return "\n".join(lines)


def _models_with_part(graph: nx.DiGraph, part: str) -> list[str]:
    return sorted(m for m in graph.predecessors(part) if graph.edges[m, part].get("type") == "HAS_PART")


def _name(graph: nx.DiGraph, node: str) -> str:
    return graph.nodes[node].get("name", node.split(":", 1)[-1])


def _fact(graph: nx.DiGraph, source: str, target: str, attrs: dict) -> str:
    return f"{_name(graph, source)} -{attrs.get('type')}-> {_name(graph, target)}"
