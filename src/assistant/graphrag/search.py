"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase

NOT_FOUND = "В базе знаний не найдено."
MAX_FACT_CHARS = 8000
PATH_CUTOFF = 3
NEIGHBOURHOOD_DEPTH = 2

# При обрезке до MAX_FACT_CHARS дальние факты этих типов важнее для ответов диспетчеру.
TYPE_PRIORITY = ["REPLACED_BY", "SUPPLIED_BY", "FIXED_BY", "INDICATES", "IS_A", "HAS_PART", "BELONGS_TO", "FAILED"]

LOCAL_SYSTEM = (
    "Ты ассистент диспетчера сервисной службы компании «ПолярТех». Отвечай на вопрос только по пронумерованным "
    "фактам из графа знаний. Факт «A -СВЯЗЬ-> B» читается так: HAS_PART – в модели A установлена деталь B; "
    "IS_A – деталь A является узлом B; INDICATES – ошибка A указывает на отказ узла B; FIXED_BY – ошибка A "
    "устраняется по регламенту B; REPLACED_BY – деталь A снята с производства и заменена деталью B; "
    "SUPPLIED_BY – деталь A поставляет B; BELONGS_TO – модель A входит в линейку B; FAILED – в модели A "
    "зафиксирован отказ узла B.\n"
    "Если нужная деталь заменена, называй последнюю деталь цепочки замен как актуальную и упомяни, что исходная "
    "снята с производства. Коды пиши точно как в фактах. Отвечай кратко, одним-двумя предложениями. "
    f"Если в фактах нет ответа, напиши: «{NOT_FOUND}»"
)

def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    anchors = find_anchors(question, kb)
    if anchors is None or not anchors:
        return Answer(NOT_FOUND, refused=True)
    if len(anchors) == 1:
        edges = _neighbourhood(graph, anchors[0])
    else:
        edges = _paths(graph, anchors)
    edges = _with_replacements(graph, edges)
    if not edges:
        return Answer(NOT_FOUND, refused=True)
    facts, used = _facts_within_budget(graph, edges)
    numbered = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, start=1))
    hint = _replacement_hint(graph, used)
    prompt = f"Факты:\n{numbered}\n\n{hint}Вопрос: {question}"
    reply = llm.chat([{"role": "system", "content": LOCAL_SYSTEM}, {"role": "user", "content": prompt}])
    text = reply.content.strip() or NOT_FOUND
    return Answer(text, sources=_documents(graph, used), facts=facts, refused=_is_refusal(text))


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    # TODO: частичные ответы по сводкам, объединение
    raise NotImplementedError("Реализуйте global_search")


def find_anchors(question: str, kb: KnowledgeBase) -> list[str] | None:
    """Узлы графа, названные в вопросе. None, если в вопросе есть код, которого в графе нет."""
    anchors: list[str] = []
    for code in find_codes(question):
        node = f"{code_type(code)}:{code}"
        if node not in kb.graph:
            return None
        anchors.append(node)
    names = [name for _, name in find_names(question, kb.vocabulary)]
    for kind, name in _vocabulary_pairs(kb.vocabulary, names):
        # «компрессор» внутри «КомпрессорМаш» – не отдельная сущность.
        if any(name.lower() in other.lower() and name != other for other in names):
            continue
        node = f"{kind}:{name}"
        if node in kb.graph and node not in anchors:
            anchors.append(node)
    return anchors


def _vocabulary_pairs(vocabulary: dict[str, list[str]], names: list[str]) -> list[tuple[str, str]]:
    wanted = set(names)
    return [(kind, name) for kind, items in vocabulary.items() for name in items if name in wanted]


Edge = tuple[str, str]


def _paths(graph: nx.DiGraph, anchors: list[str]) -> list[Edge]:
    """Рёбра всех путей длиной до трёх связей между парами опорных сущностей, направление не важно."""
    undirected = graph.to_undirected(as_view=True)
    edges: list[Edge] = []
    for first, second in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, first, second, cutoff=PATH_CUTOFF):
            for a, b in zip(path, path[1:], strict=False):
                edge = (a, b) if graph.has_edge(a, b) else (b, a)
                if edge not in edges:
                    edges.append(edge)
    return edges


def _neighbourhood(graph: nx.DiGraph, anchor: str) -> list[Edge]:
    """Рёбра окрестности узла на глубину до двух связей: сначала ближние, потом дальние по важности типа."""
    by_depth: list[list[Edge]] = []
    seen_nodes, frontier, seen_edges = {anchor}, [anchor], set()
    for _ in range(NEIGHBOURHOOD_DEPTH):
        layer: list[Edge] = []
        next_frontier: list[str] = []
        for node in frontier:
            for edge in [*graph.out_edges(node), *graph.in_edges(node)]:
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                layer.append(edge)
                other = edge[1] if edge[0] == node else edge[0]
                if other not in seen_nodes:
                    seen_nodes.add(other)
                    next_frontier.append(other)
        by_depth.append(sorted(layer, key=lambda edge: _edge_rank(graph, edge)))
        frontier = next_frontier
    return [edge for layer in by_depth for edge in layer]


def _edge_rank(graph: nx.DiGraph, edge: Edge) -> tuple[int, int, Edge]:
    attrs = graph.edges[edge]
    kind = attrs.get("type", "")
    priority = TYPE_PRIORITY.index(kind) if kind in TYPE_PRIORITY else len(TYPE_PRIORITY)
    return priority, -attrs.get("weight", 1), edge


def _with_replacements(graph: nx.DiGraph, edges: list[Edge]) -> list[Edge]:
    """Добавляет цепочки REPLACED_BY от деталей среди фактов и поставщиков последних деталей цепочек."""
    result = list(edges)
    parts = [node for edge in edges for node in edge if graph.nodes[node].get("type") == "Part"]
    for part in dict.fromkeys(parts):
        current, visited = part, {part}
        while True:
            replacements = [
                t for t in graph.successors(current) if graph.edges[current, t].get("type") == "REPLACED_BY"
            ]
            if not replacements or replacements[0] in visited:
                break
            nxt = replacements[0]
            result.append((current, nxt))
            visited.add(nxt)
            current = nxt
        if current != part:
            result += [
                (current, t) for t in graph.successors(current) if graph.edges[current, t].get("type") == "SUPPLIED_BY"
            ]
    return list(dict.fromkeys(result))


def _fact(graph: nx.DiGraph, edge: Edge) -> str:
    source, target = edge
    return f"{graph.nodes[source].get('name', source)} -{graph.edges[edge].get('type')}-> " + str(
        graph.nodes[target].get("name", target)
    )


def _facts_within_budget(graph: nx.DiGraph, edges: list[Edge]) -> tuple[list[str], list[Edge]]:
    facts, used, size = [], [], 0
    for edge in edges:
        fact = _fact(graph, edge)
        size += len(fact) + 6  # номер, точка и перевод строки
        if size > MAX_FACT_CHARS:
            break
        facts.append(fact)
        used.append(edge)
    return facts, used


def _replacement_hint(graph: nx.DiGraph, edges: list[Edge]) -> str:
    """Цепочки замен, посчитанные кодом: маленькой модели их проще прочитать готовыми."""
    replaced = {s: t for s, t in edges if graph.edges[s, t].get("type") == "REPLACED_BY"}
    starts = [part for part in replaced if part not in replaced.values()]
    chains = []
    for start in starts:
        chain, current = [start], start
        while current in replaced and replaced[current] not in chain:
            current = replaced[current]
            chain.append(current)
        chains.append(" -> ".join(graph.nodes[node].get("name", node) for node in chain))
    if not chains:
        return ""
    return "Цепочки замен (последняя деталь – актуальная): " + "; ".join(chains) + "\n\n"


def _documents(graph: nx.DiGraph, edges: list[Edge]) -> list[str]:
    chunks = [chunk for edge in edges for chunk in graph.edges[edge].get("sources", [])]
    return list(dict.fromkeys(chunk.split("#")[0] for chunk in chunks))


def _is_refusal(text: str) -> bool:
    return "не найдено" in text.lower()
