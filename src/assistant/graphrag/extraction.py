"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    EXTRACTION_SCHEMA,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

SYSTEM = """Ты извлекаешь граф знаний из документов сервисной службы компании «ПолярТех» (холодильное оборудование).
Верни JSON с двумя списками: entities (сущности) и relations (связи). Извлекай только то, что прямо сказано в тексте.

Типы сущностей и как записывать имя:
- ProductLine – линейка оборудования: Арктика, Бриз, Гольфстрим
- Model – код модели: AR-210, BZ-120, GS-510
- Component – вид узла строчными буквами в именительном падеже: пусковое реле, компрессор
- Part – номер детали: P-1042, P-1042B
- ErrorCode – код ошибки: E-17
- Procedure – номер регламента ремонта: SP-104
- Supplier – поставщик без кавычек и «ООО»: ХолодКомплект

Типы связей. Направление строго «субъект -> объект», как указано:
- BELONGS_TO: Model -> ProductLine (модель входит в линейку)
- HAS_PART: Model -> Part (в модели установлена деталь)
- IS_A: Part -> Component (деталь является узлом этого вида)
- INDICATES: ErrorCode -> Component (ошибка указывает на отказ узла)
- FIXED_BY: ErrorCode -> Procedure (ошибка устраняется по регламенту)
- REPLACED_BY: Part -> Part (старая деталь снята с производства и заменена новой; субъект – СТАРАЯ деталь)
- SUPPLIED_BY: Part -> Supplier (деталь поставляет поставщик)
- FAILED: Model -> Component (в отчёте о выезде зафиксирован отказ узла этой модели)

Правила:
- Коды пиши латиницей с дефисом: «ar 210» -> AR-210, «Е-17» -> E-17
- REPLACED_BY только если в тексте сказано, что деталь снята с производства и заменена.
  Сообщение о дефекте партии – не замена
- Номера документов (SB-2025-03, R-2025-017) не являются сущностями
- Если в тексте нет сущностей схемы, верни пустые списки"""

EXAMPLE_TEXT = """# Руководство по эксплуатации: чиллер GS-999

Чиллер GS-999 входит в линейку «Гольфстрим».

## Состав узлов

- циркуляционный насос – деталь P-9001, поставщик «ГидроТех»

## Коды ошибок

- E-77 – нет протока, отказ узла «циркуляционный насос»

# Сервисный бюллетень SB-2099-01

Деталь P-9001 (циркуляционный насос) снята с производства. Вместо неё устанавливается деталь P-9001B.
Поставщик детали P-9001B – «ВентПром».

# Регламент SP-199: Замена насоса

Регламент SP-199 применяется при ошибке E-77.

# Отчёт о выезде R-2099-001

Оборудование: чиллер gs 999. На панели горел код Е-77. Диагностика показала отказ узла: циркуляционный насос."""


def _entity(kind: str, name: str) -> dict:
    return {"type": kind, "name": name}


def _relation(subject: tuple[str, str], kind: str, obj: tuple[str, str]) -> dict:
    return {"subject_type": subject[0], "subject": subject[1], "type": kind, "object_type": obj[0], "object": obj[1]}


_MODEL, _LINE = ("Model", "GS-999"), ("ProductLine", "Гольфстрим")
_OLD, _NEW = ("Part", "P-9001"), ("Part", "P-9001B")
_PUMP, _ERROR, _PROC = ("Component", "циркуляционный насос"), ("ErrorCode", "E-77"), ("Procedure", "SP-199")
EXAMPLE_ANSWER = json.dumps(
    {
        "entities": [
            _entity(*item)
            for item in (
                _MODEL,
                _LINE,
                _OLD,
                _NEW,
                _PUMP,
                _ERROR,
                _PROC,
                ("Supplier", "ГидроТех"),
                ("Supplier", "ВентПром"),
            )
        ],
        "relations": [
            _relation(_MODEL, "BELONGS_TO", _LINE),
            _relation(_MODEL, "HAS_PART", _OLD),
            _relation(_OLD, "IS_A", _PUMP),
            _relation(_OLD, "SUPPLIED_BY", ("Supplier", "ГидроТех")),
            _relation(_ERROR, "INDICATES", _PUMP),
            _relation(_OLD, "REPLACED_BY", _NEW),
            _relation(_NEW, "SUPPLIED_BY", ("Supplier", "ВентПром")),
            _relation(_ERROR, "FIXED_BY", _PROC),
            _relation(_MODEL, "FAILED", _PUMP),
        ],
    },
    ensure_ascii=False,
)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Текст:\n{EXAMPLE_TEXT}"},
        {"role": "assistant", "content": EXAMPLE_ANSWER},
        {"role": "user", "content": f"Текст:\n{chunk.text}"},
    ]
    parsed = None
    for _ in range(2):
        parsed = _parse(llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=2048).content)
        if parsed is not None:
            break
    if parsed is None:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id)
    raw_entities, raw_relations = parsed
    names = vocabulary()

    def entity(kind: str, raw: str) -> Entity | None:
        name = canonical_name(kind, raw, names)
        return Entity(kind, name) if name else None

    result = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    entities: dict[str, Entity] = {}
    for item in raw_entities:
        entity = entity(item["type"], item["name"])
        if entity is None:
            result.dropped += 1
        else:
            entities.setdefault(entity.node_id, entity)
    relations: dict[tuple[str, str, str], Relation] = {}
    for item in raw_relations:
        subject = entity(item["subject_type"], item["subject"])
        obj = entity(item["object_type"], item["object"])
        relation = Relation(subject, item["type"], obj) if subject and obj else None
        if relation is None or not relation.matches_schema() or subject == obj:
            result.dropped += 1
            continue
        relations.setdefault((subject.node_id, relation.type, obj.node_id), relation)
        for end in (subject, obj):
            entities.setdefault(end.node_id, end)
    result.entities = list(entities.values())
    result.relations = list(relations.values())
    return result


def _parse(content: str) -> tuple[list[dict], list[dict]] | None:
    """Разбирает ответ модели. None, если ответ не соответствует схеме извлечения."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    entities, relations = data.get("entities"), data.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return None
    entity_keys = ("type", "name")
    relation_keys = ("subject_type", "subject", "type", "object_type", "object")
    if not all(_has_strings(item, entity_keys) for item in entities):
        return None
    if not all(_has_strings(item, relation_keys) for item in relations):
        return None
    return entities, relations


def _has_strings(item, keys: tuple[str, ...]) -> bool:
    return isinstance(item, dict) and all(isinstance(item.get(key), str) for key in keys)
