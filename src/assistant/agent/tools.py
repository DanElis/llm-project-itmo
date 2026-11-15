"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

_PART_ID = {"type": "string", "description": "Номер детали вида P-1042 или P-1042B"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES, "description": "Склад"}


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
TOOL_SCHEMAS: list[dict] = [
    _function(
        "search_knowledge",
        "Поиск по графу знаний о конкретных моделях, кодах ошибок, узлах, деталях, заменах деталей, поставщиках "
        "и регламентах. Возвращает ответ, факты и документы-источники. Вызывай первым, чтобы узнать номер "
        "актуальной детали: деталь может быть снята с производства и заменена.",
        {"question": {"type": "string", "description": "Вопрос на русском языке с кодами из запроса пользователя"}},
        ["question"],
    ),
    _function(
        "summarize_topic",
        "Обзорный ответ по всему корпусу документов: какие узлы чаще отказывают, у каких моделей больше отказов, "
        "какие детали сняты с производства в линейке.",
        {"question": {"type": "string", "description": "Обзорный вопрос на русском языке"}},
        ["question"],
    ),
    _function(
        "check_stock",
        "Остаток детали на складе по данным ERP и признак discontinued (снята с производства).",
        {"part_id": _PART_ID, "warehouse": _WAREHOUSE},
        ["part_id", "warehouse"],
    ),
    _function(
        "get_price",
        "Цена детали в рублях по данным ERP и признак discontinued (снята с производства).",
        {"part_id": _PART_ID},
        ["part_id"],
    ),
    _function(
        "create_service_order",
        "Создать заказ детали в ERP. Действие с последствиями: выполняется только после подтверждения человеком. "
        "Количество должен назвать пользователь.",
        {
            "part_id": _PART_ID,
            "warehouse": _WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1, "description": "Количество штук, названное пользователем"},
            "model": {"type": "string", "description": "Код модели оборудования, например AR-210"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]


class ArgumentError(ValueError):
    """Аргументы инструмента не прошли проверку. Текст ошибки возвращается модели."""


def validate_args(name: str, args: dict) -> dict:
    """Проверяет и нормализует аргументы инструмента. Бросает ArgumentError с понятным модели текстом."""
    schema = next((item["function"] for item in TOOL_SCHEMAS if item["function"]["name"] == name), None)
    if schema is None:
        raise ArgumentError(f"неизвестный инструмент {name}")
    properties = schema["parameters"]["properties"]
    missing = [key for key in schema["parameters"]["required"] if key not in args]
    extra = [key for key in args if key not in properties]
    if missing or extra:
        raise ArgumentError(f"нужны аргументы {', '.join(properties)}; не хватает: {missing}, лишние: {extra}")
    clean = dict(args)
    if "question" in clean and (not isinstance(clean["question"], str) or not clean["question"].strip()):
        raise ArgumentError("question должен быть непустой строкой")
    if "part_id" in clean:
        part = normalize_code(str(clean["part_id"]))
        if part is None or code_type(part) != "Part":
            raise ArgumentError(f"part_id {clean['part_id']!r} не похож на номер детали вида P-1042 или P-1042B")
        clean["part_id"] = part
    if "warehouse" in clean and clean["warehouse"] not in WAREHOUSES:
        raise ArgumentError(f"склад {clean['warehouse']!r} не из списка: {', '.join(WAREHOUSES)}")
    if "qty" in clean:
        qty = clean["qty"]
        if isinstance(qty, str) and qty.strip().isdigit():
            qty = int(qty)
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            raise ArgumentError(f"количество {clean['qty']!r} должно быть целым числом больше нуля")
        clean["qty"] = qty
    if "model" in clean:
        model = normalize_code(str(clean["model"]))
        if model is None or code_type(model) != "Model":
            raise ArgumentError(f"model {clean['model']!r} не похож на код модели вида AR-210")
        clean["model"] = model
    return clean


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def guarded(name: str, action: Callable[..., dict]) -> Callable[..., dict]:
        def tool(**args) -> dict:
            try:
                clean = validate_args(name, args)
            except ArgumentError as exc:
                return {"error": f"неверные аргументы: {exc}"}
            try:
                return action(**clean)
            except ErpNotFound as exc:
                return {"error": f"данных нет: ERP не знает такую позицию ({exc})", "not_found": True}
            except ErpRejected as exc:
                return {"error": f"ERP отклонила запрос: {exc}", "rejected": True}

        tool.__name__ = name
        return tool

    def search_knowledge(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        return erp.get_stock(part_id, warehouse)

    def get_price(part_id: str) -> dict:
        return erp.get_price(part_id)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        return erp.create_order(part_id, warehouse, qty, model)

    actions = {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
    return {name: guarded(name, action) for name, action in actions.items()}
