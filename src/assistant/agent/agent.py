"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS, ArgumentError, validate_args
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

ERP_ATTEMPTS = 3  # первый запрос и два повтора
ORDER_TOOL = "create_service_order"
NOT_FOUND = "В базе знаний не найдено."
ERP_FAILED = "Данные ERP получить не удалось: сервис недоступен."

SYSTEM = f"""Ты ассистент диспетчера сервисной службы компании «ПолярТех» (холодильное оборудование).
Отвечай только по результатам инструментов. Порядок работы:
1. Если в вопросе есть модель, код ошибки или узел, сначала вызови search_knowledge, чтобы узнать номер детали.
   Деталь могла быть снята с производства: тогда используй последнюю деталь цепочки замен.
2. Остаток узнавай через check_stock, цену – через get_price. Номер детали бери только из вопроса или из
   результатов инструментов, склад – из списка.
3. create_service_order вызывай, только если пользователь просит оформить заказ. Количество бери только из вопроса.
4. Обзорные вопросы по всему корпусу (что чаще отказывает, где больше отказов) – через summarize_topic.
Итоговый ответ пиши по-русски, кратко: номер детали, остаток в штуках, цену в рублях – числа точно как
в результатах. Различай три случая: «остаток 0» – деталь есть в ERP, но закончилась; «данных нет» – ERP не знает
деталь; «ERP недоступна» – данные получить не удалось. Ничего не придумывай.
Если данных для ответа нет, напиши: «{NOT_FOUND}»"""

WORD_NUMBERS = {
    "один": 1, "одну": 1, "одна": 1, "одной": 1, "два": 2, "две": 2, "двух": 2, "три": 3, "трёх": 3, "трех": 3,
    "четыре": 4, "четырёх": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
}  # fmt: skip
_NUMBER = re.compile(r"(?<![\w-])(?:\d{1,3}(?:[   ]\d{3})+|\d+)(?![\w-])")
_REPLACED = re.compile(r"(P-\d{4}[A-Z]?)\s*-REPLACED_BY->\s*(P-\d{4}[A-Z]?)")


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    run = _Run(question, llm, tools, confirm)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    while True:
        if len(run.steps) >= max_steps:
            return run.finish(messages, run.interrupted(messages, max_steps))
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        if not response.tool_calls:
            return run.finish(messages, response.content.strip())
        messages.append(response.as_message())
        for call in response.tool_calls:
            if len(run.steps) >= max_steps:
                messages.append(tool_message(call, {"error": "лимит шагов исчерпан, вызов не выполнен"}))
                continue
            stop = run.execute(call)
            if stop is not None:
                return stop
            messages.append(tool_message(call, run.observation(run.steps[-1])))


class _Run:
    """Состояние одного прогона агента: шаги, известные коды и числа, кеш вызовов."""

    def __init__(self, question: str, llm: LLMClient, tools: dict[str, Callable[..., dict]], confirm: bool) -> None:
        self.question = question
        self.llm = llm
        self.tools = tools
        self.confirm = confirm
        self.steps: list[ToolStep] = []
        self.cache: dict[str, ToolStep] = {}
        self.sources: list[str] = []
        self.erp_failed = False
        self.order_id: str | None = None

    # --- выполнение вызовов ---

    def execute(self, call: ToolCall) -> AgentResult | None:
        """Выполняет вызов и записывает шаг. Возвращает AgentResult, если агенту пора остановиться."""
        name, args = call.name, call.arguments
        if name not in self.tools:
            return self._reject(name, args, f"неизвестный инструмент {name!r}, доступны: {', '.join(self.tools)}")
        if "_raw" in args:
            return self._reject(name, args, "аргументы не разобраны как JSON, повторите вызов с корректным JSON")
        try:
            clean = validate_args(name, args)
        except ArgumentError as exc:
            return self._reject(name, args, f"неверные аргументы: {exc}")
        part = clean.get("part_id")
        if part and part not in self.known_codes():
            return self._reject(
                name,
                clean,
                f"номер детали {part} не встречался ни в вопросе, ни в результатах инструментов. "
                "Сначала вызовите search_knowledge, чтобы узнать номер детали",
            )
        if name == ORDER_TOOL:
            try:
                stop = self._check_order(clean)
            except ArgumentError as exc:
                return self._reject(name, clean, str(exc))
            if stop is not None:
                return stop
        key = json.dumps([name, clean], ensure_ascii=False, sort_keys=True)
        if key in self.cache:
            cached = self.cache[key]
            self.steps.append(ToolStep(name, clean, cached.result, cached.error))
            return None
        step = self._call(name, clean)
        self.cache[key] = step
        self.steps.append(step)
        if step.result:
            self.sources += [s for s in step.result.get("sources", []) if s not in self.sources]
            if name == ORDER_TOOL and step.error is None:
                self.order_id = str(step.result.get("order_id", "")) or None
        return None

    def _call(self, name: str, args: dict) -> ToolStep:
        for _ in range(ERP_ATTEMPTS):
            try:
                result = self.tools[name](**args)
            except ErpUnavailable:
                continue
            except Exception as exc:  # инструмент упал: модель узнает об этом и продолжит
                return ToolStep(name, args, None, f"инструмент завершился ошибкой: {exc}")
            error = result.get("error") if isinstance(result, dict) else "инструмент вернул не словарь"
            return ToolStep(name, args, result if isinstance(result, dict) else None, error)
        self.erp_failed = True
        return ToolStep(name, args, None, f"ERP недоступна после {ERP_ATTEMPTS} попыток, данные получить не удалось")

    def _reject(self, name: str, args: dict, error: str) -> None:
        self.steps.append(ToolStep(name, args, None, error))
        return None

    def _check_order(self, args: dict) -> AgentResult | None:
        part = args["part_id"]
        if args["qty"] not in self.quantities_in_question():
            self.steps.append(ToolStep(ORDER_TOOL, args, None, "количество не названо пользователем"))
            return self.result(
                f"Уточните, пожалуйста, сколько штук детали {part} заказать "
                f"для модели {args['model']} со склада {args['warehouse']}?"
            )
        replacement = self.discontinued().get(part)
        if replacement is not None:
            hint = f", замена – {replacement}" if replacement else ""
            raise ArgumentError(f"деталь {part} снята с производства{hint}. Закажите актуальную деталь")
        if self.confirm:
            return None
        answer = (
            f"Готов оформить заказ: деталь {part}, модель {args['model']}, склад {args['warehouse']}, "
            f"в количестве {args['qty']}. Заказ не создан: подтвердите его."
        )
        return self.result(answer, needs_confirmation=True, pending_action={"tool": ORDER_TOOL, "args": args})

    def observation(self, step: ToolStep) -> dict:
        """Что увидит модель после шага. К снятой с производства детали добавляется подсказка."""
        if step.error is not None:
            return {"error": step.error}
        result = dict(step.result or {})
        if result.get("discontinued") is True:
            replacement = self.discontinued().get(result.get("part_id", ""))
            result["note"] = "деталь снята с производства" + (f", замена – {replacement}" if replacement else "")
        return result

    # --- что агент уже знает ---

    def evidence(self) -> str:
        """Вопрос и результаты выполненных инструментов: всё, на что может опираться ответ."""
        results = [step.result for step in self.steps if step.result is not None]
        return self.question + "\n" + json.dumps(results, ensure_ascii=False)

    def known_codes(self) -> set[str]:
        return set(find_codes(self.evidence()))

    def quantities_in_question(self) -> set[int]:
        text = self.question.lower()
        numbers = {int(_digits(raw)) for raw in _NUMBER.findall(text)}
        words = {value for word, value in WORD_NUMBERS.items() if re.search(rf"(?<!\w){word}(?!\w)", text)}
        return numbers | words

    def discontinued(self) -> dict[str, str]:
        """Детали, снятые с производства: номер -> замена (пустая строка, если замена неизвестна)."""
        found: dict[str, str] = {}
        for step in self.steps:
            result = step.result or {}
            if result.get("discontinued") is True and result.get("part_id"):
                found.setdefault(str(result["part_id"]), "")
            for fact in result.get("facts", []) if isinstance(result.get("facts"), list) else []:
                for old, new in _REPLACED.findall(str(fact)):
                    found[old] = new
        # Доводим замену до конца цепочки: P-2054 -> P-2054B -> P-2054C.
        for old, new in list(found.items()):
            seen = {old}
            while new in found and found[new] and found[new] not in seen:
                seen.add(new)
                new = found[new]
            found[old] = new
        return found

    # --- итоговый ответ ---

    def interrupted(self, messages: list[dict], max_steps: int) -> str:
        """Ответ, когда лимит шагов исчерпан: модель подводит итог без инструментов, иначе итог пишет код."""
        prompt = (
            "Лимит вызовов инструментов исчерпан. Кратко ответь по уже полученным результатам "
            "и сообщи, что работа прервана."
        )
        reply = self.llm.chat([*messages, {"role": "user", "content": prompt}])
        text = reply.content.strip() if not reply.tool_calls else ""
        notice = f"Работа прервана: исчерпан лимит в {max_steps} вызовов инструментов."
        if not text:
            done = [step for step in self.steps if step.error is None and step.result is not None]
            learned = "; ".join(f"{step.tool}: {json.dumps(step.result, ensure_ascii=False)}" for step in done)
            return f"{notice} Удалось узнать: {learned}" if learned else notice
        return text if "прерван" in text.lower() else f"{notice} {text}"

    def finish(self, messages: list[dict], draft: str) -> AgentResult:
        """Итоговый ответ: сообщает об отказе ERP и номере созданного заказа."""
        answer = draft or NOT_FOUND
        if self.erp_failed and "не удалось" not in answer.lower():
            answer = f"{answer} {ERP_FAILED}"
        if self.order_id and self.order_id not in answer:
            answer = f"{answer} Заказ {self.order_id} создан."
        return self.result(answer, refused=self._refused(answer))

    def _refused(self, answer: str) -> bool:
        if "не найдено" in answer.lower():
            return True
        # Ни один инструмент не дал данных (и дело не в недоступной ERP) – отвечать не по чему.
        useful = [step for step in self.steps if step.error is None and not (step.result or {}).get("refused")]
        return bool(self.steps) and not useful and not self.erp_failed

    def result(self, answer: str, **fields) -> AgentResult:
        return AgentResult(answer, steps=self.steps, sources=list(self.sources), **fields)


def _digits(raw: str) -> str:
    return re.sub(r"\D", "", raw)
