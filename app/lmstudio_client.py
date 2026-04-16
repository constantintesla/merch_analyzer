from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageOps


@dataclass
class ItemClassification:
    raw_name: str
    normalized_name: str
    item_name: str
    confidence: float
    status: str


_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_ENGLISH_META_SNIPPET_RE = re.compile(
    r"(the user wants|based on (the |specific )?rules?|identify the beverage|"
    r"i will |i'?ll |assistant:|here is |according to |as an ai|"
    r"analyze the image|i see |the text on the label|it reads|"
    r"как (модель|ассистент))",
    re.IGNORECASE,
)

# Системный промпт для classify_crop (/recognize): напитки на полке, ответ строкой на русском.
BEVERAGE_CLASSIFY_SYSTEM_PROMPT = """You are an object recognition system for beverages on a shelf. The input photo always contains exactly one bottle or can. Your task: recognize what is shown and return the result strictly in Russian, in the specified format.

Important: Our company produces two brands of bottled water:
- Новотроицкая (газированная / негазированная)
- Пьютти (газированная / негазированная)

These names must be written exactly as shown above when recognized.

Rules:
- Identify the beverage brand and type as accurately as possible.
- If the product is Новотроицкая or Пьютти, specify carbonation type if visible (газированная / негазированная).
- For any other product (Coca-Cola, Fanta, Bonaqua, Святой источник, juice, etc.), output its name as recognized.
- Output format: "Напиток: {название}"
- If nothing is recognizable on the photo, output: "Ничего не обнаружено"
- No explanations, no extra text. Output only the result in Russian.

Examples:
Photo: Новотроицкая sparkling → Output: Напиток: Новотроицкая газированная
Photo: Пьютти still → Output: Напиток: Пьютти негазированная
Photo: Coca-Cola → Output: Напиток: Coca-Cola
Photo: Bonaqua still → Output: Напиток: Bonaqua негазированная
Photo: Святой источник 0.5 л → Output: Напиток: Святой источник
Photo: empty shelf → Output: Ничего не обнаружено

Follow these rules strictly. Start your response immediately with "Напиток:" or "Ничего не обнаружено".
Never output English explanations, meta-commentary, or phrases like "The user wants". Only the Russian line in the required format."""

BEVERAGE_RECHECK_USER_HINT = (
    "\n\nЭто повторная попытка: внимательно рассмотри этикетку, крышку и форму упаковки. "
    "Если бренд читается — укажи его в формате «Напиток: …». Ответь строго в том же формате, что в инструкции."
)


def _has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))


def _looks_like_english_meta_instruction(text: str) -> bool:
    """Текст похож на служебное рассуждение модели, а не на название напитка (латинские бренды не отбрасываем)."""
    t = (text or "").strip()
    if not t:
        return True
    if _ENGLISH_META_SNIPPET_RE.search(t):
        return True
    if re.search(r"\bthe user (wants|is asking|requests)\b", t, re.I):
        return True
    return False


class LMStudioClient:
    def __init__(self, base_url: str, model: str, timeout_sec: float = 25.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    @staticmethod
    def _maybe_downscale_for_lm(image: Image.Image, max_side: int) -> Image.Image:
        """Уменьшает длинную сторону до max_side (0 = без изменений)."""
        if max_side <= 0:
            return image
        w, h = image.size
        if max(w, h) <= max_side:
            return image
        return ImageOps.contain(image, (max_side, max_side), Image.Resampling.LANCZOS)

    @staticmethod
    def _image_to_data_url(image: Image.Image, *, quality: int = 88) -> str:
        from io import BytesIO

        q = max(40, min(95, int(quality)))
        buff = BytesIO()
        image.save(buff, format="JPEG", quality=q)
        payload = base64.b64encode(buff.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{payload}"

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("empty model response")
        if text.startswith("{"):
            return json.loads(text)
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        raise ValueError("cannot parse JSON payload from response")

    def _message_text_candidates(self, msg: dict[str, Any]) -> list[str]:
        """Строки для парсинга JSON: сначала content, затем reasoning_content (think-модели)."""
        candidates: list[str] = []
        raw_content = msg.get("content", "")
        if isinstance(raw_content, list):
            chunks: list[str] = []
            for part in raw_content:
                if isinstance(part, dict):
                    txt = part.get("text", part.get("content", ""))
                    if txt:
                        chunks.append(str(txt))
                elif part:
                    chunks.append(str(part))
            text = "\n".join(chunks).strip()
        else:
            text = str(raw_content or "").strip()
        if text:
            candidates.append(text)
        reasoning = str(msg.get("reasoning_content", "") or "").strip()
        if reasoning:
            candidates.append(reasoning)
        return candidates

    def _chat_completion_raw_response(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        *,
        timeout_sec: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
        }
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        wait = self.timeout_sec if timeout_sec is None else float(timeout_sec)
        req = urllib.request.Request(
            url=f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(body).encode("utf-8"),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=wait) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _parse_beverage_line(text: str) -> ItemClassification | None:
        """Парсит ответ «Напиток: …» / «Ничего не обнаружено» по всему тексту (модель может писать рассуждение сверху)."""
        t = (text or "").strip()
        if not t:
            return None
        if t.startswith("```"):
            t = re.sub(r"^```\w*\s*", "", t)
            t = re.sub(r"\s*```\s*$", "", t).strip()

        lines = [ln.strip() for ln in t.replace("\r\n", "\n").split("\n") if ln.strip()]
        for line in lines:
            low = line.lower()
            if (
                low.startswith("ничего не обнаружено")
                or "не обнаружено" in low
                or low in {"ничего", "нет товара", "товар не найден"}
            ):
                return ItemClassification(
                    raw_name="Ничего не обнаружено",
                    normalized_name="Ничего не обнаружено",
                    item_name="Ничего не обнаружено",
                    confidence=0.35,
                    status="unknown",
                )

        nap_found: list[str] = []
        for line in lines:
            m = re.match(r"^\s*Напиток\s*[:\-]\s*(.+)$", line, flags=re.IGNORECASE | re.DOTALL)
            if not m:
                continue
            name = m.group(1).strip()
            if not name or _looks_like_english_meta_instruction(name):
                continue
            nap_found.append(name)

        if nap_found:
            name = nap_found[-1]
            return ItemClassification(
                raw_name=name,
                normalized_name=name,
                item_name=name,
                confidence=0.9,
                status="ok",
            )

        # Частый случай: модель отвечает описанием и включает название в кавычках.
        quoted = re.findall(r"[\"«“](.{2,80}?)[\"»”]", t)
        for q in quoted:
            candidate = q.strip(" .,:;!?\t\r\n")
            low = candidate.lower()
            if not candidate or low in {"unknown", "неизвестно", "не знаю", "n/a"}:
                continue
            if _looks_like_english_meta_instruction(candidate):
                continue
            # Предпочитаем короткие «брендовые» куски, а не длинные слоганы.
            if len(candidate.split()) > 4:
                continue
            return ItemClassification(
                raw_name=candidate,
                normalized_name=candidate,
                item_name=candidate,
                confidence=0.82,
                status="ok",
            )

        # Некоторые VLM отдают JSON вместо строки формата «Напиток: ...».
        try:
            payload = LMStudioClient._extract_json(t)
            raw_item = str(payload.get("item_name", "")).strip()
            if raw_item and raw_item.lower() != "unknown" and not _looks_like_english_meta_instruction(raw_item):
                return ItemClassification(
                    raw_name=raw_item,
                    normalized_name=raw_item,
                    item_name=raw_item,
                    confidence=0.8,
                    status="ok",
                )
        except (ValueError, json.JSONDecodeError, TypeError):
            pass

        # Мягкий fallback: если модель написала только название без префикса.
        for line in lines:
            candidate = re.sub(r'^[>\-\*\d\.\)\s]+', "", line).strip().strip("'\"")
            candidate = re.sub(r"^(это|бренд)\s*[:\-]\s*", "", candidate, flags=re.IGNORECASE).strip()
            low = candidate.lower()
            if not candidate:
                continue
            if low in {"unknown", "неизвестно", "не знаю", "n/a"}:
                continue
            if "не обнаружено" in low:
                continue
            if len(candidate) > 64 and ("," in candidate or "." in candidate):
                continue
            if _looks_like_english_meta_instruction(candidate):
                continue
            return ItemClassification(
                raw_name=candidate,
                normalized_name=candidate,
                item_name=candidate,
                confidence=0.72,
                status="uncertain",
            )

        for m in re.finditer(r"(?im)^\s*Напиток:\s*(.+)$", t):
            name = m.group(1).strip()
            if name and not _looks_like_english_meta_instruction(name):
                return ItemClassification(
                    raw_name=name,
                    normalized_name=name,
                    item_name=name,
                    confidence=0.85,
                    status="ok",
                )
        return None

    @staticmethod
    def _coerce_result(payload: dict[str, Any]) -> ItemClassification:
        raw_name = str(payload.get("item_name", "")).strip() or "unknown"
        normalized_name = str(payload.get("normalized_name", raw_name)).strip() or raw_name
        # Не терять кириллицу: если модель дала русское в item_name, а normalized — латиницей/переводом
        if raw_name != "unknown" and _has_cyrillic(raw_name) and not _has_cyrillic(normalized_name):
            normalized_name = raw_name
        item_name = normalized_name
        try:
            conf = float(payload.get("confidence_raw", payload.get("confidence", 0.0)))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        status = str(payload.get("status", "ok")).strip() or "ok"
        return ItemClassification(
            raw_name=raw_name,
            normalized_name=normalized_name,
            item_name=item_name,
            confidence=conf,
            status=status,
        )

    def _chat_completion_content(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        *,
        timeout_sec: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response_json = self._chat_completion_raw_response(
            messages=messages,
            temperature=temperature,
            timeout_sec=timeout_sec,
            max_tokens=max_tokens,
        )
        err = response_json.get("error")
        if err:
            detail = err.get("message", err) if isinstance(err, dict) else err
            raise ValueError(f"LM Studio API error: {detail}")
        choices = response_json.get("choices") or []
        if not choices:
            raise ValueError("LM Studio: пустой choices в ответе")
        msg = choices[0].get("message") or {}
        candidates = self._message_text_candidates(msg)
        if candidates:
            return candidates[0]
        top_prediction = str(
            response_json.get("prediction", "")
            or response_json.get("generated_prediction", "")
            or ""
        ).strip()
        if top_prediction:
            return top_prediction
        return ""

    @staticmethod
    def _classification_quality(ic: ItemClassification) -> int:
        """Выше — лучше для выбора между двумя попытками."""
        if ic.status == "lmstudio_error":
            return 0
        name = (ic.item_name or "").strip().lower()
        if name in ("", "unknown"):
            return 1
        if ic.status == "uncertain":
            return 2
        if "ничего не обнаружено" in name:
            return 3
        if ic.status == "ok" and name:
            return 4
        return 2

    @staticmethod
    def _needs_recheck_classification(ic: ItemClassification) -> bool:
        """Нужна ли вторая попытка (unknown / ошибка / сомнение)."""
        if ic.status == "lmstudio_error":
            return True
        name = (ic.item_name or "").strip().lower()
        if name in ("", "unknown"):
            return True
        if ic.status == "uncertain":
            return True
        if "ничего не обнаружено" in name:
            redo = (os.getenv("LM_RECHECK_NOTHING_DETECTED") or "0").strip().lower()
            return redo in {"1", "true", "yes", "on"}
        return False

    def classify_crop_with_recheck(self, crop_image: Image.Image) -> ItemClassification:
        """Первая попытка classify_crop; при «плохом» ответе — вторая с подсказкой (если LM_RECHECK_UNKNOWN не выключен)."""
        first = self.classify_crop(crop_image, retry=False)
        off = (os.getenv("LM_RECHECK_UNKNOWN") or "1").strip().lower()
        if off in {"0", "false", "no", "off"}:
            return first
        if not self._needs_recheck_classification(first):
            return first
        second = self.classify_crop(crop_image, retry=True)
        q1 = self._classification_quality(first)
        q2 = self._classification_quality(second)
        return second if q2 > q1 else first

    def classify_crop(self, crop_image: Image.Image, *, retry: bool = False) -> ItemClassification:
        max_side_raw = (os.getenv("LM_CROP_MAX_SIDE") or "0").strip()
        try:
            max_side = int(max_side_raw)
        except ValueError:
            max_side = 0
        jpeg_q_raw = (os.getenv("LM_CROP_JPEG_QUALITY") or "88").strip()
        try:
            jpeg_q = int(jpeg_q_raw)
        except ValueError:
            jpeg_q = 88
        # 256 токенов часто обрезает JSON у think/VLM; по умолчанию 2048. 0 = не передавать max_tokens (лимит сервера).
        max_tokens: int | None
        mtr = os.getenv("LM_CLASSIFY_MAX_TOKENS")
        if mtr is None or str(mtr).strip() == "":
            max_tokens = 2048
        elif str(mtr).strip() == "0":
            max_tokens = None
        else:
            try:
                max_tokens = max(32, min(8192, int(str(mtr).strip())))
            except ValueError:
                max_tokens = 2048

        prepared = self._maybe_downscale_for_lm(crop_image.copy(), max_side)
        data_url = self._image_to_data_url(prepared, quality=jpeg_q)
        user_text = (
            "Распознай напиток на изображении. Ответь строго в формате из системных инструкций. "
            "Только одна строка на русском: «Напиток: …» или «Ничего не обнаружено». "
            "Без английского текста и без рассуждений."
        )
        if retry:
            user_text += BEVERAGE_RECHECK_USER_HINT
        messages = [
            {"role": "system", "content": BEVERAGE_CLASSIFY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        try:
            response_json = self._chat_completion_raw_response(
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            err = response_json.get("error")
            if err:
                detail = err.get("message", err) if isinstance(err, dict) else err
                raise ValueError(f"LM Studio API error: {detail}")
            choices = response_json.get("choices") or []
            if not choices:
                raise ValueError("LM Studio: пустой choices в ответе")
            msg = choices[0].get("message") or {}
            for part in self._message_text_candidates(msg):
                parsed = self._parse_beverage_line(part)
                if parsed is not None:
                    return parsed
            return ItemClassification(
                raw_name="unknown",
                normalized_name="unknown",
                item_name="unknown",
                confidence=0.25,
                status="unknown",
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError):
            return ItemClassification(
                raw_name="unknown",
                normalized_name="unknown",
                item_name="unknown",
                confidence=0.0,
                status="lmstudio_error",
            )

    @staticmethod
    def _batch_classify_from_model_text(text: str, n: int) -> list[ItemClassification]:
        """Парсит JSON с полями results/items: [{slot, line}, ...] в список классификаций длины n."""
        def _unk() -> ItemClassification:
            return ItemClassification(
                raw_name="unknown",
                normalized_name="unknown",
                item_name="unknown",
                confidence=0.25,
                status="unknown",
            )

        if n <= 0:
            return []
        out = [_unk() for _ in range(n)]
        t = (text or "").strip()
        if not t:
            return out

        if n == 1 and not (t.startswith("{") or t.startswith("[")):
            one = LMStudioClient._parse_beverage_line(t)
            if one is not None:
                return [one]

        payload: Any = None
        try:
            if t.startswith("{") or t.startswith("["):
                payload = json.loads(t)
            else:
                payload = LMStudioClient._extract_json(t)
        except (json.JSONDecodeError, ValueError):
            try:
                payload = LMStudioClient._extract_json(t)
            except (ValueError, json.JSONDecodeError):
                return out

        rows: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            arr = payload.get("results") or payload.get("items") or payload.get("data")
            if isinstance(arr, list):
                rows = [x for x in arr if isinstance(x, dict)]
        elif isinstance(payload, list):
            rows = [x for x in payload if isinstance(x, dict)]

        by_slot: dict[int, str] = {}
        for row in rows:
            sl = row.get("slot", row.get("index", row.get("i")))
            if sl is None:
                continue
            try:
                si = int(sl)
            except (TypeError, ValueError):
                continue
            line = row.get("line") or row.get("text") or row.get("napitok") or row.get("answer")
            if line is None:
                continue
            by_slot[si] = str(line).strip()

        for si in range(n):
            line = by_slot.get(si)
            if not line:
                continue
            ic = LMStudioClient._parse_beverage_line(line)
            if ic is not None:
                out[si] = ic
        return out

    def _classify_max_tokens_env(self) -> int | None:
        mtr = os.getenv("LM_CLASSIFY_MAX_TOKENS")
        if mtr is None or str(mtr).strip() == "":
            return 2048
        if str(mtr).strip() == "0":
            return None
        try:
            return max(32, min(8192, int(str(mtr).strip())))
        except ValueError:
            return 2048

    def _batch_classify_max_tokens(self, n: int) -> int | None:
        raw = (os.getenv("LM_BATCH_CLASSIFY_MAX_TOKENS") or "").strip()
        if raw and raw != "0":
            try:
                return max(256, min(8192, int(raw)))
            except ValueError:
                pass
        base = self._classify_max_tokens_env()
        if base is None:
            return None
        return max(base, min(8192, 256 + 128 * max(1, n)))

    def classify_crops_batch(self, crop_images: list[Image.Image]) -> list[ItemClassification]:
        """Один запрос chat/completions с N изображениями; ответ — JSON с results[{slot, line}]."""
        n = len(crop_images)
        if n == 0:
            return []
        max_side_raw = (os.getenv("LM_CROP_MAX_SIDE") or "0").strip()
        try:
            max_side = int(max_side_raw)
        except ValueError:
            max_side = 0
        jpeg_q_raw = (os.getenv("LM_CROP_JPEG_QUALITY") or "88").strip()
        try:
            jpeg_q = int(jpeg_q_raw)
        except ValueError:
            jpeg_q = 88

        err_ic = ItemClassification(
            raw_name="unknown",
            normalized_name="unknown",
            item_name="unknown",
            confidence=0.0,
            status="lmstudio_error",
        )

        user_intro = (
            f"Перед тобой {n} изображений в порядке slot 0..{n - 1} "
            f"(первое изображение = slot 0, далее по порядку). "
            "Для каждого слота распознай напиток по правилам из системного промпта. "
            "Верни ТОЛЬКО один JSON без markdown и без текста вокруг. Формат:\n"
            '{"results":[{"slot":0,"line":"Напиток: … или Ничего не обнаружено"}, ...]}\n'
            f"Поле line — ровно одна строка на русском для этого слота. "
            f"В массиве results должны быть все слоты от 0 до {n - 1}."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user_intro}]
        for crop in crop_images:
            prepared = self._maybe_downscale_for_lm(crop.copy(), max_side)
            data_url = self._image_to_data_url(prepared, quality=jpeg_q)
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages = [
            {"role": "system", "content": BEVERAGE_CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        max_tokens = self._batch_classify_max_tokens(n)
        try:
            response_json = self._chat_completion_raw_response(
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            err = response_json.get("error")
            if err:
                detail = err.get("message", err) if isinstance(err, dict) else err
                raise ValueError(f"LM Studio API error: {detail}")
            choices = response_json.get("choices") or []
            if not choices:
                raise ValueError("LM Studio: пустой choices в ответе")
            msg = choices[0].get("message") or {}
            combined = "\n".join(self._message_text_candidates(msg))
            return self._batch_classify_from_model_text(combined, n)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError):
            return [err_ic] * n

    def classify_crops_batch_chunked(self, crop_images: list[Image.Image]) -> list[ItemClassification]:
        """Несколько classify_crops_batch, если кропов больше LM_BATCH_MAX_CROPS_PER_REQUEST."""
        n = len(crop_images)
        if n == 0:
            return []
        raw = (os.getenv("LM_BATCH_MAX_CROPS_PER_REQUEST") or "12").strip()
        try:
            cap = max(1, min(64, int(raw)))
        except ValueError:
            cap = 12
        out: list[ItemClassification] = []
        for i in range(0, n, cap):
            chunk = crop_images[i : i + cap]
            out.extend(self.classify_crops_batch(chunk))
        return out

    def recheck_item_name_text(self, item_name: str) -> ItemClassification:
        prompt = (
            "Проверь название товара и нормализуй его. "
            "КРИТИЧНО: не переводи на английский; если вход на русском (кириллица), ответ тоже на русском. "
            "Если название не похоже на товар или неуверенно, верни unknown. "
            "Верни только JSON с полями: "
            '{"item_name":"string","normalized_name":"string","confidence_raw":0.0,"status":"ok|uncertain|unknown"}. '
            f"Входное название: {item_name!r}"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            content = self._chat_completion_content(messages=messages, temperature=0.0)
            parsed = self._extract_json(str(content))
            return self._coerce_result(parsed)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            return ItemClassification(
                raw_name="unknown",
                normalized_name="unknown",
                item_name="unknown",
                confidence=0.0,
                status="lmstudio_error",
            )

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, v))

    @staticmethod
    def _parse_bbox_norm(raw: Any) -> dict[str, float] | None:
        if not isinstance(raw, dict):
            return None
        x1 = LMStudioClient._clamp01(raw.get("x1", raw.get("left", 0)))
        y1 = LMStudioClient._clamp01(raw.get("y1", raw.get("top", 0)))
        x2 = LMStudioClient._clamp01(raw.get("x2", raw.get("right", 0)))
        y2 = LMStudioClient._clamp01(raw.get("y2", raw.get("bottom", 0)))
        if x2 <= x1 or y2 <= y1:
            return None
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    @staticmethod
    def _planogram_items_from_payload(payload: dict[str, Any], iw: int, ih: int) -> list[dict[str, Any]]:
        """Преобразует JSON модели в элементы, совместимые с build_planogram_template_from_items."""
        w = max(1, int(iw))
        h = max(1, int(ih))
        rows: list[tuple[int, int, str, dict[str, float], float]] = []

        def push(shelf_id: int, pos: int, name: str, bn: dict[str, float] | None, conf: float) -> None:
            sid = int(shelf_id)
            p = int(pos)
            nm = str(name).strip() or "unknown"
            if sid <= 0 or p <= 0:
                return
            if bn is None:
                return
            rows.append((sid, p, nm, bn, conf))

        shelves = payload.get("shelves")
        if isinstance(shelves, list) and shelves:
            for sh in shelves:
                if not isinstance(sh, dict):
                    continue
                sid = int(sh.get("shelf_id", sh.get("id", 0)) or 0)
                items = sh.get("items", sh.get("positions", []))
                if not isinstance(items, list):
                    continue
                for ent in items:
                    if not isinstance(ent, dict):
                        continue
                    pos = int(ent.get("position_in_shelf", ent.get("position", 0)) or 0)
                    nm = str(ent.get("item_name", ent.get("name", ""))).strip()
                    bn = LMStudioClient._parse_bbox_norm(ent.get("bbox_norm", ent.get("bbox")))
                    try:
                        cf = float(ent.get("confidence", ent.get("confidence_raw", 0.75)))
                    except (TypeError, ValueError):
                        cf = 0.75
                    cf = max(0.0, min(1.0, cf))
                    push(sid, pos, nm, bn, cf)
        else:
            positions_flat = payload.get("positions")
            if isinstance(positions_flat, list):
                for ent in positions_flat:
                    if not isinstance(ent, dict):
                        continue
                    sid = int(ent.get("shelf_id", 0) or 0)
                    pos = int(ent.get("position_in_shelf", ent.get("position", 0)) or 0)
                    nm = str(ent.get("item_name", ent.get("name", ""))).strip()
                    bn = LMStudioClient._parse_bbox_norm(ent.get("bbox_norm", ent.get("bbox")))
                    try:
                        cf = float(ent.get("confidence", ent.get("confidence_raw", 0.75)))
                    except (TypeError, ValueError):
                        cf = 0.75
                    cf = max(0.0, min(1.0, cf))
                    push(sid, pos, nm, bn, cf)

        if not rows:
            raise ValueError("в ответе модели нет валидных позиций с bbox_norm")

        rows.sort(key=lambda t: (t[0], t[1]))
        out: list[dict[str, Any]] = []
        for i, (sid, pos, nm, bn, cf) in enumerate(rows, start=1):
            x1p = bn["x1"] * w
            y1p = bn["y1"] * h
            x2p = bn["x2"] * w
            y2p = bn["y2"] * h
            out.append(
                {
                    "item_id": i,
                    "group_id": 0,
                    "bbox": {"x1": x1p, "y1": y1p, "x2": x2p, "y2": y2p},
                    "row": sid,
                    "shelf_id": sid,
                    "position_in_shelf": pos,
                    "lm_item_name": nm,
                    "lm_confidence": cf,
                }
            )
        return out

    def planogram_from_full_image(
        self,
        image: Image.Image,
        *,
        timeout_sec: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Одно изображение полки: модель возвращает JSON с полками (сверху вниз = 1,2,…),
        позиции на полке слева направо, для каждой — item_name и bbox_norm в долях кадра [0,1].
        """
        data_url = self._image_to_data_url(image)
        prompt = (
            "Ты анализируешь фото торговой полки (планограмма). "
            "Нужно перечислить ВСЕ видимые товарные позиции: порядок слева направо на каждой полке, "
            "полки считай сверху вниз (верхняя полка = shelf_id 1, следующая = 2, …). "
            "Не выдумывай товары: если не читается — кратко опиши что видно или unknown. "
            "КРИТИЧНО: название item_name на том же языке, что на упаковке (кириллица — без перевода на английский). "
            "Для КАЖДОЙ позиции обязательно укажи bbox_norm: прямоугольник вокруг товара в долях ширины/высоты кадра, "
            "ключи x1,y1,x2,y2, значения от 0 до 1 (левый верхний угол кадра = 0,0). "
            "Верни ТОЛЬКО один JSON-объект без markdown и без текста вокруг. Формат:\n"
            '{"shelves":[{"shelf_id":1,"items":['
            '{"position_in_shelf":1,"item_name":"строка","bbox_norm":{"x1":0.0,"y1":0.0,"x2":0.1,"y2":0.1},"confidence":0.8}'
            "]}]}\n"
            "Поле confidence — твоя уверенность в названии [0,1]. Если несколько одинаковых SKU подряд — отдельные позиции."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        wait = timeout_sec if timeout_sec is not None else max(self.timeout_sec, 90.0)
        content = self._chat_completion_content(
            messages=messages,
            temperature=0.0,
            timeout_sec=wait,
            max_tokens=4096,
        )
        try:
            payload = self._extract_json(str(content))
        except (json.JSONDecodeError, ValueError):
            # Иногда VLM отвечает длинным текстом без валидного JSON.
            # Повторно просим ту же модель преобразовать ответ в строгий JSON.
            repair_prompt = (
                "Преобразуй следующий текст в СТРОГИЙ JSON-объект формата:\n"
                '{"shelves":[{"shelf_id":1,"items":[{"position_in_shelf":1,"item_name":"string","bbox_norm":{"x1":0.0,"y1":0.0,"x2":0.1,"y2":0.1},"confidence":0.8}]}]}\n'
                "Только JSON, без markdown и без комментариев. "
                "Если данных не хватает, верни пустой объект с shelves: [].\n\n"
                f"Текст:\n{content}"
            )
            repair_content = self._chat_completion_content(
                messages=[{"role": "user", "content": repair_prompt}],
                temperature=0.0,
                timeout_sec=wait,
                max_tokens=2048,
            )
            payload = self._extract_json(str(repair_content))
        if not isinstance(payload, dict):
            raise ValueError("модель вернула не объект JSON")
        return self._planogram_items_from_payload(payload, image.width, image.height)

    def assess_placement(
        self,
        *,
        sku: str,
        reference_result: dict[str, Any],
        fact_result: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "Ты сравниваешь эталон и факт выкладки одного SKU. "
            "Нужно вернуть, насколько плохо расставлено, в формате JSON. "
            "Верни ТОЛЬКО JSON с полями: "
            '{"badness_score":0,"verdict":"ok|minor|bad|critical","reason":"string","recommendation":"string"}.\n'
            "Шкала badness_score: 0=идеально, 100=очень плохо.\n"
            f"SKU: {sku}\n"
            f"Эталон: {json.dumps(reference_result, ensure_ascii=False)}\n"
            f"Факт: {json.dumps(fact_result, ensure_ascii=False)}"
        )
        try:
            content = self._chat_completion_content(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
            )
            payload = self._extract_json(str(content))
            score = int(max(0, min(100, round(float(payload.get("badness_score", 100))))))
            verdict = str(payload.get("verdict", "bad")).strip() or "bad"
            reason = str(payload.get("reason", "")).strip() or "Не удалось получить объяснение."
            recommendation = (
                str(payload.get("recommendation", "")).strip()
                or "Проверьте соответствие SKU эталону и переустановите товар."
            )
            return {
                "badness_score": score,
                "verdict": verdict,
                "reason": reason,
                "recommendation": recommendation,
                "status": "ok",
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, TypeError):
            return {
                "badness_score": 100,
                "verdict": "bad",
                "reason": "LM Studio недоступен или вернул невалидный ответ.",
                "recommendation": "Проверьте подключение к LM Studio и повторите запрос.",
                "status": "lmstudio_error",
            }
