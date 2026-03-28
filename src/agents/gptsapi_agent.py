# src/agents/gptsapi_agent.py
import asyncio
import json
import os
import random
import socket
import time
from typing import Optional, Dict, Any, List
from urllib import error as urlerror
from urllib import request as urlrequest

# Auto-load .env if available.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from .base import BaseAgent, AgentResult
from ..utils.tokens import approx_tokens, cost_usd

DEFAULT_TEMP_ONLY_MODELS = {"gpt-5-mini"}
RETRIABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


class _SimpleHttpError(Exception):
    def __init__(self, status_code: int, url: str, body: str):
        super().__init__(f"HTTP {status_code} for {url}")
        self.status_code = int(status_code)
        self.url = url
        self.body = body


class GptsApiAgent(BaseAgent):
    """
    OpenAI-compatible client for gptsapi.net using stdlib urllib.
    This avoids the environment-specific hangs we observed on `import httpx`.
    """

    _sema_map: Dict[int, asyncio.Semaphore] = {}

    def __init__(self, **kwargs):
        extra = kwargs.get("extra") or {}
        super().__init__(**kwargs)

        bv = extra.get("base_url")
        if isinstance(bv, str) and bv.startswith("${") and bv.endswith("}"):
            bv = os.getenv(bv[2:-1], "")
        env_base = os.getenv("GPTSAPI_BASE_URL", "")
        self.base_url = (bv or env_base or "https://api.gptsapi.net").rstrip("/")

        api_key_env = extra.get("api_key_env", "GPTSAPI_API_KEY")
        self.api_key = os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError(
                f"GptsApiAgent needs API key in env {api_key_env} or OPENAI_API_KEY."
            )

        self._endpoint = (
            "/chat/completions"
            if self.base_url.endswith("/v1")
            else "/v1/chat/completions"
        )
        self.max_retries = int(os.getenv("GPTSAPI_RETRIES", "4"))
        self.backoff_base = float(os.getenv("GPTSAPI_BACKOFF_BASE", "0.6"))
        self._max_conc = int(os.getenv("GPTSAPI_MAX_CONCURRENCY", "3"))
        self._timeout_s = float(os.getenv("GPTSAPI_TIMEOUT", "300"))

        st = extra.get("send_temperature")
        if st is None:
            self._send_temperature = self.model not in DEFAULT_TEMP_ONLY_MODELS
        else:
            self._send_temperature = bool(st)

    @classmethod
    def _get_sema(cls, max_conc: int) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        key = id(loop)
        sem = cls._sema_map.get(key)
        if sem is None:
            sem = asyncio.Semaphore(max_conc)
            cls._sema_map[key] = sem
        return sem

    def _messages(self, user_prompt: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if (
            isinstance(self.system_prompt, str)
            and self.system_prompt.strip()
            and "claude" not in self.model.lower()
        ):
            msgs.append({"role": "system", "content": self.system_prompt.replace("\r\n", "\n")})
        if not isinstance(user_prompt, str):
            user_prompt = str(user_prompt)
        user_prompt = user_prompt.replace("\r\n", "\n")
        if not user_prompt.strip():
            user_prompt = "Please answer the question."
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    def _request_once(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        url = f"{self.base_url}{self._endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url, data=data, headers=headers, method="POST")
        opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))
        try:
            with opener.open(req, timeout=self._timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body)
        except urlerror.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise _SimpleHttpError(e.code, url, body) from e

    async def _post_with_retry(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        sem = self._get_sema(self._max_conc)
        last_exc: Optional[Exception] = None
        temp_stripped_once = False

        async with sem:
            for attempt in range(self.max_retries):
                try:
                    return await asyncio.to_thread(self._request_once, payload, headers)
                except _SimpleHttpError as e:
                    print(
                        f"[DEBUG][HTTPError] status={e.status_code} url={e.url} body={e.body[:800]}"
                    )
                    if (
                        e.status_code == 400
                        and not temp_stripped_once
                        and "temperature" in e.body.lower()
                        and "unsupported" in e.body.lower()
                        and "temperature" in payload
                    ):
                        temp_stripped_once = True
                        del payload["temperature"]
                        print(
                            "[DEBUG] Detected temperature incompatibility. Stripped 'temperature' and retrying."
                        )
                        last_exc = e
                        continue
                    last_exc = e
                    if e.status_code not in RETRIABLE_STATUS:
                        raise
                except (urlerror.URLError, socket.timeout, TimeoutError, OSError) as e:
                    last_exc = e

                delay = self.backoff_base * (2 ** attempt) + random.uniform(0.0, 0.2)
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AgentResult:
        t0 = time.time()
        temperature = self.temperature if temperature is None else float(temperature)
        max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(prompt),
            "max_tokens": max_tokens,
        }
        if isinstance(self.system_prompt, str) and self.system_prompt.strip() and "claude" in self.model.lower():
            payload["system"] = self.system_prompt.replace("\r\n", "\n")
        if self._send_temperature and temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "python-httpx/0.27.2",
            "Connection": "keep-alive",
        }

        try:
            msgs = payload["messages"]
            preview = []
            for m in msgs[:2]:
                c = m.get("content", "")
                if not isinstance(c, str):
                    c = str(c)
                preview.append({"role": m.get("role"), "content_head": c[:120], "len": len(c)})
        except Exception:
            preview = []
        temp_dbg = payload.get("temperature", "<omitted>")
        print(
            f"[DEBUG] model={self.model} temp={temp_dbg} max_tokens={max_tokens} msgs={len(payload.get('messages', []))}"
        )
        print(f"[DEBUG] messages_preview_first2={preview}")

        data = await self._post_with_retry(payload, headers)

        text = ""
        finish_reason = "stop"
        if isinstance(data.get("choices"), list) and data["choices"]:
            ch = data["choices"][0]
            msg = ch.get("message") or {}
            text = (msg.get("content") or "").strip()
            finish_reason = ch.get("finish_reason") or "stop"

        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens") or approx_tokens(str(payload["messages"])))
        ct = int(usage.get("completion_tokens") or approx_tokens(text))

        return AgentResult(
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=cost_usd(pt, ct, self.ppk, self.cpk),
            latency_s=time.time() - t0,
            finish_reason=finish_reason,
        )
