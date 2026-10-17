import os
import httpx
from langchain_mistralai import ChatMistralAI
from langchain_core.rate_limiters import InMemoryRateLimiter

# The video uses "mistral-small-latest", but Mistral's current free tier allows 0 req/min on it
# (HTTP 429). "ministral-14b-latest" is free at 30 req/min. Override with MISTRAL_MODEL in .env.
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "ministral-14b-latest")

# The free tier allows 30 requests/minute for this model. Summaries + extractions fire many requests
# in parallel, so every call goes through one shared limiter to avoid HTTP 429 errors.
RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=float(os.getenv("MISTRAL_RPS", "0.45")),
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

_llm = None


def get_llm():

    global _llm

    if _llm is None:
        if not os.getenv("MISTRAL_API_KEY"):
            raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
        _llm = ChatMistralAI(
            model=MISTRAL_MODEL,
            temperature=0.2,
            max_retries=3,
            rate_limiter=RATE_LIMITER,
        ).with_retry(
            # langchain-mistralai doesn't retry HTTP errors like 429 itself — back off and retry.
            retry_if_exception_type=(httpx.HTTPStatusError,),
            wait_exponential_jitter=True,
            stop_after_attempt=6,
        )
    return _llm
