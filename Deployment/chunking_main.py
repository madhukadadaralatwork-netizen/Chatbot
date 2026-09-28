"""
chunking_main.py

Core config + Gemini client setup, shared by generate.py. The old chunking /
retrieval / regex-based answer logic that used to live here has been
replaced by loader.py + retrieval_v2.py + generate.py + answer_helpers.py +
prompt.py, and removed from this file so there's exactly one place each
piece of logic lives.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from functools import lru_cache

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "all-miniLM-L6-v2")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-flash-lite-latest")
TOP_K = int(os.getenv("TOP_K", 5))
NOT_FOUND_MESSAGE = os.getenv(
    "NOT_FOUND_MESSAGE",
    "I could not find this information in the provided technical documentation.",
)


@lru_cache(maxsize=1)
def _get_genai_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to a .env file in the project root.")
    return genai.Client(api_key=api_key)


@retry(
    retry=retry_if_exception_type(genai_errors.ServerError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _generate_content_with_retry(prompt):
    client = _get_genai_client()
    return client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)