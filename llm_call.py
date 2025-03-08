# llm_call.py

import asyncio
import time
from typing import Optional

from google import genai
from google.genai import types
from google.genai.errors import ServerError

from clean_llm_output import clean_llm_output
from config import Config


async def llm_call(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Asynchronously gets a text response from a specified model based on the provided prompt and optional system prompt.

    Parameters:
        prompt (str): The text prompt to send to the model.
        system_prompt (Optional[str]): Additional system instructions (default is None).

    Returns:
        response_text (str): The generated text response.
    """

    GOOGLE_MODELS = {"gemini-2.0-flash", "gemini-2.0-flash-thinking-exp"}

    if Config.DEFAULT_MODEL_NAME in GOOGLE_MODELS:
        client = genai.Client(api_key=Config.GEMINI_API_KEY)

    # We'll attempt up to MAX_RETRIES times, using exponential backoff
    for attempt in range(Config.MAX_RETRIES):
        try:
            if Config.DEFAULT_MODEL_NAME in GOOGLE_MODELS:
                config_obj = (
                    types.GenerateContentConfig(system_instruction=system_prompt)
                    if system_prompt
                    else None
                )
                # Wrap the blocking call in asyncio.to_thread so as not to block the event loop
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=Config.DEFAULT_MODEL_NAME,
                    contents=[prompt],
                    config=config_obj
                )
                output = clean_llm_output(response.text)
                return output
            else:
                raise ValueError(f"Unsupported model: {Config.DEFAULT_MODEL_NAME}")
        except ServerError as e:
            if "503" in str(e) and "UNAVAILABLE" in str(e):
                if attempt < Config.MAX_RETRIES - 1:
                    sleep_time = Config.BACKOFF_FACTOR * (2 ** attempt)
                    if Config.PRINT_SERVER_ERROR:
                        print(
                            f"Server overloaded (503). "
                            f"Retrying in {sleep_time:.1f} seconds... "
                            f"(Attempt {attempt+1} of {Config.MAX_RETRIES})"
                        )
                    await asyncio.sleep(sleep_time)
                else:
                    raise
            else:
                raise