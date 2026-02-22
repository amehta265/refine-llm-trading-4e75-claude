"""
LLM Trading Agent using OpenAI GPT-4.1-mini.
Makes BUY/HOLD/SELL decisions based on price data and technical indicators.
"""

import os
import json
import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You are a stock trading analyst. You will be given recent price data and technical indicators for a stock. Your task is to make a trading decision.

Rules:
- You can only output one of: BUY, HOLD, or SELL
- BUY: Enter or maintain a long position (you believe the stock will go up)
- HOLD: Maintain current position (no strong signal)
- SELL: Exit long position or stay out (you believe the stock will go down or sideways)
- Base your decision on the price trends, technical indicators, and overall market conditions shown in the data
- Consider the SMA crossovers, RSI levels, and recent return momentum

Respond with ONLY a JSON object in this exact format:
{"decision": "BUY" or "HOLD" or "SELL", "reasoning": "brief 1-sentence explanation"}"""


def build_decision_prompt(ticker: str, frequency: str, price_history: str,
                          current_position: str) -> str:
    """Build the user prompt for the LLM trading decision.

    Args:
        ticker: Stock ticker symbol
        frequency: Trading frequency (daily/weekly/monthly)
        price_history: Formatted price history string
        current_position: Current position ('LONG' or 'OUT')

    Returns:
        Formatted prompt string
    """
    freq_context = {
        "daily": "You are making a DAILY trading decision. Consider short-term price momentum and daily patterns.",
        "weekly": "You are making a WEEKLY trading decision. Consider medium-term trends and weekly patterns. Your next decision will be in one week.",
        "monthly": "You are making a MONTHLY trading decision. Consider long-term trends and macro patterns. Your next decision will be in one month.",
    }

    prompt = f"""Stock: {ticker}
Decision Frequency: {frequency.upper()}
Current Position: {current_position}

{freq_context[frequency]}

Recent Price Data:
{price_history}

Based on this data, should I BUY, HOLD, or SELL {ticker}?
Respond with ONLY a JSON object: {{"decision": "BUY/HOLD/SELL", "reasoning": "..."}}"""

    return prompt


def get_llm_decision(ticker: str, frequency: str, price_history: str,
                     current_position: str, model: str = "gpt-4.1-mini",
                     temperature: float = 0.0, max_retries: int = 3) -> dict:
    """Query the LLM for a trading decision.

    Args:
        ticker: Stock ticker
        frequency: Trading frequency
        price_history: Formatted price data
        current_position: Current position state
        model: OpenAI model name
        temperature: Sampling temperature
        max_retries: Number of retries on failure

    Returns:
        Dict with 'decision' (BUY/HOLD/SELL) and 'reasoning'
    """
    user_prompt = build_decision_prompt(ticker, frequency, price_history, current_position)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=150,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            decision = result.get("decision", "HOLD").upper()

            if decision not in ("BUY", "HOLD", "SELL"):
                decision = "HOLD"

            return {
                "decision": decision,
                "reasoning": result.get("reasoning", ""),
                "raw_response": content,
                "model": model,
                "temperature": temperature,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
            }

        except json.JSONDecodeError:
            # Try to extract decision from free-form text
            content_upper = content.upper()
            if "BUY" in content_upper:
                decision = "BUY"
            elif "SELL" in content_upper:
                decision = "SELL"
            else:
                decision = "HOLD"

            return {
                "decision": decision,
                "reasoning": content[:200],
                "raw_response": content,
                "model": model,
                "temperature": temperature,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
            }

        except Exception as e:
            logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"All retries exhausted for {ticker} {frequency}")
                return {
                    "decision": "HOLD",
                    "reasoning": f"API error: {str(e)}",
                    "raw_response": "",
                    "model": model,
                    "temperature": temperature,
                    "tokens_used": 0,
                }
