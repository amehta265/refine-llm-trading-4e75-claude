import os
import re
import json
import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

"""
    REVIEW (Note):
        The constrained output i.e. a structured JSON object with a decision and reasoning field makes parsing responses deterministic.
        This is shown in the "FLAG-Trader" core paper from the literature review.
"""

""" 
    REVIEW (Improvement):
        The prompt does not ask for a confidence score i.e. how confident are you in your decision to BUY/SELL/HOLD.
        This is important to determine position sizing - a fundamental concept in trading that seems to be looked over in this implementation.
        There is a school of thought (https://www.heygotrade.com/en/blog/position-sizing-explained/) that position sizing is more important than entries (when you choose to buy or sell).
        A real trader (even a simple one like me) chooses the value I wish to invest based on the risk I am willing to take and my conviction about the stocks future. 
        Here, we go with a binary all-in or all-out approach, meaning the model treats a "slightly bullish" signal the same as a "strongly bullish" one.
        - Adding a confidence score would enable position sizing where a high confidence -> full position and low confidence -> partial position in the stock.
        - Such `Risk-Aware Position Sizing` is what I will implement.
"""

"""
    REVIEW (Improvement):
        This prompt excludes market sentiment, news, and macroeconomic data and only considers price and indicators. This is explained by claude 
        in `execution_claude_transcript.jsonl` that "I'm keeping the scope focused on price action and technical indicators to isolate the frequency variable cleanly—adding news or sentiment would muddy the results.."
        I feel like this context is essential. My opinion is that trades are made as a result of decisions in the ecologic space where all these factors are important. 
        Research on this matter can be seen in this paper: Enhancing Trading Performance Through Sentiment Analysis with Large Language Models: Evidence from the S&P 500 (https://arxiv.org/html/2507.09739v1)
        - Furthermore, https://github.com/amehta265/refine-llm-trading-4e75-claude/blob/main/datasets/dataset_search_results.md shows the prevalence of such data making it easier to integrate.
"""

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

    """
        REVIEW (Note):
            I think the explicit context being added below forces the LLM to reason explicitly about the decision frequency (day/month/week), 
            which is the central variable under study.
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

"""
REVIEW (Note):
    No layered memory context is passed to the function below across decisions. This probably does not matter much
    for day trading but for longer time horizons the lack of "history" means the model cannot learn from its prior output.
"""
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
            """
            REVIEW (Improvement):
                A single-agent architecture is used below on line 121 meaning that each trading decision is produced by exactly one LLM invocation.
                This design is appropriate as it isolates the independent variable (trading frequency) being tested.
                But its important to note that it limits the model's ability to incorporate diverse analytical perspectives as shown by the
                TradingAgents (Xiao et al., 2025, https://tradingagents-ai.github.io) core paper in the literature review. They mimic a trading firm by
                creating specialized agents with finance related roles (analysts, research, risk-management, fund manager, trader).
                - A multi-agent framework where we could chain multiple calls across these specialized roles could improve trading outcomes.
            """
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=150,  

                #  REVIEW (Improvement):
                # I feel like the max token limit (which dictates output size) is too small. The agent is expected to return a decision (Buy/Sell/Hold) AND a brief explanation
                # in about 1-2 sentences. This can especially impact monthly frequency trades where a greater reasoning to evaluate trends is required for the longer time horizon.
                #   Maybe increase this to 300? It is at least worth asking whether this might improve decision quality.
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
            """
            Review (Note):
                The conditionals below can misclassify text especially when the temperature, which controls randomness and creativity of response, is increased.
                In the current state this is likely never triggered as the model is set to have a completely deterministic output (temperature=0.0), but if this increases and the model
                chooses to ignore the prompt's JSON structure with negative, normative responses like "Do not buy this stock". 
                This will be classified as a "BUY" which is incorrect.
                I'd improve the logic below by using regex to match "decision": "(BUY|SELL|HOLD)"
                I have updated this flow.
            """
            json_match = re.search(r'"decision"\s*:\s*"(BUY|SELL|HOLD)"', content, re.IGNORECASE)
            if json_match:
                decision = json_match.group(1).upper()
            else:
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
            """
            REVIEW (Note):
                The exponential-backoff API retry logic is very sound. Increasing delay time on successive failures means you wouldn't overwhelm the server!
            """
            logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
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
