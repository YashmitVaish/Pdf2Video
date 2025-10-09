# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 15:50:49 2025

@author: ADMIN
"""

# api_cost_tracker.py
class OpenAICostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_chars = 0
        self.breakdown = []

        self.chat_pricing = {
            "gpt-4o": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},
            "gpt-4o-mini": {"prompt": 0.00025 / 1000, "completion": 0.00075 / 1000},
            "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000}
        }
        self.tts_pricing = {
            "tts-1": 0.015 / 1000,
            "tts-1-hd": 0.030 / 1000
        }
        self.chat_pricing["gemini-2.5-flash-lite"] = {
        # Google Gemini Flash-Lite: $0.10 input, $0.40 output per 1M tokens
        "prompt": 0.10 / 1_000_000,      # = 0.00000010
        "completion": 0.40 / 1_000_000   # = 0.00000040
    }

    def track_chat_langchain(self, ai_message, model_name):
        usage = ai_message.additional_kwargs.get("usage_metadata", {})
        prompt_tokens = usage.get("prompt_token_count", 0)
        completion_tokens = usage.get("candidates_token_count", 0)

        prompt_cost = prompt_tokens * self.chat_pricing.get(model_name, {}).get("prompt", 0)
        completion_cost = completion_tokens * self.chat_pricing.get(model_name, {}).get("completion", 0)
        cost = prompt_cost + completion_cost

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += cost

        self.breakdown.append({
            "type": "chat",
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost
        })

        return cost

    def track_chat(self, response, model_name=None):
    # Handle OpenAI-like responses
        if hasattr(response, "usage"):
            usage = response.usage
            model = getattr(response, "model", model_name or "unknown")
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

        # Handle LangChain AIMessage (Gemini)
        elif hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            model = model_name or "gemini"
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

        else:
            return 0.0  # nothing to track

        # Calculate cost
        pricing = self.chat_pricing.get(model, {})
        prompt_cost = prompt_tokens * pricing.get("prompt", 0)
        completion_cost = completion_tokens * pricing.get("completion", 0)
        cost = prompt_cost + completion_cost

        # Update totals
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += cost

        self.breakdown.append({
            "type": "chat",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost
        })

        return cost
    def track_tts(self, model, text):
        num_chars = len(text)
        rate = self.tts_pricing.get(model, 0)
        cost = num_chars * rate
        self.total_chars += num_chars
        self.total_cost += cost

        self.breakdown.append({
            "type": "tts",
            "model": model,
            "characters": num_chars,
            "cost": cost
        })
        return cost

    def summary(self):
        print("\n📊 API Cost Summary:")
        print(f"🧾 Total Prompt Tokens: {self.total_prompt_tokens}")
        print(f"🧾 Total Completion Tokens: {self.total_completion_tokens}")
        print(f"🔤 Total TTS Characters: {self.total_chars}")
        print(f"💰 Total Estimated Cost: ${self.total_cost:.6f}")
        print("\n🔍 Breakdown:")
        for i, item in enumerate(self.breakdown, 1):
            if item["type"] == "chat":
                print(f"{i}. 🗨️ Chat ({item['model']}) - Prompt: {item['prompt_tokens']}, Completion: {item['completion_tokens']}, Cost: ${item['cost']:.6f}")
            elif item["type"] == "tts":
                print(f"{i}. 🎧 TTS ({item['model']}) - Chars: {item['characters']}, Cost: ${item['cost']:.6f}")


# Global singleton instance
tracker = OpenAICostTracker()
