"""Local pretrained extractor: a small instruction-tuned model writes the Extraction JSON.

Runs fully on this machine (default Qwen2.5-1.5B-Instruct, Apache-2.0). Output is
validated against the Extraction schema; if it isn't valid JSON of the right shape,
the model gets one retry with the error. Anything still invalid raises, and the
evaluation counts it as a failed extraction rather than guessing.

The worked example in the prompt is about gyms, a domain that isn't in the
benchmark, so the prompt doesn't leak benchmark answers.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from adapt.extraction.schema import Extraction

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM = """You turn a person's description of a decision into JSON. Output ONLY a JSON object:
{
  "options": [{"label": str, "attributes": {"<attribute>": {"value": number|true|false, "unit": str|null}}}],
  "criteria": [{"attribute": str, "direction": "maximize"|"minimize", "priority": "high"|"medium"|"low"|null}],
  "constraints": [{"attribute": str, "operator": "<="|">="|"=="|"!=", "value": number|true|false, "unit": str|null}]
}
Rules:
- One option per alternative the person names, with the label exactly as written.
- Use short snake_case attribute names, and the SAME name everywhere an attribute appears.
- Only include values the text states. Never invent values. Write numbers without commas or currency signs.
- criteria = things to trade off (what is better: more or less). priority: "high" if it matters most / top
  priority, "medium" if fairly important, "low" if it only matters a little, null if not said.
- constraints = hard limits that must hold ("can't be more than", "at least", "must be", "can't").
- Yes/no facts (e.g. "is open source", "requires relocation") are true/false attribute values."""

EXAMPLE_TEXT = (
    "I'm picking a gym. Gym A: a monthly fee of ₹1,800 and distance 2 km. Gym B has monthly fee ₹2,500 and "
    "distance 0.5 km. Gym B has a pool. Lower monthly fee is better, and it's my top priority. "
    "I want less distance, and it only matters a little. The monthly fee can't be more than ₹3,000."
)
EXAMPLE_JSON = {
    "options": [
        {"label": "Gym A", "attributes": {"monthly_fee": {"value": 1800, "unit": "INR"}, "distance": {"value": 2, "unit": "km"}}},
        {"label": "Gym B", "attributes": {"monthly_fee": {"value": 2500, "unit": "INR"}, "distance": {"value": 0.5, "unit": "km"},
                                          "has_pool": {"value": True, "unit": None}}},
    ],
    "criteria": [
        {"attribute": "monthly_fee", "direction": "minimize", "priority": "high"},
        {"attribute": "distance", "direction": "minimize", "priority": "low"},
    ],
    "constraints": [{"attribute": "monthly_fee", "operator": "<=", "value": 3000, "unit": "INR"}],
}


def parse_json(text: str) -> Extraction:
    """First balanced {...} block in the model output, validated against the schema."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in output")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            block = re.sub(r",\s*([}\]])", r"\1", text[start : i + 1])  # tolerate trailing commas
            return Extraction.model_validate(json.loads(block))
    raise ValueError("unbalanced JSON in output")


class LLMExtractor:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None, max_new_tokens: int = 700):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def _generate(self, messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def __call__(self, text: str) -> Extraction:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": EXAMPLE_TEXT},
            {"role": "assistant", "content": json.dumps(EXAMPLE_JSON)},
            {"role": "user", "content": text},
        ]
        raw = self._generate(messages)
        try:
            return parse_json(raw)
        except (ValueError, ValidationError) as e:
            log.info("retrying after invalid output: %s", str(e)[:200])
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That was not valid: {str(e)[:300]}. Reply with only the corrected JSON."},
            ]
            return parse_json(self._generate(messages))
