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
import os
import re

from pydantic import ValidationError

from adapt import config
from adapt.extraction.grounding import ground
from adapt.extraction.schema import Extraction

# Keep multi-GB model files on D: (C: has little free space). Must be set before transformers imports.
os.environ.setdefault("HF_HOME", str(config.DATA_DIR / "hf-cache"))

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM = """You turn a person's description of a decision into JSON. Output ONLY a JSON object, on ONE line, no indentation:
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

# v2: rules aimed at errors seen on the VALIDATION split (made-up limits, wrong or invented
# priorities, two names for one attribute, dropped yes/no facts). Example still about gyms,
# which is not a benchmark domain.
SYSTEM_V2 = SYSTEM + """
- NEVER add a constraint the text doesn't state. A preference ("lower fee is better") is a criterion, not a constraint.
- priority comes ONLY from words about importance in the same sentence. If none, priority is null.
- If different words name the same attribute (e.g. "fee" and "charge"), pick ONE name and use it everywhere.
- Include every yes/no fact about an option: "X has a pool" -> has_pool: true, "X has no pool" -> has_pool: false."""

EXAMPLE_TEXT_V2 = (
    "Help me pick a gym. Gym A: a monthly fee of ₹1,800 and distance 2 km. Gym A has no pool. "
    "With Gym B I'd get a charge of ₹2,500 a month, distance of 0.5 km and a class rating of 8/10. Gym B has a pool. "
    "Lower monthly fee is better, and it's my top priority. I want less distance, and it only matters a little. "
    "Higher class rating is better. It must have a pool."
)
EXAMPLE_JSON_V2 = {
    "options": [
        {"label": "Gym A", "attributes": {"monthly_fee": {"value": 1800, "unit": "INR"}, "distance": {"value": 2, "unit": "km"},
                                          "has_pool": {"value": False, "unit": None}}},
        {"label": "Gym B", "attributes": {"monthly_fee": {"value": 2500, "unit": "INR"}, "distance": {"value": 0.5, "unit": "km"},
                                          "class_rating": {"value": 8, "unit": None}, "has_pool": {"value": True, "unit": None}}},
    ],
    "criteria": [
        {"attribute": "monthly_fee", "direction": "minimize", "priority": "high"},
        {"attribute": "distance", "direction": "minimize", "priority": "low"},
        {"attribute": "class_rating", "direction": "maximize", "priority": None},
    ],
    "constraints": [{"attribute": "has_pool", "operator": "==", "value": True, "unit": None}],
}

PROMPTS = {
    "v1": (SYSTEM, EXAMPLE_TEXT, EXAMPLE_JSON),
    "v2": (SYSTEM_V2, EXAMPLE_TEXT_V2, EXAMPLE_JSON_V2),
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
    """Call with one text, or use .batch() for many: on this GPU one text at a time is
    CPU-bound (~8 tokens/s, GPU ~23% busy), and batching amortises that overhead."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        prompt: str = "v2",
        device: str | None = None,
        max_new_tokens: int = 1200,  # 6 options can need ~800 tokens; 700 truncated them
        grounding: bool = True,  # drop numbers that are not in the text (see grounding.py)
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.prompt_version = prompt
        self.system, self.example_text, self.example_json = PROMPTS[prompt]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.grounding = grounding
        self.dropped = 0  # ungrounded numbers removed so far
        self._torch = torch

    @property
    def version(self) -> str:
        return f"{self.model_id.split('/')[-1]}-prompt{self.prompt_version}{'-grounded' if self.grounding else ''}"

    def _messages(self, text: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.example_text},
            {"role": "assistant", "content": json.dumps(self.example_json)},
            {"role": "user", "content": text},
        ]

    def _generate(self, conversations: list[list[dict]]) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in conversations
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        n = inputs["input_ids"].shape[1]
        return [self.tokenizer.decode(row[n:], skip_special_tokens=True) for row in out]

    def _retry(self, conversation: list[dict], raw: str, err: Exception) -> Extraction:
        log.info("retrying after invalid output: %s", str(err)[:200])
        conversation = conversation + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"That was not valid: {str(err)[:300]}. Reply with only the corrected JSON."},
        ]
        return parse_json(self._generate([conversation])[0])

    def batch(self, texts: list[str]) -> list[Extraction | Exception]:
        """Extract many texts at once. Failures come back as the exception, not raised."""
        convs = [self._messages(t) for t in texts]
        out: list[Extraction | Exception] = []
        for conv, raw in zip(convs, self._generate(convs)):
            try:
                out.append(parse_json(raw))
            except (ValueError, ValidationError) as e:
                try:
                    out.append(self._retry(conv, raw, e))
                except (ValueError, ValidationError) as e2:
                    out.append(e2)
        if self.grounding:
            for i, (text, r) in enumerate(zip(texts, out)):
                if isinstance(r, Extraction):
                    out[i], n = ground(r, text)
                    self.dropped += n
        return out

    def __call__(self, text: str) -> Extraction:
        result = self.batch([text])[0]
        if isinstance(result, Exception):
            raise result
        return result

