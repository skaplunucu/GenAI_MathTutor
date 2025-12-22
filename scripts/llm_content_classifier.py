#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re
from pathlib import Path


@dataclass
class ContentType:
    type: str
    confidence: float
    reasoning: str = ""


class LLMContentClassifier:
    def __init__(
        self,
        model_path: str = "/home/sskaplun/study/genAI/kaggle/models/gemma-2-9b-it",
        backend: str = "auto",  # auto, transformers, or llamacpp
        device: str = "cuda",
        load_in_4bit: bool = True,
        verbose: bool = False
    ):
        self.model_path = Path(model_path)
        self.backend = backend
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.verbose = verbose

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if self.backend == "auto":
            try:
                self._load_transformers()
                self.backend = "transformers"
                return
            except ImportError:
                print("Transformers not available, trying llama.cpp...")
                self._load_llamacpp()
                self.backend = "llamacpp"
        elif self.backend == "transformers":
            self._load_transformers()
        elif self.backend == "llamacpp":
            self._load_llamacpp()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _load_transformers(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers accelerate")

        print(f"Loading model from {self.model_path} using transformers...")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        if self.load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print(" Model loaded with 4-bit quantization")

    def _load_llamacpp(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")

        print(f"Loading model from {self.model_path} using llama.cpp...")

        gguf_files = list(self.model_path.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF files found in {self.model_path}")

        model_file = gguf_files[0]
        print(f"Using: {model_file.name}")

        n_gpu_layers = -1 if self.device == "cuda" else 0

        self.model = Llama(
            model_path=str(model_file),
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=n_gpu_layers,
            verbose=self.verbose
        )
        print(" Model loaded")

    def _build_classification_prompt(self, texts: List[str]) -> str:
        """Build prompt for batch classification."""

        prompt = """You are classifying Ukrainian mathematics textbook content. For each text excerpt, determine its content type.

**Content Types:**

1. **definition** - Formal mathematical definitions
   - Keywords: Означення, Визначення, називається, називають
   - Example: "Означення. Квадратним рівнянням називається рівняння виду ax² + bx + c = 0"

2. **theorem** - Mathematical theorems, properties, lemmas
   - Keywords: Теорема, Властивість, Властивості, Лема, Наслідок
   - Example: "Теорема 1. Сума кутів трикутника дорівнює 180°"

3. **proof** - Mathematical proofs and detailed solutions
   - Keywords: Доведення, Розв'язання, Розв'язок, Спочатку, Отже, Таким чином
   - Example: "Доведення. Розглянемо трикутник ABC. Проведемо через вершину B..."

4. **example** - Worked examples demonstrating concepts
   - Keywords: Приклад, Розглянемо, Нехай задано
   - Example: "Приклад 1. Розглянемо рівняння x² - 5x + 6 = 0"

5. **problem** - Practice exercises for students
   - Numbered format: "9.8.", "10.15•", etc.
   - Imperative commands: Знайдіть, Обчисліть, Доведіть що, Розв'яжіть
   - Example: "9.8. Знайдіть площу трикутника зі сторонами 3, 4 і 5"

6. **formula** - Standalone mathematical formulas
   - Format: Variable = expression
   - Example: "S = πr²", "V = 4/3πr³"

7. **explanation** - General explanatory text, descriptions, context
   - Default category for text that doesn't fit above categories

**Your Task:**
Classify each text below. Return ONLY a JSON array:
[
  {"index": 0, "type": "definition", "confidence": 0.95},
  {"index": 1, "type": "problem", "confidence": 0.9},
  ...
]

**Texts to classify:**

"""

        for i, text in enumerate(texts):
            text_preview = text[:200] if len(text) > 200 else text
            prompt += f"\n[{i}] {text_preview}\n"

        prompt += "\n\nJSON array:"

        return prompt

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 20,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> List[ContentType]:
        if not texts:
            return []

        all_results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = i // batch_size + 1

            # Show progress for large jobs (every 10 batches)
            if total_batches > 10 and (batch_num % 10 == 0 or batch_num == 1 or batch_num == total_batches):
                progress_pct = (batch_num / total_batches) * 100
                print(f"  Batch {batch_num}/{total_batches} ({progress_pct:.1f}%) - {len(all_results)}/{len(texts)} texts classified")
            elif self.verbose:
                print(f"Processing batch {batch_num}/{total_batches}")

            results = self._classify_batch_single(batch, max_tokens, temperature)
            all_results.extend(results)

        return all_results

    def _classify_batch_single(
        self,
        texts: List[str],
        max_tokens: int,
        temperature: float
    ) -> List[ContentType]:
        prompt = self._build_classification_prompt(texts)

        if self.backend == "transformers":
            response_text = self._generate_transformers(prompt, max_tokens, temperature)
        else:  # llamacpp
            response_text = self._generate_llamacpp(prompt, max_tokens, temperature)

        if self.verbose:
            print(f"LLM Response:\n{response_text}\n")

        # Parse response
        classifications = self._parse_response(response_text, len(texts))

        return classifications

    def _generate_transformers(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import torch

        # Format with chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        return response

    def _generate_llamacpp(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "\n\n\n"],
            echo=False
        )

        return response['choices'][0]['text']

    def _parse_response(self, response_text: str, expected_count: int) -> List[ContentType]:
        json_match = re.search(r'\[[\s\S]*?\]', response_text)

        if not json_match:
            if self.verbose:
                print("Warning: Could not parse LLM response as JSON")
            # Return default classifications
            return [ContentType(type="explanation", confidence=0.1, reasoning="parse_error")
                    for _ in range(expected_count)]

        try:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)

            # Convert to ContentType objects
            results = []
            for item in parsed:
                results.append(ContentType(
                    type=item.get('type', 'explanation'),
                    confidence=float(item.get('confidence', 0.5)),
                    reasoning=item.get('reasoning', '')
                ))

            # Fill in missing indices
            while len(results) < expected_count:
                results.append(ContentType(type="explanation", confidence=0.1, reasoning="missing"))

            return results[:expected_count]

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Warning: JSON decode error: {e}")
            return [ContentType(type="explanation", confidence=0.1, reasoning="json_error")
                    for _ in range(expected_count)]

    def get_statistics(self, classifications: List[ContentType]) -> Dict[str, Any]:
        """Get statistics for classified content."""
        from collections import Counter

        type_counts = Counter(c.type for c in classifications)
        avg_confidence = sum(c.confidence for c in classifications) / len(classifications)

        return {
            'total': len(classifications),
            'types': dict(type_counts),
            'average_confidence': avg_confidence,
            'high_confidence': sum(1 for c in classifications if c.confidence >= 0.8),
            'medium_confidence': sum(1 for c in classifications if 0.5 <= c.confidence < 0.8),
            'low_confidence': sum(1 for c in classifications if c.confidence < 0.5),
            'type_percentages': {
                t: count / len(classifications) * 100
                for t, count in type_counts.items()
            }
        }