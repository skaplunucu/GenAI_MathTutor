#!/usr/bin/env python3
"""
VLM Image Analyzer for Math Textbooks

Uses local Vision-Language Model (LLaVA) to analyze and describe images from
Ukrainian math textbooks. Extracts visual content, formulas, and diagrams.

Completely local - no API calls!
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from PIL import Image
import json


@dataclass
class ImageAnalysis:
    """Analysis result for a single image."""
    image_path: str
    image_type: str  # diagram, graph, formula, table, illustration
    description_uk: str  # Ukrainian description
    description_en: str  # English description (fallback)
    mathematical_concepts: List[str]
    extracted_text: List[str]  # OCR'd text/formulas
    confidence: float
    related_content_type: str  # theorem, definition, problem, etc.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VLMImageAnalyzer:
    """Analyze math textbook images using local VLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",  # Default to Qwen2-VL
        device: str = "cuda",
        load_in_4bit: bool = True,
        verbose: bool = False
    ):
        """
        Initialize VLM analyzer.

        Args:
            model_name: HuggingFace model ID or local path
            device: cuda or cpu
            load_in_4bit: Use 4-bit quantization to save VRAM
            verbose: Print debug info
        """
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.verbose = verbose

        self.model = None
        self.processor = None

        # Detect model type
        self.model_type = self._detect_model_type()
        self._load_model()

    def _detect_model_type(self) -> str:
        """Detect which VLM architecture to use."""
        if "qwen" in self.model_name.lower():
            return "qwen2vl"
        elif "llava" in self.model_name.lower():
            return "llava"
        else:
            return "llava"  # Default

    def _load_model(self):
        """Load the VLM model."""
        if self.model_type == "qwen2vl":
            self._load_qwen2vl()
        else:
            self._load_llava()

    def _load_qwen2vl(self):
        """Load Qwen2-VL model."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
        except ImportError:
            print("ERROR: transformers not installed!")
            print("Install with: pip install transformers accelerate pillow qwen-vl-utils")
            raise

        print(f"Loading Qwen2-VL from {self.model_name}...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Load model with quantization if requested
        if self.load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                print(" Qwen2-VL loaded with 4-bit quantization")

            except ImportError:
                print("Warning: bitsandbytes not available, loading without quantization")
                self._load_qwen2vl_no_quant()
        else:
            self._load_qwen2vl_no_quant()

    def _load_qwen2vl_no_quant(self):
        """Load Qwen2-VL without quantization."""
        from transformers import Qwen2VLForConditionalGeneration
        import torch

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print(" Qwen2-VL loaded")

    def _load_llava(self):
        """Load LLaVA model."""
        try:
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            import torch
        except ImportError:
            print("ERROR: transformers not installed!")
            print("Install with: pip install transformers accelerate pillow")
            raise

        print(f"Loading LLaVA from {self.model_name}...")

        # Load processor
        self.processor = LlavaProcessor.from_pretrained(self.model_name)

        # Load model with quantization if requested
        if self.load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                print(" LLaVA loaded with 4-bit quantization")

            except ImportError:
                print("Warning: bitsandbytes not available, loading without quantization")
                self._load_llava_no_quant()
        else:
            self._load_llava_no_quant()

    def _load_llava_no_quant(self):
        """Load LLaVA without quantization."""
        from transformers import LlavaForConditionalGeneration
        import torch

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print(" LLaVA loaded")

    def _build_analysis_prompt(self) -> str:
        """Build prompt for image analysis."""

        prompt = """Analyze this mathematics textbook image and provide ONLY a JSON response.

Example response format:
{"image_type": "diagram", "description_uk": "Трикутник ABC", "description_en": "Triangle ABC", "mathematical_concepts": ["трикутник"], "extracted_text": ["A", "B", "C"], "confidence": 0.9, "related_content_type": "definition"}

Now analyze the actual image:
- What shapes do you see? (circles, triangles, squares, etc.)
- What text or labels are visible?
- What mathematical concept is shown?

Provide your answer as JSON with these keys:
image_type (diagram/graph/formula/table/illustration)
description_uk (Ukrainian description)
description_en (English description)
mathematical_concepts (list of terms)
extracted_text (list of visible text)
confidence (0.0 to 1.0)
related_content_type (theorem/definition/problem/example/explanation)

JSON response:"""

        return prompt

    def analyze_image(
        self,
        image_path: Path,
        max_tokens: int = 256,
        temperature: float = 0.3
    ) -> ImageAnalysis:
        """
        Analyze a single image.

        Args:
            image_path: Path to image file
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature

        Returns:
            ImageAnalysis object
        """
        if self.model_type == "qwen2vl":
            return self._analyze_image_qwen2vl(image_path, max_tokens, temperature)
        else:
            return self._analyze_image_llava(image_path, max_tokens, temperature)

    def _analyze_image_qwen2vl(
        self,
        image_path: Path,
        max_tokens: int,
        temperature: float
    ) -> ImageAnalysis:
        """Analyze image with Qwen2-VL."""
        import torch

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self._create_fallback_analysis(image_path, "load_error")

        # Build prompt
        prompt_text = self._build_analysis_prompt()

        # Format for Qwen2-VL (conversation format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )

        # Decode
        response = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Extract assistant response (after the last prompt)
        if "assistant\n" in response:
            response = response.split("assistant\n")[-1].strip()

        if self.verbose:
            print(f"VLM response for {image_path.name}:\n{response}\n")

        # Parse response
        analysis = self._parse_response(response, image_path)

        return analysis

    def _analyze_image_llava(
        self,
        image_path: Path,
        max_tokens: int,
        temperature: float
    ) -> ImageAnalysis:
        """Analyze image with LLaVA."""
        import torch

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self._create_fallback_analysis(image_path, "load_error")

        # Build prompt
        prompt_text = self._build_analysis_prompt()

        # Format for LLaVA 1.5 (simple prompt with USER/ASSISTANT tags)
        prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"

        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        # Decode (skip the prompt part)
        full_response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response (after "ASSISTANT:")
        if "ASSISTANT:" in full_response:
            response = full_response.split("ASSISTANT:")[-1].strip()
        else:
            response = full_response

        if self.verbose:
            print(f"VLM response for {image_path.name}:\n{response}\n")

        # Parse response
        analysis = self._parse_response(response, image_path)

        return analysis

    def _parse_response(self, response: str, image_path: Path) -> ImageAnalysis:
        """Parse VLM response into ImageAnalysis."""

        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response)

        if not json_match:
            if self.verbose:
                print(f"Warning: Could not parse VLM response as JSON for {image_path.name}")
            return self._create_fallback_analysis(image_path, "parse_error")

        try:
            # Clean up escaped underscores that the model sometimes generates
            json_str = json_match.group(0).replace('\\_', '_')
            parsed = json.loads(json_str)

            return ImageAnalysis(
                image_path=str(image_path),
                image_type=parsed.get('image_type', 'illustration'),
                description_uk=parsed.get('description_uk', ''),
                description_en=parsed.get('description_en', ''),
                mathematical_concepts=parsed.get('mathematical_concepts', []),
                extracted_text=parsed.get('extracted_text', []),
                confidence=float(parsed.get('confidence', 0.5)),
                related_content_type=parsed.get('related_content_type', 'explanation')
            )

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"JSON decode error for {image_path.name}: {e}")
            return self._create_fallback_analysis(image_path, "json_error")

    def _create_fallback_analysis(self, image_path: Path, reason: str) -> ImageAnalysis:
        """Create fallback analysis when VLM fails."""
        return ImageAnalysis(
            image_path=str(image_path),
            image_type="unknown",
            description_uk="Аналіз зображення недоступний",
            description_en="Image analysis unavailable",
            mathematical_concepts=[],
            extracted_text=[],
            confidence=0.1,
            related_content_type="unknown"
        )

    def analyze_batch(
        self,
        image_paths: List[Path],
        batch_size: int = 1  # VLMs are memory intensive, process one at a time
    ) -> List[ImageAnalysis]:
        """
        Analyze multiple images.

        Args:
            image_paths: List of image paths
            batch_size: Currently only 1 supported (VLMs use lots of VRAM)

        Returns:
            List of ImageAnalysis objects
        """
        results = []

        total = len(image_paths)
        for i, image_path in enumerate(image_paths, 1):
            if self.verbose or i % 5 == 0:
                print(f"Analyzing image {i}/{total}: {image_path.name}")

            try:
                analysis = self.analyze_image(image_path)
                results.append(analysis)
            except Exception as e:
                print(f"Error analyzing {image_path}: {e}")
                results.append(self._create_fallback_analysis(image_path, "error"))

        return results

    def get_statistics(self, analyses: List[ImageAnalysis]) -> Dict[str, Any]:
        """Get statistics for analyzed images."""
        from collections import Counter

        type_counts = Counter(a.image_type for a in analyses)
        content_counts = Counter(a.related_content_type for a in analyses)
        avg_confidence = sum(a.confidence for a in analyses) / len(analyses) if analyses else 0

        # Extract all concepts
        all_concepts = []
        for a in analyses:
            all_concepts.extend(a.mathematical_concepts)
        concept_counts = Counter(all_concepts)

        return {
            'total_images': len(analyses),
            'image_types': dict(type_counts),
            'content_types': dict(content_counts),
            'average_confidence': avg_confidence,
            'high_confidence': sum(1 for a in analyses if a.confidence >= 0.8),
            'low_confidence': sum(1 for a in analyses if a.confidence < 0.5),
            'top_concepts': dict(concept_counts.most_common(10)),
            'images_with_text': sum(1 for a in analyses if a.extracted_text),
            'images_with_concepts': sum(1 for a in analyses if a.mathematical_concepts)
        }


if __name__ == "__main__":
    import sys

    print("="*80)
    print("VLM IMAGE ANALYZER TEST")
    print("="*80)

    # Find test images
    test_images_dir = Path(__file__).parent.parent / 'data' / 'images'

    # Get first few images from any PDF
    image_dirs = list(test_images_dir.glob('*/'))

    if not image_dirs:
        print("No images found in data/images/")
        print("Run the pipeline first to extract images.")
        sys.exit(1)

    # Get first 3 images
    test_images = []
    for img_dir in image_dirs:
        test_images.extend(list(img_dir.glob('*.png'))[:3])
        if len(test_images) >= 3:
            break

    test_images = test_images[:3]

    if not test_images:
        print("No PNG images found")
        sys.exit(1)

    print(f"\nFound {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {img.name}")

    print("\n Initializing VLM analyzer...")
    print("   This will download LLaVA model (~13GB) on first run")
    print("   Subsequent runs will use cached model")

    try:
        analyzer = VLMImageAnalyzer(verbose=True, load_in_4bit=True)
    except Exception as e:
        print(f"\n ERROR loading VLM: {e}")
        print("\nTroubleshooting:")
        print("1. Install: pip install transformers accelerate pillow bitsandbytes")
        print("2. Ensure CUDA is available")
        sys.exit(1)

    print("\n Analyzing images...")
    results = analyzer.analyze_batch(test_images)

    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)

    for i, (img_path, analysis) in enumerate(zip(test_images, results), 1):
        print(f"\n{i}. {img_path.name}")
        print(f"   Type: {analysis.image_type}")
        print(f"   Confidence: {analysis.confidence:.2f}")
        print(f"   Description (UK): {analysis.description_uk[:100]}...")
        print(f"   Concepts: {', '.join(analysis.mathematical_concepts[:3])}")
        if analysis.extracted_text:
            print(f"   Extracted text: {', '.join(analysis.extracted_text[:5])}")

    # Statistics
    stats = analyzer.get_statistics(results)

    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total images: {stats['total_images']}")
    print(f"Average confidence: {stats['average_confidence']:.2f}")
    print(f"\nImage types: {stats['image_types']}")
    print(f"Content types: {stats['content_types']}")
    print(f"\nImages with extracted text: {stats['images_with_text']}")
    print(f"Images with identified concepts: {stats['images_with_concepts']}")

    if stats['top_concepts']:
        print(f"\nTop concepts found:")
        for concept, count in list(stats['top_concepts'].items())[:5]:
            print(f"  - {concept}: {count}")

    print("\n VLM analyzer test complete!")
