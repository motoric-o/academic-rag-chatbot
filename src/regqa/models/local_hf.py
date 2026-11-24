"""Local Hugging Face + bitsandbytes adapter (Qwen3 family).

Owner: Member A (see docs/project_spec.md §0.2, §7.1, §7.4).
Guards CUDA imports/calls so this module can be imported safely on CPU-only machines.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from regqa.models.base import GenerationConfig, GenerationResult, ModelAdapter

logger = logging.getLogger(__name__)

VALID_PRECISIONS = frozenset({"fp16", "8bit", "4bit"})
PRECISION_ALIASES = {
    "4-bit": "4bit",
    "8-bit": "8bit",
    "float16": "fp16",
}
VALID_4BIT_QUANT_TYPES = frozenset({"nf4", "fp4"})
VALID_COMPUTE_DTYPES = frozenset({"float16", "bfloat16", "float32"})


def _normalize_precision(precision: str) -> str:
    key = precision.strip().lower()
    return PRECISION_ALIASES.get(key, key)


def _optional_version(module_name: str) -> str | None:
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


def _resolve_torch_dtype(dtype_name: str, torch_mod: Any) -> Any:
    mapping = {
        "float16": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16,
        "float32": torch_mod.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported compute dtype {dtype_name!r}; expected one of {sorted(mapping)}"
        )
    return mapping[dtype_name]


def _is_oom_error(exc: BaseException, torch_mod: Any | None = None) -> bool:
    cuda_oom = getattr(getattr(torch_mod, "cuda", None), "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _infer_input_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            first_param = next(parameters())
            return first_param.device
        except StopIteration:
            return None
    return None


def collect_local_hardware_info(
    *,
    hf_model_id: str | None = None,
    revision: str | None = None,
    quantization_method: str | None = None,
    target_hardware: str | None = None,
    torch_mod: Any | None = None,
) -> dict[str, Any]:
    """Collect local GPU, CUDA, and library version metadata without assuming CUDA."""
    if torch_mod is None:
        try:
            torch_mod = importlib.import_module("torch")
        except Exception:
            torch_mod = None

    cuda_available = False
    gpu_name: str | None = None
    gpu_vram_total_mb: float | None = None
    cuda_version: str | None = None
    torch_version: str | None = None

    if torch_mod is not None:
        torch_version = getattr(torch_mod, "__version__", None)
        cuda_version = getattr(getattr(torch_mod, "version", None), "cuda", None)
        cuda_mod = getattr(torch_mod, "cuda", None)
        if cuda_mod is not None and callable(getattr(cuda_mod, "is_available", None)):
            cuda_available = bool(cuda_mod.is_available())
            if cuda_available:
                try:
                    gpu_name = str(cuda_mod.get_device_name(0))
                except Exception:
                    gpu_name = None
                try:
                    props = cuda_mod.get_device_properties(0)
                    gpu_vram_total_mb = round(props.total_memory / (1024**2), 2)
                except Exception:
                    gpu_vram_total_mb = None

    return {
        "device": "cuda" if cuda_available else "cpu",
        "gpu_name": gpu_name,
        "gpu_vram_total_mb": gpu_vram_total_mb,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "transformers_version": _optional_version("transformers"),
        "accelerate_version": _optional_version("accelerate"),
        "bitsandbytes_version": _optional_version("bitsandbytes"),
        "quantization_method": quantization_method,
        "hf_model_id": hf_model_id,
        "revision": revision,
        "target_hardware": target_hardware,
        "platform": platform.platform(),
    }


@dataclass(frozen=True)
class LocalHFConfig:
    """Configuration loaded from configs/models/local_*.yaml."""

    name: str
    hf_model_id: str
    precision: str  # "fp16" | "8bit" | "4bit"
    revision: str | None = None
    quantization: dict[str, Any] | None = None
    enable_thinking: bool = False
    family: str = "qwen3"
    params_b: float | None = None
    active_params_b: float | None = None
    target_hardware: str | None = None
    device_map: str | dict[str, Any] = "auto"
    trust_remote_code: bool = False
    default_generation: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        normalized = _normalize_precision(self.precision)
        object.__setattr__(self, "precision", normalized)

        if normalized not in VALID_PRECISIONS:
            raise ValueError(
                f"Unsupported precision {self.precision!r}; expected one of "
                f"{sorted(VALID_PRECISIONS)}"
            )

        quant = self.quantization
        if normalized == "fp16":
            if quant is not None:
                raise ValueError("fp16 precision must not specify a quantization config")
        elif normalized == "8bit":
            if quant is not None:
                method = quant.get("method")
                if method is not None and method != "bitsandbytes_int8":
                    raise ValueError(
                        f"8bit precision requires method='bitsandbytes_int8', got {method!r}"
                    )
        elif normalized == "4bit" and quant is not None:
            method = quant.get("method")
            if method is not None and method != "bitsandbytes_nf4":
                raise ValueError(
                    f"4bit precision requires method='bitsandbytes_nf4', got {method!r}"
                )
            quant_type = quant.get("bnb_4bit_quant_type", "nf4")
            if quant_type not in VALID_4BIT_QUANT_TYPES:
                raise ValueError(
                    f"Unsupported bnb_4bit_quant_type {quant_type!r}; expected one of "
                    f"{sorted(VALID_4BIT_QUANT_TYPES)}"
                )
            compute_dtype = quant.get("bnb_4bit_compute_dtype", "float16")
            if compute_dtype not in VALID_COMPUTE_DTYPES:
                raise ValueError(
                    f"Unsupported bnb_4bit_compute_dtype {compute_dtype!r}; expected one of "
                    f"{sorted(VALID_COMPUTE_DTYPES)}"
                )

    @property
    def quantization_method(self) -> str | None:
        if self.precision == "fp16":
            return None
        if self.quantization and self.quantization.get("method"):
            return str(self.quantization["method"])
        return "bitsandbytes_int8" if self.precision == "8bit" else "bitsandbytes_nf4"

    @classmethod
    def from_yaml(cls, path: str | Path) -> LocalHFConfig:
        """Load and validate a LocalHFConfig from configs/models/local_*.yaml."""
        config_path = Path(path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{config_path}: expected a YAML mapping")
        provider = data.get("provider")
        if provider != "local_hf":
            raise ValueError(f"{config_path}: expected provider='local_hf', got {provider!r}")

        generation = data.get("generation") or {}
        if not isinstance(generation, dict):
            raise ValueError(f"{config_path}: 'generation' must be a mapping")

        return cls(
            name=str(data["name"]),
            hf_model_id=str(data["hf_model_id"]),
            precision=str(data["precision"]),
            revision=data.get("revision"),
            quantization=data.get("quantization"),
            enable_thinking=bool(generation.get("enable_thinking", False)),
            family=str(data.get("family", "qwen3")),
            params_b=float(data["params_b"]) if data.get("params_b") is not None else None,
            active_params_b=(
                float(data["active_params_b"])
                if data.get("active_params_b") is not None
                else None
            ),
            target_hardware=data.get("target_hardware"),
            device_map=data.get("device_map", "auto"),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
            default_generation=dict(generation) if generation else None,
        )

    def default_generation_config(self) -> GenerationConfig:
        """Build a GenerationConfig from the YAML generation defaults."""
        gen = self.default_generation or {}
        return GenerationConfig(
            temperature=float(gen.get("temperature", 0.0)),
            do_sample=bool(gen.get("do_sample", False)),
            max_new_tokens=int(gen.get("max_new_tokens", 256)),
            enable_thinking=bool(gen.get("enable_thinking", self.enable_thinking)),
        )


class LocalHFAdapter(ModelAdapter):
    """Local Hugging Face Transformers adapter with optional bitsandbytes quantization."""

    def __init__(self, config: LocalHFConfig) -> None:
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self.load_time_s: float | None = None
        self.load_peak_vram_mb: float | None = None
        self.resolved_revision: str | None = config.revision
        self._hardware_info: dict[str, Any] = {}

    def __enter__(self) -> LocalHFAdapter:
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.unload()

    def _import_runtime(self) -> tuple[Any, Any]:
        try:
            torch_mod = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for LocalHFAdapter. Install the 'local' extra or run in "
                "the GPU worker container."
            ) from exc

        try:
            transformers_mod = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for LocalHFAdapter. Run 'uv sync --extra local'."
            ) from exc

        return torch_mod, transformers_mod

    def _require_quantization_runtime(self, cuda_available: bool) -> None:
        if not cuda_available:
            raise RuntimeError(
                f"Precision '{self.config.precision}' requires CUDA. Verify GPU availability "
                "or fall back to Colab/Kaggle (docs/project_spec.md §0.4)."
            )
        if _optional_version("bitsandbytes") is None:
            raise RuntimeError(
                f"Precision '{self.config.precision}' requires 'bitsandbytes'. Run inside the "
                "Docker GPU worker (`make gpu-up`) or install `--extra local`."
            )
        if _optional_version("accelerate") is None:
            raise RuntimeError(
                f"Precision '{self.config.precision}' requires 'accelerate'. Run inside the "
                "Docker GPU worker (`make gpu-up`) or install `--extra local`."
            )

    def _build_load_kwargs(
        self,
        torch_mod: Any,
        transformers_mod: Any,
        cuda_available: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.config.revision:
            kwargs["revision"] = self.config.revision
        if self.config.trust_remote_code:
            kwargs["trust_remote_code"] = True

        quant = self.config.quantization or {}
        if self.config.precision == "fp16":
            if cuda_available:
                kwargs["dtype"] = torch_mod.float16
                kwargs["device_map"] = self.config.device_map
            else:
                # CPU fallback for tiny test models without requiring accelerate/CUDA.
                kwargs["dtype"] = torch_mod.float32
        elif self.config.precision == "8bit":
            self._require_quantization_runtime(cuda_available)
            bnb_kwargs: dict[str, Any] = {"load_in_8bit": True}
            if "llm_int8_threshold" in quant:
                bnb_kwargs["llm_int8_threshold"] = float(quant["llm_int8_threshold"])
            if "llm_int8_enable_fp32_cpu_offload" in quant:
                bnb_kwargs["llm_int8_enable_fp32_cpu_offload"] = bool(
                    quant["llm_int8_enable_fp32_cpu_offload"]
                )
            kwargs["quantization_config"] = transformers_mod.BitsAndBytesConfig(**bnb_kwargs)
            kwargs["device_map"] = self.config.device_map
        elif self.config.precision == "4bit":
            self._require_quantization_runtime(cuda_available)
            compute_dtype_name = str(quant.get("bnb_4bit_compute_dtype", "float16"))
            bnb_config = transformers_mod.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(quant.get("bnb_4bit_quant_type", "nf4")),
                bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
                bnb_4bit_compute_dtype=_resolve_torch_dtype(compute_dtype_name, torch_mod),
            )
            kwargs["quantization_config"] = bnb_config
            kwargs["device_map"] = self.config.device_map

        return kwargs

    def load(self) -> float:
        """Load tokenizer and model weights, returning load time in seconds."""
        torch_mod, transformers_mod = self._import_runtime()
        cuda_available = bool(torch_mod.cuda.is_available())

        tokenizer_kwargs: dict[str, Any] = {}
        if self.config.revision:
            tokenizer_kwargs["revision"] = self.config.revision
        if self.config.trust_remote_code:
            tokenizer_kwargs["trust_remote_code"] = True

        model_kwargs = self._build_load_kwargs(torch_mod, transformers_mod, cuda_available)

        if cuda_available:
            torch_mod.cuda.synchronize()
            torch_mod.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        tokenizer = transformers_mod.AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            **tokenizer_kwargs,
        )
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                tokenizer.pad_token_id = eos_id

        model = transformers_mod.AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            **model_kwargs,
        )
        if callable(getattr(model, "eval", None)):
            model.eval()

        if cuda_available:
            torch_mod.cuda.synchronize()
            self.load_peak_vram_mb = round(
                torch_mod.cuda.max_memory_allocated() / (1024**2),
                2,
            )
        else:
            self.load_peak_vram_mb = None

        self.load_time_s = round(time.perf_counter() - t0, 4)
        self._tokenizer = tokenizer
        self._model = model
        self.resolved_revision = (
            getattr(getattr(model, "config", None), "_commit_hash", None)
            or self.config.revision
        )
        self._hardware_info = collect_local_hardware_info(
            hf_model_id=self.config.hf_model_id,
            revision=self.resolved_revision,
            quantization_method=self.config.quantization_method,
            target_hardware=self.config.target_hardware,
            torch_mod=torch_mod,
        )
        return self.load_time_s

    def unload(self) -> None:
        """Release model and tokenizer references and free CUDA cache if available."""
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            torch_mod = importlib.import_module("torch")
            if torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
        except Exception:
            pass

    def warmup(self, gen_config: GenerationConfig | None = None) -> None:
        """Run a single untimed warm-up pass before latency benchmarking (§11.5)."""
        cfg = gen_config or GenerationConfig(max_new_tokens=8, do_sample=False, temperature=0.0)
        self.generate(
            messages=[{"role": "user", "content": "Reply OK."}],
            gen_config=cfg,
        )

    def _format_messages(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool,
    ) -> str:
        assert self._tokenizer is not None
        try:
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return str(formatted)

    def _build_generate_kwargs(self, gen_config: GenerationConfig) -> dict[str, Any]:
        assert self._tokenizer is not None
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(gen_config.max_new_tokens),
        }
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            kwargs["pad_token_id"] = pad_token_id

        greedy = (not gen_config.do_sample) or (gen_config.temperature <= 0.0)
        if greedy:
            kwargs["do_sample"] = False
            # Clear Qwen3 default sampling fields to avoid warnings and ensure strict greedy decode.
            kwargs["temperature"] = None
            kwargs["top_p"] = None
            kwargs["top_k"] = None
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = float(gen_config.temperature)
            for key, value in gen_config.extra.items():
                kwargs[key] = value

        return kwargs

    def generate(
        self,
        messages: list[dict[str, str]],
        gen_config: GenerationConfig,
    ) -> GenerationResult:
        """Generate a response using chat template with thinking disabled by default."""
        if self._model is None or self._tokenizer is None:
            self.load()

        torch_mod, _ = self._import_runtime()
        cuda_available = bool(torch_mod.cuda.is_available())
        effective_thinking = bool(gen_config.enable_thinking or self.config.enable_thinking)

        prompt_text = self._format_messages(messages, enable_thinking=effective_thinking)
        encoded = self._tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        input_len = int(input_ids.shape[-1])

        target_device = _infer_input_device(self._model)
        if target_device is not None:
            encoded = {
                key: value.to(target_device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }

        gen_kwargs = self._build_generate_kwargs(gen_config)
        if cuda_available:
            torch_mod.cuda.synchronize()
            torch_mod.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        try:
            with torch_mod.inference_mode():
                sequences = self._model.generate(**encoded, **gen_kwargs)
            if cuda_available:
                torch_mod.cuda.synchronize()
                peak_vram_mb: float | None = round(
                    torch_mod.cuda.max_memory_allocated() / (1024**2),
                    2,
                )
            else:
                peak_vram_mb = None
            latency_s = round(time.perf_counter() - t0, 4)
        except Exception as exc:
            latency_s = round(time.perf_counter() - t0, 4)
            peak_vram_mb = None
            if cuda_available:
                try:
                    peak_vram_mb = round(
                        torch_mod.cuda.max_memory_allocated() / (1024**2),
                        2,
                    )
                except Exception:
                    peak_vram_mb = None
                try:
                    torch_mod.cuda.empty_cache()
                except Exception:
                    pass

            if _is_oom_error(exc, torch_mod):
                error_msg = f"OOM: {exc}"
                logger.error("LocalHFAdapter OOM on %s: %s", self.config.name, exc)
            else:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.error("LocalHFAdapter error on %s: %s", self.config.name, error_msg)

            return GenerationResult(
                raw_output="",
                tokens_in=input_len,
                tokens_out=0,
                latency_s=latency_s,
                model=self.config.name,
                precision=self.config.precision,
                hardware=dict(self._hardware_info),
                peak_vram_mb=peak_vram_mb,
                error=error_msg,
            )

        output_ids = sequences[0][input_len:]
        tokens_out = int(output_ids.shape[-1])
        raw_output = str(
            self._tokenizer.decode(output_ids, skip_special_tokens=True)
        ).strip()

        return GenerationResult(
            raw_output=raw_output,
            tokens_in=input_len,
            tokens_out=tokens_out,
            latency_s=latency_s,
            model=self.config.name,
            precision=self.config.precision,
            hardware=dict(self._hardware_info),
            peak_vram_mb=peak_vram_mb,
            error=None,
        )


def run_smoke_suite(
    config: LocalHFConfig,
    *,
    corpus_path: str | Path = "tests/fixtures/mini_corpus.json",
    eval_path: str | Path = "tests/fixtures/mini_eval.jsonl",
    repeats: int = 2,
    max_questions: int | None = None,
    output_dir: str | Path = "runs/local/smoke",
    adapter: LocalHFAdapter | None = None,
) -> dict[str, Any]:
    """Run a multi-question local smoke suite and write predictions + efficiency summary."""
    from regqa.evaluation.efficiency import (
        compute_efficiency_summary,
        format_efficiency_markdown,
    )
    from regqa.generation.parser import parse_output
    from regqa.generation.prompt_builder import PromptBuilder

    corpus_data = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    sections_by_id = {
        str(sec["section_id"]): sec for sec in corpus_data.get("sections", [])
    }
    eval_items: list[dict[str, Any]] = []
    with Path(eval_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                eval_items.append(json.loads(stripped))
    if max_questions is not None:
        eval_items = eval_items[:max_questions]

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / config.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    summary_path = run_dir / "efficiency_summary.json"

    builder = PromptBuilder()
    gen_config = config.default_generation_config()
    owns_adapter = adapter is None
    active_adapter = adapter if adapter is not None else LocalHFAdapter(config)
    records: list[dict[str, Any]] = []

    try:
        if active_adapter._model is None or active_adapter._tokenizer is None:
            active_adapter.load()
        # Exclude 1 warm-up generation from timed benchmark metrics (§11.5).
        active_adapter.warmup()

        default_fallback_sec = next(iter(sections_by_id.values()), None)
        with predictions_path.open("a", encoding="utf-8") as out_f:
            for repeat_idx in range(1, repeats + 1):
                for item in eval_items:
                    spans = item.get("gold_spans") or []
                    chunks: list[dict[str, Any]] = []
                    for span in spans:
                        sec = sections_by_id.get(str(span.get("section_id", "")))
                        if sec is not None:
                            sec_title = sec.get("title", "")
                            chunks.append(
                                {
                                    "section_id": sec["section_id"],
                                    "header": f"[{sec['section_id']}] {sec_title}".strip(),
                                    "text": sec["text"],
                                }
                            )
                    if not chunks and default_fallback_sec is not None:
                        chunks.append(
                            {
                                "section_id": default_fallback_sec["section_id"],
                                "header": f"[{default_fallback_sec['section_id']}]",
                                "text": default_fallback_sec["text"],
                            }
                        )

                    messages = builder.build_rag_messages(
                        question=str(item["question"]),
                        chunks=chunks,
                    )
                    prompt_hash = builder.compute_prompt_hash(messages)
                    result = active_adapter.generate(messages, gen_config)
                    parsed = parse_output(result.raw_output)

                    record = {
                        "qid": str(item["id"]),
                        "model": result.model,
                        "precision": result.precision,
                        "run_id": f"{run_id}_r{repeat_idx}",
                        "prompt_hash": prompt_hash,
                        "raw_output": result.raw_output,
                        "parsed": parsed.to_dict(),
                        "tokens_in": result.tokens_in,
                        "tokens_out": result.tokens_out,
                        "latency_s": result.latency_s,
                        "peak_vram_mb": result.peak_vram_mb,
                        "hardware": result.hardware,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "error": result.error,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    records.append(record)
    finally:
        if owns_adapter:
            active_adapter.unload()

    summary = compute_efficiency_summary(
        records,
        model=config.name,
        precision=config.precision,
        load_time_s=active_adapter.load_time_s,
        warmup_count=0,
        enforce_reference_gpu_for_latency=True,
    )
    summary_dict = summary.to_dict()
    summary_dict["warmup_excluded"] = 1
    summary_dict["resolved_revision"] = active_adapter.resolved_revision
    summary_dict["predictions_path"] = str(predictions_path)
    summary_path.write_text(json.dumps(summary_dict, indent=2) + "\n", encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "predictions_path": str(predictions_path),
        "summary_path": str(summary_path),
        "summary": summary_dict,
        "markdown": format_efficiency_markdown([summary]),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Member A local Hugging Face adapter utilities.")
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Print JSON hardware, CUDA, and library readiness report.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a configs/models/local_*.yaml file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Load --config and run a single RAG-format smoke prompt.",
    )
    parser.add_argument(
        "--smoke-suite",
        action="store_true",
        help="Run a multi-question smoke suite and write predictions.jsonl + efficiency summary.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("tests/fixtures/mini_corpus.json"),
        help="Structured corpus JSON for --smoke-suite.",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("tests/fixtures/mini_eval.jsonl"),
        help="Evaluation JSONL for --smoke-suite.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Repeats per question for --smoke-suite (default 2 -> 20 calls on mini_eval).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap on questions for --smoke-suite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/local/smoke"),
        help="Output root directory for --smoke-suite.",
    )
    args = parser.parse_args()

    if args.check_env:
        print(json.dumps(collect_local_hardware_info(), indent=2))
        return

    if args.smoke_suite:
        if args.config is None:
            parser.error("--smoke-suite requires --config <path/to/local_*.yaml>")
        cfg = LocalHFConfig.from_yaml(args.config)
        out = run_smoke_suite(
            cfg,
            corpus_path=args.corpus,
            eval_path=args.eval,
            repeats=args.repeats,
            max_questions=args.max_questions,
            output_dir=args.output_dir,
        )
        print(out["markdown"])
        print(json.dumps(out["summary"], indent=2))
        return

    if args.smoke:
        if args.config is None:
            parser.error("--smoke requires --config <path/to/local_*.yaml>")
        from regqa.generation.parser import parse_output
        from regqa.generation.prompt_builder import PromptBuilder

        cfg = LocalHFConfig.from_yaml(args.config)
        builder = PromptBuilder()
        messages = builder.build_rag_messages(
            question="What is the minimum GPA required to take Practical Work?",
            chunks=[
                {
                    "section_id": "8.1",
                    "header": "[8.1]",
                    "text": (
                        "Requirements to take Practical Work: "
                        "1. Complete at least 60 credit units. "
                        "2. Cumulative GPA >= 2.0."
                    ),
                }
            ],
        )
        with LocalHFAdapter(cfg) as adapter:
            result = adapter.generate(messages, cfg.default_generation_config())
            parsed = parse_output(result.raw_output)
            summary = {
                "model": result.model,
                "precision": result.precision,
                "load_time_s": adapter.load_time_s,
                "load_peak_vram_mb": adapter.load_peak_vram_mb,
                "latency_s": result.latency_s,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "peak_vram_mb": result.peak_vram_mb,
                "error": result.error,
                "raw_output": result.raw_output,
                "parsed": parsed.to_dict(),
                "hardware": result.hardware,
            }
            print(json.dumps(summary, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
