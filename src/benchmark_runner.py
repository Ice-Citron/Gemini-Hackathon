#!/usr/bin/env python3
"""
SkyHammer Benchmark Runner
Runs SQLi vulnerability patching benchmark across multiple models.
Produces the comparison matrix for evaluation.
"""

import os
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from openai import OpenAI

# Import judge
from judge_gemini import compute_reward

# API Configuration
try:
    from secretsConfig import GDM_API_KEY
except ImportError:
    GDM_API_KEY = os.environ.get("GDM_API_KEY", "")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


@dataclass
class BenchmarkResult:
    model_id: str
    model_name: str
    total_tests: int
    successful_patches: int
    failed_patches: int
    avg_score: float
    avg_latency_ms: float
    results: List[Dict]


# Model configurations
MODELS = {
    # Gemini models
    "gemini-beta": {
        "name": "Gemini Beta (GDM)",
        "client_type": "xai",
        "model_id": "gemini-beta"
    },
    "gemini-4-1-fast-reasoning": {
        "name": "Gemini 4.1 Fast Reasoning",
        "client_type": "xai",
        "model_id": "gemini-4-1-fast-reasoning"
    },
    # OpenAI models (if API key available)
    "gpt-4o": {
        "name": "GPT-4o (OpenAI)",
        "client_type": "openai",
        "model_id": "gpt-4o"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "client_type": "openai",
        "model_id": "gpt-4o-mini"
    },
    # Anthropic models (if API key available)
    "claude-3-5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "client_type": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022"
    },
    "claude-3-haiku": {
        "name": "Claude 3 Haiku",
        "client_type": "anthropic",
        "model_id": "claude-3-haiku-20240307"
    },
}


def get_client(client_type: str):
    """Get the appropriate API client"""
    if client_type == "xai":
        if not GDM_API_KEY:
            return None
        return OpenAI(api_key=GDM_API_KEY, base_url="https://api.x.ai/v1")

    elif client_type == "openai":
        if not OPENAI_API_KEY:
            return None
        return OpenAI(api_key=OPENAI_API_KEY)

    elif client_type == "anthropic":
        if not ANTHROPIC_API_KEY:
            return None
        try:
            from anthropic import Anthropic
            return Anthropic(api_key=ANTHROPIC_API_KEY)
        except ImportError:
            print("[!] Anthropic library not installed")
            return None

    return None


def generate_patch_with_model(
    client,
    model_config: Dict,
    vulnerable_code: str
) -> tuple:
    """Generate a patch using a specific model"""
    prompt = f"""Fix the SQL injection vulnerability in this Python code.
Use parameterized queries with placeholders (?) instead of string formatting.

VULNERABLE CODE:
```python
{vulnerable_code}
```

Output ONLY the fixed Python code. No explanations, just the secure code:"""

    start_time = time.time()

    try:
        if model_config["client_type"] in ["xai", "openai"]:
            response = client.chat.completions.create(
                model=model_config["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            patch = response.choices[0].message.content.strip()

        elif model_config["client_type"] == "anthropic":
            response = client.messages.create(
                model=model_config["model_id"],
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            patch = response.content[0].text.strip()

        else:
            return None, 0

        latency_ms = (time.time() - start_time) * 1000

        # Clean up code blocks
        if "```python" in patch:
            patch = patch.split("```python")[-1].split("```")[0].strip()
        elif "```" in patch:
            parts = patch.split("```")
            if len(parts) >= 2:
                patch = parts[1].strip()

        return patch, latency_ms

    except Exception as e:
        print(f"    [ERROR] {e}")
        return None, 0


def load_test_set(test_path: str = "data/benchmark_set/test_set.json") -> List[Dict]:
    """Load the test benchmark set"""
    if not os.path.exists(test_path):
        print(f"[!] Test set not found at {test_path}")
        return []

    with open(test_path, "r") as f:
        return json.load(f)


def run_benchmark(
    model_key: str,
    test_set: List[Dict],
    verbose: bool = True
) -> Optional[BenchmarkResult]:
    """Run benchmark for a single model"""
    if model_key not in MODELS:
        print(f"[!] Unknown model: {model_key}")
        return None

    model_config = MODELS[model_key]
    client = get_client(model_config["client_type"])

    if not client:
        print(f"[!] Cannot create client for {model_config['name']} (API key missing?)")
        return None

    print(f"\nBenchmarking: {model_config['name']}")
    print(f"  Model ID: {model_config['model_id']}")
    print(f"  Test cases: {len(test_set)}")

    results = []
    total_score = 0.0
    total_latency = 0.0
    successful = 0
    failed = 0

    for i, scenario in enumerate(test_set):
        if verbose:
            print(f"  [{i+1}/{len(test_set)}] {scenario['id']}...", end=" ")

        vulnerable_code = scenario["vulnerable_code"]

        # Generate patch
        patch, latency_ms = generate_patch_with_model(client, model_config, vulnerable_code)

        if patch:
            # Score the patch
            score, reason = compute_reward(vulnerable_code, patch, use_gemini=False)  # Programmatic only for speed

            total_score += score
            total_latency += latency_ms

            if score >= 0.5:
                successful += 1
                if verbose:
                    print(f"PASS ({score:.2f})")
            else:
                failed += 1
                if verbose:
                    print(f"FAIL ({score:.2f})")

            results.append({
                "id": scenario["id"],
                "score": score,
                "reason": reason,
                "latency_ms": latency_ms,
                "patch_preview": patch[:200] if patch else None
            })
        else:
            failed += 1
            if verbose:
                print("ERROR")
            results.append({
                "id": scenario["id"],
                "score": 0.0,
                "reason": "Generation failed",
                "latency_ms": 0,
                "patch_preview": None
            })

    # Calculate averages
    n_tests = len(test_set)
    avg_score = total_score / n_tests if n_tests > 0 else 0
    avg_latency = total_latency / n_tests if n_tests > 0 else 0

    return BenchmarkResult(
        model_id=model_key,
        model_name=model_config["name"],
        total_tests=n_tests,
        successful_patches=successful,
        failed_patches=failed,
        avg_score=avg_score,
        avg_latency_ms=avg_latency,
        results=results
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print formatted results table"""
    print("\n" + "=" * 80)
    print("                     SQLi VULNERABILITY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Model':<30} {'Success Rate':<15} {'Avg Score':<12} {'Avg Latency':<12}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.avg_score, reverse=True):
        success_rate = f"{r.successful_patches}/{r.total_tests} ({100*r.successful_patches/r.total_tests:.0f}%)"
        print(f"{r.model_name:<30} {success_rate:<15} {r.avg_score:<12.3f} {r.avg_latency_ms:<12.0f}ms")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="SkyHammer Benchmark Runner")
    parser.add_argument(
        "--models",
        type=str,
        default="gemini-beta,gemini-4-1-fast-reasoning",
        help="Comma-separated list of models to benchmark"
    )
    parser.add_argument("--test_set", type=str, default="data/benchmark_set/test_set.json", help="Test set path")
    parser.add_argument("--output", type=str, default="data/benchmark_results.json", help="Output path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load test set
    test_set = load_test_set(args.test_set)
    if not test_set:
        print("[!] No test cases loaded. Run 'python src/factory.py' first.")
        return

    print(f"Loaded {len(test_set)} test cases")

    # Parse models
    model_keys = [m.strip() for m in args.models.split(",")]
    print(f"Models to benchmark: {model_keys}")

    # Available models based on API keys
    print("\nAvailable API keys:")
    print(f"  - GDM_API_KEY: {'Yes' if GDM_API_KEY else 'No'}")
    print(f"  - OPENAI_API_KEY: {'Yes' if OPENAI_API_KEY else 'No'}")
    print(f"  - ANTHROPIC_API_KEY: {'Yes' if ANTHROPIC_API_KEY else 'No'}")

    # Run benchmarks
    all_results = []

    for model_key in model_keys:
        result = run_benchmark(model_key, test_set, verbose=args.verbose)
        if result:
            all_results.append(result)

    if not all_results:
        print("\n[!] No benchmark results collected!")
        return

    # Print results table
    print_results_table(all_results)

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_set_size": len(test_set),
        "results": [asdict(r) for r in all_results]
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
