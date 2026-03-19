"""
Local validation for kbench notebooks with mock SDK.

Since kaggle_benchmarks isn't available locally, this provides a mock
and validates assertions against gold data and Phase 3 model responses.

Tests:
  1. Gold replay: perfect responses → all assertions pass (256 items)
  2. Bad response replay: empty/random → assertions fail
  3. Phase 3 replay: model responses from pilot results → correlate with original metrics
  4. Prompt size check: flag any prompt > 15,000 chars
  5. Assertion count: verify totals per notebook

CLI:
  python src/test_kbench_local.py --validate-all
  python src/test_kbench_local.py --replay-phase3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_config import get_task_registry
from src.kbench_assertions import build_gold_json, run_assertions


# ── Perfect response generators per task type ─────────────────────────

def generate_perfect_response(instance) -> str:
    """Generate a synthetically perfect response for a task instance."""
    t = instance.task_type
    meta = instance.metadata

    if t == "capacity":
        # Format: ANSWER:\n- Person: item
        lines = ["ANSWER:"]
        for person in meta["people"]:
            lines.append(f"- {person}: {instance.gold_answer[person]}")
        return "\n".join(lines)

    elif t == "interference":
        lines = ["ANSWER:"]
        for key in meta["key_names"]:
            lines.append(f"- {key}: {instance.gold_answer[key]}")
        return "\n".join(lines)

    elif t == "sustained":
        items = ", ".join(instance.gold_answer)
        return f"ANSWER: {items}"

    elif t == "stream_segregation":
        gold = instance.gold_answer
        lines = ["ANSWER:"]
        lines.append(f"1. {gold['1']}")
        if "2" in gold:
            lines.append(f"2. {gold['2']}")
        return "\n".join(lines)

    elif t == "selective":
        items = ", ".join(instance.gold_answer)
        return f"ANSWER: {items}"

    elif t == "stroop":
        lines = ["ANSWER:"]
        for idx_str, val in instance.gold_answer.items():
            lines.append(f"{idx_str}. {val}")
        return "\n".join(lines)

    elif t == "shifting":
        lines = ["ANSWER:"]
        for idx_str, val in instance.gold_answer.items():
            lines.append(f"{idx_str}. {val}")
        return "\n".join(lines)

    elif t == "anomaly":
        primary = instance.gold_answer["primary"]
        anomaly_type = instance.gold_answer["anomaly_type"]
        keywords = meta["detection_keywords"]
        kw = keywords[0] if keywords else anomaly_type
        lines = ["ANSWER:"]
        lines.append(f"1. {primary}")
        lines.append(f"2. There is a {anomaly_type} — {kw} detected in the passage.")
        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown task type: {t}")


def generate_bad_response(_instance) -> str:
    """Generate a response that should fail all assertions."""
    return "I don't know the answer to this question."


# ── Test runners ──────────────────────────────────────────────────────

def test_gold_replay(seed: int = 2026) -> dict:
    """Test 1: Feed perfect responses through assertions → all must pass."""
    print("\n" + "=" * 60)
    print("TEST 1: Gold Replay (perfect responses → all pass)")
    print("=" * 60)

    registry = get_task_registry()
    all_instances = []
    for task_type, (gen_fn, _) in registry.items():
        all_instances.extend(gen_fn(seed=seed))

    total_assertions = 0
    passed_assertions = 0
    failed_items = []
    per_type_stats = defaultdict(lambda: {"total": 0, "passed": 0})

    for inst in all_instances:
        response = generate_perfect_response(inst)
        gold_json = build_gold_json(inst)
        results = run_assertions(inst.task_type, response, gold_json)

        for pattern, expectation, passed in results:
            total_assertions += 1
            per_type_stats[inst.task_type]["total"] += 1
            if passed:
                passed_assertions += 1
                per_type_stats[inst.task_type]["passed"] += 1
            else:
                failed_items.append({
                    "task_id": inst.task_id,
                    "task_type": inst.task_type,
                    "expectation": expectation,
                    "response_preview": response[:200],
                })

    print(f"\nTotal assertions: {total_assertions}")
    print(f"Passed: {passed_assertions}")
    print(f"Failed: {total_assertions - passed_assertions}")
    print(f"Pass rate: {passed_assertions / total_assertions:.1%}")

    print("\nPer task type:")
    for tt in sorted(per_type_stats.keys()):
        stats = per_type_stats[tt]
        rate = stats["passed"] / stats["total"] if stats["total"] else 0
        status = "✓" if rate == 1.0 else "✗"
        print(f"  {status} {tt}: {stats['passed']}/{stats['total']} ({rate:.1%})")

    if failed_items:
        print(f"\nFirst 5 failures:")
        for fi in failed_items[:5]:
            print(f"  {fi['task_id']}: {fi['expectation']}")
            print(f"    response: {fi['response_preview'][:100]}...")

    return {
        "total": total_assertions,
        "passed": passed_assertions,
        "failed": total_assertions - passed_assertions,
        "pass_rate": passed_assertions / total_assertions if total_assertions else 0,
        "all_passed": passed_assertions == total_assertions,
    }


def test_bad_replay(seed: int = 2026) -> dict:
    """Test 2: Bad responses → assertions must fail appropriately."""
    print("\n" + "=" * 60)
    print("TEST 2: Bad Response Replay (garbage → assertions fail)")
    print("=" * 60)

    registry = get_task_registry()
    all_instances = []
    for task_type, (gen_fn, _) in registry.items():
        all_instances.extend(gen_fn(seed=seed))

    total_assertions = 0
    failed_assertions = 0

    for inst in all_instances:
        response = generate_bad_response(inst)
        gold_json = build_gold_json(inst)
        results = run_assertions(inst.task_type, response, gold_json)

        for pattern, expectation, passed in results:
            total_assertions += 1
            if not passed:
                failed_assertions += 1

    fail_rate = failed_assertions / total_assertions if total_assertions else 0

    print(f"\nTotal assertions: {total_assertions}")
    print(f"Failed (expected): {failed_assertions}")
    print(f"Incorrectly passed: {total_assertions - failed_assertions}")
    print(f"Fail rate: {fail_rate:.1%}")

    return {
        "total": total_assertions,
        "failed": failed_assertions,
        "fail_rate": fail_rate,
        "mostly_failed": fail_rate > 0.90,
    }


def _reconstruct_gold_json_from_pilot(item: dict) -> str:
    """Reconstruct gold_json from pilot results' details field.

    Since instance regeneration may not be deterministic across environments,
    we extract gold data from the pilot results themselves.
    """
    tt = item["task_type"]
    details = item.get("details", {})

    if tt == "capacity":
        people = list(details.keys())
        answers = {p: d["gold"] for p, d in details.items()}
        return json.dumps({"answers": answers, "people": people})

    elif tt == "interference":
        key_names = list(details.keys())
        final_values = {k: d["gold"] for k, d in details.items()}
        return json.dumps({"final_values": final_values, "key_names": key_names})

    elif tt == "sustained":
        targets = details.get("gold_targets", [])
        nearmisses = []  # not available in results, not needed for assertion
        return json.dumps({"targets": targets, "nearmisses": nearmisses})

    elif tt == "stream_segregation":
        first_number = details.get("q1_gold", "unknown")
        has_bt = details.get("q2_correct") is not None
        return json.dumps({"first_number": str(first_number), "has_breakthrough": has_bt})

    elif tt == "selective":
        signals = details.get("gold_signals", [])
        distractors = details.get("distractor_values", [])
        return json.dumps({"signals": signals, "distractors": distractors})

    elif tt == "stroop":
        answers = {}
        traps = {}
        for idx_str, d in details.items():
            answers[idx_str] = d["gold"]
            if "trap" in d:
                traps[idx_str] = d["trap"]
        return json.dumps({"answers": answers, "traps": traps})

    elif tt == "shifting":
        answers = {idx_str: d["gold"] for idx_str, d in details.items()}
        return json.dumps({"answers": answers})

    elif tt == "anomaly":
        primary_answer = str(details.get("gold_primary", ""))
        anomaly_type = details.get("anomaly_type", "")
        # detection_keywords not in results — use type_hints fallback
        type_keywords = {
            "language_switch": ["french", "spanish", "german", "italian", "foreign", "language"],
            "code_block": ["code", "programming", "function", "sql", "script"],
            "factual_absurdity": ["incorrect", "wrong", "false", "error", "impossible"],
            "numerical_anomaly": ["large", "amount", "expensive", "unusual", "million"],
            "name_inconsistency": ["name", "changed", "inconsistent", "different"],
        }
        keywords = type_keywords.get(anomaly_type, ["anomaly"])
        return json.dumps({
            "primary_answer": primary_answer,
            "detection_keywords": keywords,
            "anomaly_type": anomaly_type,
        })

    return json.dumps({})


def test_phase3_replay() -> dict:
    """Test 3: Phase 3 model responses → pass rate correlates with original metrics."""
    print("\n" + "=" * 60)
    print("TEST 3: Phase 3 Replay (model responses → correlation)")
    print("=" * 60)

    results_dir = PROJECT_ROOT / "results"
    models = ["llama8b", "phi3", "qwen72b"]

    model_results = {}

    for model in models:
        result_file = results_dir / f"pilot_{model}_results.json"
        if not result_file.exists():
            print(f"  SKIP {model}: {result_file} not found")
            continue

        with open(result_file) as f:
            pilot_results = json.load(f)

        per_type_original = defaultdict(list)
        per_type_assertion = defaultdict(list)

        for item in pilot_results:
            task_type = item["task_type"]
            raw_response = item.get("raw_response", "")
            if not raw_response:
                continue

            # Reconstruct gold_json from pilot results
            gold_json = _reconstruct_gold_json_from_pilot(item)
            if gold_json == "{}":
                continue

            # Run assertions
            try:
                assertion_results = run_assertions(task_type, raw_response, gold_json)
            except Exception:
                continue

            if assertion_results:
                pass_rate = sum(1 for _, _, p in assertion_results if p) / len(assertion_results)
            else:
                pass_rate = 0.0

            per_type_assertion[task_type].append(pass_rate)

            # Get original primary metric
            primary_metrics = {
                "capacity": "accuracy",
                "interference": "accuracy",
                "sustained": "recall",
                "stream_segregation": "stream_accuracy",
                "selective": "sas_score",
                "stroop": "accuracy",
                "shifting": "overall_accuracy",
                "anomaly": "dual_task_score",
            }
            metric_name = primary_metrics.get(task_type)
            if metric_name and metric_name in item:
                per_type_original[task_type].append(item[metric_name])

        # Compute correlation per task type
        print(f"\n  {model}:")
        correlations = []
        for tt in sorted(per_type_original.keys()):
            orig = per_type_original[tt]
            assrt = per_type_assertion[tt]
            if len(orig) < 2 or len(assrt) < 2:
                continue
            # Pearson correlation (manual computation, no numpy dependency)
            n = min(len(orig), len(assrt))
            orig = orig[:n]
            assrt = assrt[:n]
            mean_o = sum(orig) / n
            mean_a = sum(assrt) / n
            cov = sum((o - mean_o) * (a - mean_a) for o, a in zip(orig, assrt)) / n
            std_o = (sum((o - mean_o) ** 2 for o in orig) / n) ** 0.5
            std_a = (sum((a - mean_a) ** 2 for a in assrt) / n) ** 0.5
            if std_o > 0 and std_a > 0:
                r = cov / (std_o * std_a)
            else:
                r = 1.0 if std_o == std_a == 0 else 0.0

            avg_orig = mean_o
            avg_assrt = mean_a
            correlations.append(r)
            status = "✓" if r > 0.5 else "~" if r > 0 else "✗"
            print(f"    {status} {tt}: orig_mean={avg_orig:.3f}, "
                  f"assert_mean={avg_assrt:.3f}, r={r:.3f}")

        if correlations:
            mean_r = sum(correlations) / len(correlations)
            print(f"    Mean correlation: {mean_r:.3f}")
            model_results[model] = mean_r
        else:
            print(f"    No correlations computed")
            model_results[model] = 0.0

    # Overall assessment
    all_r = list(model_results.values())
    overall_r = sum(all_r) / len(all_r) if all_r else 0.0
    print(f"\nOverall mean correlation: {overall_r:.3f}")

    return {
        "model_correlations": model_results,
        "overall_correlation": overall_r,
        "passes": overall_r > 0.5,
    }


def test_prompt_sizes(seed: int = 2026) -> dict:
    """Test 4: Flag any prompt > 15,000 chars."""
    print("\n" + "=" * 60)
    print("TEST 4: Prompt Size Check (flag > 15,000 chars)")
    print("=" * 60)

    registry = get_task_registry()
    flagged = []
    max_size = 0
    total = 0

    for task_type, (gen_fn, _) in registry.items():
        for inst in gen_fn(seed=seed):
            total += 1
            prompt_len = len(inst.prompt)
            if prompt_len > max_size:
                max_size = prompt_len
            if prompt_len > 15000:
                flagged.append({
                    "task_id": inst.task_id,
                    "task_type": inst.task_type,
                    "difficulty": inst.difficulty,
                    "prompt_chars": prompt_len,
                })

    print(f"\nTotal prompts: {total}")
    print(f"Max prompt size: {max_size:,} chars")
    print(f"Flagged (> 15,000): {len(flagged)}")

    if flagged:
        for f in flagged[:10]:
            print(f"  ⚠ {f['task_id']}: {f['prompt_chars']:,} chars")

    return {
        "total": total,
        "max_size": max_size,
        "flagged": len(flagged),
        "all_under_limit": len(flagged) == 0,
    }


def test_assertion_counts(seed: int = 2026) -> dict:
    """Test 5: Verify assertion counts per notebook group."""
    print("\n" + "=" * 60)
    print("TEST 5: Assertion Count Verification")
    print("=" * 60)

    registry = get_task_registry()
    notebook_groups = {
        "task_a_capacity": ["capacity", "interference"],
        "task_b_sustained": ["sustained", "stream_segregation"],
        "task_c_selective": ["selective", "stroop"],
        "task_d_shifting": ["shifting"],
        "task_e_anomaly": ["anomaly"],
    }

    results = {}
    total_assertions = 0
    total_items = 0

    for nb_name, task_types in notebook_groups.items():
        nb_items = 0
        nb_assertions = 0
        for tt in task_types:
            gen_fn, _ = registry[tt]
            instances = gen_fn(seed=seed)
            nb_items += len(instances)
            for inst in instances:
                gold_json = build_gold_json(inst)
                response = generate_perfect_response(inst)
                assertion_results = run_assertions(tt, response, gold_json)
                nb_assertions += len(assertion_results)

        total_items += nb_items
        total_assertions += nb_assertions
        results[nb_name] = {"items": nb_items, "assertions": nb_assertions}
        print(f"  {nb_name}: {nb_items} items, {nb_assertions} assertions")

    print(f"\nTotal: {total_items} items, {total_assertions} assertions")

    return {
        "per_notebook": results,
        "total_items": total_items,
        "total_assertions": total_assertions,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local kbench validation")
    parser.add_argument(
        "--validate-all", action="store_true",
        help="Run all validation tests (gold replay, bad replay, prompt check, counts)",
    )
    parser.add_argument(
        "--replay-phase3", action="store_true",
        help="Run Phase 3 replay test only",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not args.validate_all and not args.replay_phase3:
        args.validate_all = True
        args.replay_phase3 = True

    all_pass = True

    if args.validate_all:
        r1 = test_gold_replay(args.seed)
        if not r1["all_passed"]:
            print(f"\n✗ Gold replay FAILED: {r1['failed']} assertions failed")
            all_pass = False
        else:
            print(f"\n✓ Gold replay PASSED: all {r1['total']} assertions pass")

        r2 = test_bad_replay(args.seed)
        if not r2["mostly_failed"]:
            print(f"\n✗ Bad replay FAILED: only {r2['fail_rate']:.1%} assertions failed")
            all_pass = False
        else:
            print(f"\n✓ Bad replay PASSED: {r2['fail_rate']:.1%} assertions correctly fail")

        r4 = test_prompt_sizes(args.seed)
        if not r4["all_under_limit"]:
            print(f"\n⚠ Prompt size WARNING: {r4['flagged']} prompts over 15K chars")
        else:
            print(f"\n✓ Prompt sizes OK: max={r4['max_size']:,} chars")

        r5 = test_assertion_counts(args.seed)
        print(f"\n✓ Assertion counts: {r5['total_assertions']} total across {r5['total_items']} items")

    if args.replay_phase3:
        r3 = test_phase3_replay()
        if r3["passes"]:
            print(f"\n✓ Phase 3 replay PASSED: overall r={r3['overall_correlation']:.3f}")
        else:
            print(f"\n⚠ Phase 3 replay: r={r3['overall_correlation']:.3f} (below 0.5 threshold)")
            # Note: this is a warning, not a failure — regex-based assertions
            # won't perfectly correlate with fuzzy scorers

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CORE TESTS PASSED")
    else:
        print("SOME TESTS FAILED — review output above")
    print("=" * 60)


if __name__ == "__main__":
    main()
