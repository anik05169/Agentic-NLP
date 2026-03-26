"""
Exploratory Data Analysis (EDA) Report for Legal AI NLP Pipeline
================================================================
Analyzes all three datasets: LegalBench, LexGLUE, and Auxiliary (Dolly).
Produces console output + saves structured results to eda_results.json.

Usage:
    python eda_report.py
"""

import os
import json
import statistics
from collections import Counter, defaultdict
from glob import glob

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
LB_MASTER     = os.path.join(BASE_DIR, "legalbench", "data", "legalbench_master.jsonl")
LG_DIR        = os.path.join(BASE_DIR, "lex_glue", "lex_glue_data")
AUX_MASTER    = os.path.join(BASE_DIR, "auxiliary", "data", "auxiliary_master.jsonl")
OUTPUT_FILE   = os.path.join(BASE_DIR, "eda_results.json")


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list[dict]:
    """Reads a JSONL file and returns a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def word_count(text) -> int:
    """Handles both str and list-of-str (e.g. ECtHR paragraphs)."""
    if isinstance(text, list):
        return sum(len(s.split()) for s in text)
    return len(str(text).split())


def char_count(text) -> int:
    if isinstance(text, list):
        return sum(len(s) for s in text)
    return len(str(text))


def length_stats(lengths: list[int]) -> dict:
    """Returns descriptive stats dict for a list of integer lengths."""
    if not lengths:
        return {"count": 0}
    return {
        "count":  len(lengths),
        "min":    min(lengths),
        "max":    max(lengths),
        "mean":   round(statistics.mean(lengths), 1),
        "median": round(statistics.median(lengths), 1),
        "stdev":  round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
        "p5":     round(sorted(lengths)[int(len(lengths) * 0.05)], 1),
        "p95":    round(sorted(lengths)[int(len(lengths) * 0.95)], 1),
    }


def top_n(counter: Counter, n: int = 10) -> dict:
    """Returns top-n items from a Counter as {key: count}."""
    return dict(counter.most_common(n))


def print_section(title: str):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def print_stats(label: str, stats: dict):
    print(f"\n  {label}:")
    for k, v in stats.items():
        print(f"    {k:>8}: {v}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. LegalBench EDA
# ══════════════════════════════════════════════════════════════════════════════
def eda_legalbench() -> dict:
    print_section("1. LEGALBENCH DATASET ANALYSIS")

    records = load_jsonl(LB_MASTER)
    total = len(records)
    print(f"\n  Total records: {total:,}")

    # Per-task counts
    task_counter = Counter(r["task"] for r in records)
    print(f"  Unique tasks:  {len(task_counter)}")

    # Split distribution
    split_counter = Counter(r["split"] for r in records)
    print(f"  Splits:        {dict(split_counter)}")

    # Instruction lengths (word-level)
    instr_words = [word_count(r["instruction"]) for r in records]
    print_stats("Instruction length (words)", length_stats(instr_words))

    # Instruction lengths (char-level)
    instr_chars = [char_count(r["instruction"]) for r in records]
    print_stats("Instruction length (chars)", length_stats(instr_chars))

    # Response lengths
    resp_words = [word_count(r["response"]) for r in records]
    print_stats("Response length (words)", length_stats(resp_words))

    # Unique response values per task (class count)
    task_classes = defaultdict(set)
    for r in records:
        task_classes[r["task"]].add(r["response"])

    class_counts = {t: len(classes) for t, classes in task_classes.items()}
    binary_tasks = [t for t, c in class_counts.items() if c == 2]
    multi_tasks  = [t for t, c in class_counts.items() if c > 2]

    print(f"\n  Binary classification tasks: {len(binary_tasks)}")
    print(f"  Multi-class tasks:          {len(multi_tasks)}")

    # Smallest / largest tasks
    sorted_tasks = sorted(task_counter.items(), key=lambda x: x[1])
    print(f"\n  5 smallest tasks (by record count):")
    for t, c in sorted_tasks[:5]:
        print(f"    {t}: {c}")
    print(f"  5 largest tasks:")
    for t, c in sorted_tasks[-5:]:
        print(f"    {t}: {c}")

    # Class imbalance check — look at most imbalanced task
    print(f"\n  Class imbalance spotlight (top 5 most imbalanced tasks):")
    imbalance_scores = {}
    for task, classes in task_classes.items():
        task_records = [r for r in records if r["task"] == task]
        resp_dist = Counter(r["response"] for r in task_records)
        if len(resp_dist) >= 2:
            counts = list(resp_dist.values())
            ratio = max(counts) / max(min(counts), 1)
            imbalance_scores[task] = round(ratio, 1)

    for task, ratio in sorted(imbalance_scores.items(), key=lambda x: -x[1])[:5]:
        task_records = [r for r in records if r["task"] == task]
        dist = Counter(r["response"] for r in task_records)
        print(f"    {task}: ratio={ratio}x  distribution={dict(dist)}")

    # Empty responses
    empty_responses = sum(1 for r in records if not r["response"].strip())
    print(f"\n  Records with empty responses: {empty_responses}")

    return {
        "total_records":    total,
        "unique_tasks":     len(task_counter),
        "splits":           dict(split_counter),
        "instruction_word_stats": length_stats(instr_words),
        "instruction_char_stats": length_stats(instr_chars),
        "response_word_stats":    length_stats(resp_words),
        "binary_tasks":     len(binary_tasks),
        "multi_class_tasks": len(multi_tasks),
        "empty_responses":  empty_responses,
        "smallest_tasks":   dict(sorted_tasks[:5]),
        "largest_tasks":    dict(sorted_tasks[-5:]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  2. LexGLUE EDA
# ══════════════════════════════════════════════════════════════════════════════
def eda_lex_glue() -> dict:
    print_section("2. LEXGLUE DATASET ANALYSIS")

    files = sorted(glob(os.path.join(LG_DIR, "*.jsonl")))
    print(f"\n  Total split files found: {len(files)}")

    # Group by sub-dataset
    subdatasets = defaultdict(dict)
    for fp in files:
        name = os.path.basename(fp).replace(".jsonl", "")
        # e.g., "eurlex_test" -> dataset="eurlex", split="test"
        # Handle datasets with underscores (ecthr_a, ecthr_b, unfair_tos, case_hold)
        known_prefixes = ["ecthr_a", "ecthr_b", "unfair_tos", "case_hold"]
        dataset = None
        for prefix in known_prefixes:
            if name.startswith(prefix + "_"):
                dataset = prefix
                split = name[len(prefix) + 1:]
                break
        if dataset is None:
            parts = name.rsplit("_", 1)
            dataset = parts[0]
            split = parts[1] if len(parts) > 1 else "unknown"

        subdatasets[dataset][split] = fp

    results = {}
    for ds_name in sorted(subdatasets.keys()):
        splits = subdatasets[ds_name]
        print(f"\n  ─── {ds_name.upper()} ───")

        ds_result = {}
        all_text_words = []
        all_labels = []

        for split_name in ["train", "validation", "test"]:
            if split_name not in splits:
                continue
            records = load_jsonl(splits[split_name])
            count = len(records)
            ds_result[f"{split_name}_count"] = count
            print(f"    {split_name}: {count:,} records")

            # Text length
            text_key = "text"
            if records and text_key in records[0]:
                words = [word_count(r[text_key]) for r in records]
                all_text_words.extend(words)

            # Labels
            for r in records:
                if "labels" in r:
                    lbl = r["labels"]
                    if isinstance(lbl, list):
                        all_labels.extend(lbl)
                    else:
                        all_labels.append(lbl)
                elif "label" in r:
                    all_labels.append(r["label"])

        # Text stats
        if all_text_words:
            ts = length_stats(all_text_words)
            ds_result["text_word_stats"] = ts
            print_stats("Text length (words)", ts)

        # Label stats
        if all_labels:
            label_counter = Counter(all_labels)
            ds_result["unique_labels"] = len(label_counter)
            ds_result["total_label_occurrences"] = len(all_labels)
            print(f"    Unique labels: {len(label_counter)}")

            # For multi-label datasets, show labels per example
            if records and "labels" in records[0]:
                labels_per_example = [len(r.get("labels", [])) for r in records]
                avg_labels = round(statistics.mean(labels_per_example), 2) if labels_per_example else 0
                ds_result["avg_labels_per_example"] = avg_labels
                print(f"    Avg labels per example: {avg_labels}")

        results[ds_name] = ds_result

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  3. Auxiliary (Dolly) EDA
# ══════════════════════════════════════════════════════════════════════════════
def eda_auxiliary() -> dict:
    print_section("3. AUXILIARY (DOLLY 15K) DATASET ANALYSIS")

    records = load_jsonl(AUX_MASTER)
    total = len(records)
    print(f"\n  Total records: {total:,}")

    # Task categories
    task_counter = Counter(r["task"] for r in records)
    print(f"  Task categories: {len(task_counter)}")
    for task, cnt in task_counter.most_common():
        print(f"    {task}: {cnt:,}")

    # Instruction lengths
    instr_words = [word_count(r["instruction"]) for r in records]
    print_stats("Instruction length (words)", length_stats(instr_words))

    # Response lengths
    resp_words = [word_count(r["response"]) for r in records]
    print_stats("Response length (words)", length_stats(resp_words))

    # Empty responses
    empty = sum(1 for r in records if not r["response"].strip())
    print(f"\n  Records with empty responses: {empty}")

    return {
        "total_records":    total,
        "task_categories":  dict(task_counter),
        "instruction_word_stats": length_stats(instr_words),
        "response_word_stats":    length_stats(resp_words),
        "empty_responses":  empty,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4. Cross-Dataset Summary
# ══════════════════════════════════════════════════════════════════════════════
def cross_dataset_summary(lb: dict, lg: dict, aux: dict):
    print_section("4. CROSS-DATASET SUMMARY")

    lg_total = sum(
        v.get("train_count", 0) + v.get("validation_count", 0) + v.get("test_count", 0)
        for v in lg.values()
    )

    print(f"\n  LegalBench total records:  {lb['total_records']:>10,}")
    print(f"  LexGLUE total records:     {lg_total:>10,}")
    print(f"  Auxiliary total records:    {aux['total_records']:>10,}")
    print(f"  {'─' * 40}")
    print(f"  Combined total:            {lb['total_records'] + lg_total + aux['total_records']:>10,}")

    # Dataset size on disk
    sizes = {}
    for label, path in [("LegalBench", LB_MASTER), ("Auxiliary", AUX_MASTER)]:
        if os.path.exists(path):
            sizes[label] = os.path.getsize(path)

    lg_size = sum(os.path.getsize(f) for f in glob(os.path.join(LG_DIR, "*.jsonl")))
    sizes["LexGLUE"] = lg_size

    print(f"\n  Disk usage:")
    for label, size in sizes.items():
        print(f"    {label}: {size / (1024 * 1024):.1f} MB")
    print(f"    Total: {sum(sizes.values()) / (1024 * 1024):.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "▓" * 70)
    print("  AGENTIC NLP — EXPLORATORY DATA ANALYSIS REPORT")
    print("▓" * 70)

    lb_results  = eda_legalbench()
    lg_results  = eda_lex_glue()
    aux_results = eda_auxiliary()

    cross_dataset_summary(lb_results, lg_results, aux_results)

    # Save results
    all_results = {
        "legalbench": lb_results,
        "lex_glue":   lg_results,
        "auxiliary":   aux_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\n✅ Full EDA results saved to: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
