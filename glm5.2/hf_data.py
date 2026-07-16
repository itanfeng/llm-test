import argparse
import json
import random
from collections import defaultdict

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_bm25_40K",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="swe_prompts.jsonl")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)

    repo_to_indices = defaultdict(list)

    for dataset_index, row in enumerate(dataset):
        repo = row.get("repo", "")

        if not repo:
            continue

        if "text" not in row:
            raise KeyError(
                f"Dataset row has no 'text' field. "
                f"Fields: {list(row.keys())}"
            )

        repo_to_indices[repo].append(dataset_index)

    unique_repos = list(repo_to_indices.keys())

    if args.count > len(unique_repos):
        raise ValueError(
            f"Requested {args.count} unique repositories, but the dataset "
            f"contains only {len(unique_repos)} repositories."
        )

    rng = random.Random(args.seed)

    selected_repos = rng.sample(unique_repos, args.count)

    selected_entries = []

    for repo in selected_repos:
        dataset_index = rng.choice(repo_to_indices[repo])
        row = dataset[dataset_index]

        selected_entries.append(
            {
                "dataset_index": dataset_index,
                "instance_id": row.get("instance_id", ""),
                "repo": repo,
                "prompt": row["text"],
            }
        )

    with open(args.output, "w", encoding="utf-8") as output_file:
        for record in selected_entries:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(
        f"Saved {len(selected_entries)} prompts from "
        f"{len(selected_repos)} unique repositories to {args.output}"
    )

    print("\nSelected prompts:")
    for index, record in enumerate(selected_entries):
        print(
            f"{index:02d}: "
            f"repo={record['repo']}, "
            f"instance_id={record['instance_id']}, "
            f"dataset_index={record['dataset_index']}"
        )


if __name__ == "__main__":
    main()