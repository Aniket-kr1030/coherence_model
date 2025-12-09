#!/usr/bin/env python3
"""
Download a bundle of coherence / dialogue / QA / narrative datasets
(no login required) plus a few world-simulation repos, and zip them
into dream_coherence_bundle.zip.

Datasets are stored using `datasets.save_to_disk(...)` under ./datasets/.
Repos are shallow-cloned under ./world_envs/.
"""

import os
import shutil
import subprocess

from datasets import load_dataset


# Hugging Face datasets to pull (name -> (hf_id, subset_or_None))
HF_DATASETS = {
    # NLI / coherence / similarity
    "snli": ("stanfordnlp/snli", None),
    "multi_nli": ("nyu-mll/multi_nli", None),
    "anli": ("facebook/anli", None),
    "sick": ("mteb/sickr-sts", None),
    "sts_benchmark": ("mteb/stsbenchmark-sts", None),
    "msr_paraphrase": ("glue", "mrpc"),

    # QA / dialogue
    "natural_questions": ("sentence-transformers/natural-questions", None),
    "boolq": ("google/boolq", None),
    "coqa": ("stanfordnlp/coqa", None),
    "quac": ("allenai/quac", None),
    "trivia_qa": ("mandarjoshi/trivia_qa", None),

    # narrative / stories
    "rocstories": ("mintujupally/ROCStories", None),
    "writingprompts": ("euclaise/writingprompts", None),

    # dialogue / task-oriented
    "daily_dialog": ("agentlans/li2017dailydialog", None),
    "multiwoz21": ("ConvLab/multiwoz21", None),

    # assistant-style conversations
    "openassistant_oasst1": ("OpenAssistant/oasst1", None),
}


# World / environment repos to shallow-clone
GIT_REPOS = {
    "TextWorld": "https://github.com/microsoft/TextWorld.git",
    "Jericho": "https://github.com/microsoft/jericho.git",
    "ALFWorld": "https://github.com/alfworld/alfworld.git",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_hf_datasets(root: str = "datasets") -> None:
    """Download all HF datasets into arrow format under `root`."""
    ensure_dir(root)
    for name, (hf_id, subset) in HF_DATASETS.items():
        out_dir = os.path.join(root, name)
        if os.path.exists(out_dir):
            print(f"[skip] {name}: already exists at {out_dir}")
            continue

        print(f"[download] {name} from {hf_id!r}"
              + (f" (config={subset!r})" if subset else "")
              + " ...")

        try:
            if subset is None:
                ds = load_dataset(hf_id)
            else:
                ds = load_dataset(hf_id, subset)

            ds.save_to_disk(out_dir)
            print(f"[ok] {name}: saved to {out_dir}")
        except Exception as e:
            print(f"[error] {name}: {e}")


def clone_repos(root: str = "world_envs") -> None:
    """Shallow-clone world / environment repos into `root`."""
    ensure_dir(root)
    for name, url in GIT_REPOS.items():
        target = os.path.join(root, name)
        if os.path.exists(target):
            print(f"[skip] {name}: already exists at {target}")
            continue

        print(f"[clone] {name} from {url} ...")
        try:
            subprocess.check_call(
                ["git", "clone", "--depth", "1", url, target]
            )
            print(f"[ok] {name}: cloned into {target}")
        except Exception as e:
            print(f"[error] {name}: {e}")


def make_zip(
    archive_name: str = "dream_coherence_bundle",
    roots=("datasets", "world_envs"),
) -> None:
    """
    Create archive_name.zip containing the given roots under a
    common parent directory.
    """
    parent = "dream_coherence_bundle"
    if os.path.exists(parent):
        print(f"[info] removing existing directory {parent}")
        shutil.rmtree(parent)

    os.makedirs(parent, exist_ok=True)

    # Move existing roots into the parent
    for root in roots:
        if os.path.exists(root):
            dest = os.path.join(parent, os.path.basename(root))
            print(f"[pack] moving {root} -> {dest}")
            shutil.move(root, dest)
        else:
            print(f"[warn] {root} does not exist, skipping in zip")

    # Create zip archive
    shutil.make_archive(archive_name, "zip", root_dir=parent)
    print(f"[ok] created {archive_name}.zip")

    # Move things back out for reuse (optional)
    for sub in os.listdir(parent):
        src = os.path.join(parent, sub)
        dst = sub
        if not os.path.exists(dst):
            shutil.move(src, dst)

    shutil.rmtree(parent)


if __name__ == "__main__":
    download_hf_datasets()
    clone_repos()
    make_zip()
