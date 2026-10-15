# MIDA: Multimodal Interactive Deception Assessment

[Paper (arXiv)](https://arxiv.org/abs/2511.16221) | [Dataset (Hugging Face)](https://huggingface.co/datasets/Ukpkmkkk/MIDA) | [Code (GitHub)](https://github.com/kkkcx/Can-MLLMs-Read-the-Room)

This repository contains the MIDA benchmark and reference implementation accompanying the EMNLP 2026 (Main Conference) paper *"Can MLLMs Read the Room? A Multimodal Benchmark for Assessing Deception in Multi-Party Social Interactions"*. MIDA evaluates whether multimodal large language models can ground language in the visual and auditory cues of multi-party social interaction and assess deception, using the social-deduction game *One Night Ultimate Werewolf* as a controlled but ecologically valid testbed with verifiable ground truth.

The release covers:
- **MIDA-Ego4D**: 819 utterances annotated across 8 game sessions, drawn from the larger 40-session Werewolf subset of Ego4D (Grauman et al., 2022).
- **MIDA-YouTube**: 1,541 utterances annotated across 12 game sessions, drawn from the 151-video YouTube subset curated by Lai et al. (2023).
- **Manually annotated `nightActions` for every labelled game session** (20/20) — a key dataset contribution that records the precise actor, target, and outcome of every night-phase action (Doppelganger / Seer / Robber / Troublemaker / Drunk). Together with the start/end roles and voting outcome, these annotations make every player's private knowledge state at every moment of the game *deterministically reconstructible*, which is what gives MIDA its verifiable per-utterance ground truth (see Section "Why `nightActions` matters" below).
- A reference benchmark runner for both closed-source (OpenAI-compatible APIs) and open-source (HuggingFace Transformers) MLLMs.
- A semi-automated annotation pipeline driven by a Gemini-compatible LLM, including the full annotator rulebook (`code/prompts/annotation_rules.txt`).

## Repository layout

```
Can-MLLMs-Read-the-Room/
├── README.md             # this file
├── DATASHEET.md          # dataset documentation (responsible AI fields)
├── LICENSE               # MIT for code; CC BY 4.0 for annotations
├── REPRODUCE.md          # exact commands to reproduce paper numbers
├── requirements.txt
├── code/
│   ├── annotation/       # Gemini-driven labeling pipeline
│   ├── benchmark/        # closed-source and open-source benchmark runners
│   ├── eval/             # F1 / Macro-F1 / Binary accuracy
│   ├── prompts/          # annotation rulebook + game rules + main task prompt
│   └── utils/            # dataset loader
└── data/
    ├── ego4d/
    │   ├── labels/       # 8 labelled CSVs, 819 utterances
    │   ├── transcripts/  # raw per-segment transcripts
    │   ├── metadata/
    │   │   ├── game_timestamp/   # per-recording game start/end timestamps
    │   │   ├── vote_outcome/     # start/end roles, voting outcome, night actions, warnings
    │   │   └── videos_link.txt   # how to obtain the underlying Ego4D videos
    │   └── splits/       # train/val/test/avalon split JSONs (from Lai et al. 2023)
    └── youtube/
        ├── labels/       # 12 labelled CSVs, 1,541 utterances
        └── metadata/
            ├── vote_outcome/
            └── videos_link.txt   # YouTube video URLs for the 7 source videos
```

The underlying *raw videos* are not redistributed: Ego4D is gated under its own license and YouTube videos must be obtained from their original publishers. See `data/ego4d/metadata/videos_link.txt`, `data/youtube/metadata/videos_link.txt` and `DATASHEET.md` for access instructions.

## Quick start

```bash
pip install -r requirements.txt

# Sanity-check dataset loading
python code/utils/dataset.py --data-root data --source ego4d

# Run a closed-source MLLM (text-only)
export OPENAI_API_KEY=sk-...
python code/benchmark/run_closed_source.py \
    --data-root data --source ego4d \
    --model gpt-4o-mini \
    --out predictions/ego4d_gpt4o_mini_text.csv

# Run an open-source MLLM with one frame per utterance
python code/benchmark/run_open_source.py \
    --data-root data --source ego4d \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --frame-dir /path/to/your/extracted/frames \
    --out predictions/ego4d_qwen25vl_v1f.csv
```

See `REPRODUCE.md` for the full set of commands (closed-source / open-source benchmarks, MIDA-YouTube, frame preparation).

## Why `nightActions` matters

In *One Night Ultimate Werewolf*, the public game record (start/end roles + voting outcome) is **insufficient** to recover what each player actually knew at speaking time: roles can be swapped, faces of cards can be peeked, and many of these events leave no trace in the start/end deltas (e.g. a Seer that looks at center cards, a Drunk that swaps blindly with the center, a Troublemaker that swaps two other players neutrally). Without these mid-night events being recorded, an annotator cannot decide whether a given utterance is TRUE, FALSE, or NEUTRAL — and existing Werewolf datasets stop at the public record.

We therefore manually re-watched every *labelled* game session (20 games total) and produced a structured `nightActions` field, encoding the actor, target, and (where relevant) action_type / copied_role / center_card_swapped of each role's night action. The schema is:

```json
"nightActions": {
  "doppelganger": {"actor": null, "target": null, "copied_role": null},
  "seer":         {"actor": null, "action_type": null, "targets": null},
  "robber":       {"actor": null, "target": null},
  "troublemaker": {"actor": null, "targets": null},
  "drunk":        {"actor": null, "center_card_swapped": null}
}
```

`null` denotes "this role did not act" (or is not in play). The field lives inside each per-game block of `data/{ego4d,youtube}/metadata/vote_outcome/<recording>.json`, alongside `playerNames`, `votingOutcome`, `startRoles`, `endRoles`. For a small number of YouTube games where the source video does not show the night phase (Retro 9 Game3 and Game4), we instead set `"nightActions": null` and add `"nightActions_note"` describing why. Per-recording `vote_outcome` JSONs additionally retain entries for unlabelled games from the same video for context; those entries do not carry a `nightActions` field.

`nightActions` is consumed by the annotation pipeline (see `code/prompts/annotation_rules.txt`, "Suggested Workflow" step 1: *Reconstruct and Document Night Events*). Releasing it lets others **re-derive** the per-utterance veracity labels from scratch and verify our annotations end-to-end.

## Citation

If you use MIDA, please cite our paper and the Werewolf Among Us dataset it builds on:

```bibtex
@inproceedings{kang2026mida,
  title     = {Can MLLMs Read the Room? A Multimodal Benchmark for Assessing Deception in Multi-Party Social Interactions},
  author    = {Kang, Caixin and Huang, Yifei and Ouyang, Liangyang and Zhang, Mingfang and Liu, Ruicong and Sato, Yoichi},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  eprint        = {2511.16221},
  archivePrefix = {arXiv}
}

@inproceedings{lai-etal-2023-werewolf,
  title     = {Werewolf Among Us: Multimodal Resources for Modeling Persuasion Behaviors in Social Deduction Games},
  author    = {Lai, Bolin and Zhang, Hongxin and Liu, Miao and Pariani, Aryan and Ryan, Fiona and Jia, Wenqi and Hayati, Shirley Anugrah and Rehg, James and Yang, Diyi},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
  year      = {2023},
  url       = {https://aclanthology.org/2023.findings-acl.411/}
}
```

## License

- Code: MIT.
- Annotations / labels / metadata in `data/`: CC BY 4.0, except the third-party material below.
- Third-party material from Werewolf Among Us (Lai et al., 2023) is redistributed under its original **Apache-2.0** license (see `LICENSE` and `LICENSES/Apache-2.0.txt`): `data/ego4d/splits/`, `data/ego4d/transcripts/`, `data/ego4d/metadata/game_timestamp/`, the utterance text / speaker / timestamp fields and the persuasive-strategy labels (`true_category`, unmodified) in the label CSVs, and the player names, roles, votes and game times in `vote_outcome/*.json`.
- Underlying video corpora retain their original licenses (Ego4D Terms of Use; respective YouTube content under YouTube ToS).

## Responsible AI

A complete RAI statement (limitations, biases, data subjects, intended use, risks, provenance) accompanies the release as `DATASHEET.md`. See those files for details.
