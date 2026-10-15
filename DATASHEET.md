# MIDA Dataset Card / Datasheet

This datasheet documents MIDA, released with our EMNLP 2026 (Main Conference) paper.

## 1. Motivation and Construct Validity

MIDA is designed to measure whether multimodal large language models can perform *deception assessment in multi-party social interaction* — that is, classify each utterance in a free-flowing group conversation as TRUE, FALSE, or NEUTRAL using the speaker's private knowledge state at the moment of speaking. Prior datasets either constrain the setting to monologue-style videos (Real-Life Trial), structured dyads (Box of Lies), or text-only signals; MIDA fills the gap of multi-party, multimodal interaction with verifiable ground truth.

Ground truth is verifiable because the source corpora record the social-deduction game *One Night Ultimate Werewolf*, whose deterministic rules make every player's private knowledge inferable from their start/end roles, the recorded night actions, and the voting outcome.

## 2. Composition

- **MIDA-Ego4D**: 819 utterances annotated across 8 labelled games drawn from the larger 40-session *Werewolf Among Us* subset of the Ego4D corpus (Grauman et al., 2022). Game sessions were originally recorded in a third-person setup so that all participants are visible.
- **MIDA-YouTube**: 1,541 utterances annotated across 12 labelled games drawn from the larger 151-video YouTube subset curated by Lai et al. (2023, Findings of ACL).
- **Per-utterance labels** (CSV columns):
  - `Timestamp`, `Speaker`, `Line`
  - `true_category`: `No Strategy`, or one or more of {Identity Declaration, Accusation, Defense, Evidence, Interrogation, Call for Action}, comma-separated when an utterance uses several strategies (e.g. `Defense, Evidence`). Inherited unchanged from Lai et al. (2023); MIDA adds the `Truthfulness` and `Explanation` columns
  - `Truthfulness` ∈ {TRUE, FALSE, Neutral}
  - `Explanation` (1–2 sentences)
- **Game metadata**:
  - `vote_outcome/*.json`: per-game record containing
    - `playerNames`, `votingOutcome`, `startRoles`, `endRoles`, `warning` (free-text);
    - **`nightActions`** — *one of MIDA's primary annotation contributions.* For every labelled game we manually re-watched the recording and annotated, for each player who acts at night (Doppelganger, Seer, Robber, Troublemaker, Drunk), the actor, target, and (where applicable) the action_type, copied_role, or center_card_swapped. The schema is:
      ```json
      "nightActions": {
        "doppelganger": {"actor": null, "target": null, "copied_role": null},
        "seer":         {"actor": null, "action_type": null, "targets": null},
        "robber":       {"actor": null, "target": null},
        "troublemaker": {"actor": null, "targets": null},
        "drunk":        {"actor": null, "center_card_swapped": null}
      }
      ```
      `null` denotes "this role did not act / is not in play". For a small number of YouTube games where the source footage does not show the night phase (Retro 9 Game3, Retro 9 Game4), the field itself is set to `null` and accompanied by a `nightActions_note` string. Per-recording `vote_outcome` JSONs additionally retain entries for unlabelled games from the same video for context; those entries do not carry a `nightActions` field. Without these annotations the public record (start/end roles + votes) is insufficient to reconstruct each player's private knowledge state at speaking time, which is what gives MIDA its verifiable per-utterance ground truth.
  - `game_timestamp/*.txt` (Ego4D only): per-recording start/end times of each game segment.
  - `splits/*.json` (Ego4D only): train/val/test splits inherited from Lai et al. 2023.
  - `videos_link.txt`: where to obtain the source videos (Ego4D: the Werewolf Among Us Hugging Face repo; YouTube: one URL per source video).
- **No raw video or audio** is redistributed in this release. The Ego4D corpus is gated and must be downloaded under its own ToS; YouTube videos are linked back to the original uploaders (`data/youtube/metadata/videos_link.txt`).

## 3. Provenance

- **Source datasets** (`prov:wasDerivedFrom`):
  - Ego4D — `https://ego4d-data.org/` (Grauman et al., 2022). Used under the Ego4D license.
  - YouTube Werewolf subset — split definitions and base transcripts from Lai et al. 2023 (`https://aclanthology.org/2023.findings-acl.411/`); raw videos remain on the original uploaders' YouTube channels.
- **Provenance activities** (`prov:wasGeneratedBy`):
  - *Manual annotation of `nightActions`*: the authors re-watched every game session and annotated, for each role that acts at night (Doppelganger, Seer, Robber, Troublemaker, Drunk), the actor, target, and any action-specific outcome (action_type, copied_role, center_card_swapped). This per-game JSON is the primary upstream input to the labelling pipeline and is what makes the per-utterance veracity labels deterministically reconstructible from the public game record.
  - *Semi-automated labelling*: Gemini-2.5-Pro was prompted with the rulebook reproduced in `code/prompts/annotation_rules.txt` (mirrored verbatim in the paper Appendix A.2) to act as a "Game Analyst for AI Deception Detection" and assign per-utterance Truthfulness labels using the manually annotated night actions, voting outcome, and game rules as deterministic context.
  - *Human validation*: a 20% subset of MIDA-Ego4D was re-checked by the authors, yielding 96.6% agreement with the LLM-assigned labels (see paper Appendix D).

## 4. Intended Use Cases (`rai:dataUseCases`)

Validity has been established for:
- **Diagnostic evaluation** of MLLMs on multi-party deception detection.
- **Benchmarking** persuasive-strategy classification and veracity assessment under varying multimodal context (text-only / one frame / multiple frames).
- **Ablation studies** isolating the contribution of multimodal grounding and conversational history.

Validity has *not* been established for, and the dataset is *not* recommended for:
- Operational, real-world deception detection (e.g., interrogation, security screening, hiring).
- Training general-purpose lie detectors on individuals — labels reflect Werewolf game mechanics, not real-world deceptive intent.
- Inferring personal traits, demographics, or psychological profiles of any individual whose recording appears in Ego4D or YouTube.

## 5. Data Limitations (`rai:dataLimitations`)

- **Domain restriction**: The textual content concerns a single social-deduction game; transfer of behavioural cues to other domains is plausible but unverified by us.
- **Class imbalance**: ~85–90% of utterances are NEUTRAL. Reported metrics include macro-F1 and Binary Accuracy specifically because they are robust to this skew.
- **Speaker pool**: Ego4D players are predominantly novices recorded in the U.S.; YouTube players are predominantly experienced English-speaking content creators. Performance on speakers from other linguistic, cultural, or skill backgrounds is not characterised.
- **Single-run evaluation**: To control API/GPU cost, each evaluated MLLM was queried once per utterance with deterministic decoding; we do not report error bars.
- **Annotation scope**: MIDA does not currently provide labels for non-verbal events (gestures, gaze shifts, laughter); it labels only verbal utterances.

## 6. Data Biases (`rai:dataBiases`)

- **Selection bias**: Sources were chosen because they record gameplay with visible participants and verifiable outcomes; this disproportionately samples extroverted, on-camera-comfortable adults.
- **Demographic skew**: As inherited from Ego4D and the curated YouTube channels, the speaker pool over-represents young adult, English-speaking, U.S./Western participants.
- **Strategic-skill skew**: MIDA-YouTube players exhibit substantially higher strategic density and deception rates than MIDA-Ego4D novices (see paper Sec. 3.3); models that overfit to one corpus will not transfer perfectly to the other.
- **Label bias**: Veracity labels (`Truthfulness`, `Explanation`) were generated by an LLM annotator and audited by the authors; while inter-source agreement is high (96.6% on the audited subset), residual systematic errors of the annotator may persist.

## 7. Personal or Sensitive Information (`rai:personalSensitiveInformation`)

- The labelled CSVs contain **first-name pseudonyms or game handles** as they appear in the original recordings. These names are already public in Ego4D and on YouTube; no additional personal identifiers (full names, addresses, contact info) are included.
- No medical, biometric, financial, political, or religious information is recorded.
- No demographic attributes (age, gender, ethnicity) are annotated by us.

If a person depicted in the source corpora wishes their content removed, please contact us by opening an issue at https://github.com/kkkcx/Can-MLLMs-Read-the-Room/issues; matching utterances and metadata will be redacted within 14 days.

## 8. Synthetic Data (`rai:hasSyntheticData`)

`false`. All utterances and game outcomes originate from real human gameplay. Veracity labels are *generated* by an LLM annotator but *grounded* in human-authored rules and human-recorded ground truth (start/end roles, voting outcomes, manually annotated night actions). The labels are not a synthetic augmentation of a smaller human-labelled set.

## 9. Social Impact (`rai:dataSocialImpact`)

**Positive impacts.** MIDA enables systematic diagnosis of MLLMs' social-cognitive shortcomings — limited multimodal grounding, weak Theory-of-Mind, conservative alignment in adversarial settings — that directly bear on the safety and trustworthiness of conversational AI assistants, content-moderation systems, and computational social-science platforms.

**Potential negative impacts.** Deception-detection technology is dual-use. Improvements demonstrated on MIDA might be repurposed for surveillance, manipulation, or coercive interrogation. We mitigate this by (i) explicitly framing MIDA as a *diagnostic* benchmark rather than a deployable detector, (ii) restricting the dataset to a controlled game environment whose outputs are *not* directly transferable to real-world deception about identity, intent, or facts, and (iii) refusing to release any pre-trained classifier built on MIDA. Researchers reusing this dataset should preserve these constraints.

## 10. License and Hosting

- Annotations, metadata, and prompts are released under **CC BY 4.0**, except the third-party material below.
- Third-party material from Werewolf Among Us (Lai et al., 2023) is redistributed under its original **Apache-2.0** license (see `LICENSE` and `LICENSES/Apache-2.0.txt`): `data/ego4d/splits/`, `data/ego4d/transcripts/`, `data/ego4d/metadata/game_timestamp/`, the utterance text / speaker / timestamp fields and the persuasive-strategy labels (`true_category`, unmodified) in the label CSVs, and the player names, roles, votes and game times in `vote_outcome/*.json`.
- Code is released under **MIT**.
- Code: https://github.com/kkkcx/Can-MLLMs-Read-the-Room. Data: https://huggingface.co/datasets/Ukpkmkkk/MIDA. Paper: https://arxiv.org/abs/2511.16221.

## 11. Maintenance

- The dataset will be versioned. Each future revision will receive a new tag on the hosting platform and a corresponding entry in `CHANGELOG.md`.
- Errata and re-annotation requests will be tracked through the public issue tracker.
