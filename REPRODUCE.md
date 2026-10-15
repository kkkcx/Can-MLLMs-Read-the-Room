# Reproducing the MIDA Benchmark Results

This document gives the exact commands needed to reproduce the benchmark numbers reported in the paper (Tables 1 and 2 of the EMNLP 2026 / arXiv v2 version). All commands assume you are at the repository root and that the data layout matches the one described in `README.md`.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

API-based runners read credentials strictly from environment variables:

```bash
export OPENAI_API_KEY=<your_key>
export OPENAI_BASE_URL=<optional_proxy>    # optional, e.g. for Gemini/Anthropic via OpenAI-compatible proxy
```

## 1. Data sanity check

```bash
python code/utils/dataset.py --data-root data --source ego4d
# expected: ego4d: 8 games, 819 utterances
python code/utils/dataset.py --data-root data --source youtube
# expected: youtube: 12 games, 1541 utterances
```

## 2. Frame preparation (only required for vision-augmented runs)

The release does *not* redistribute raw videos. Once you have downloaded Ego4D under its license and the YouTube source videos from the channels listed in the metadata, extract one frame per labelled timestamp into a directory of your choice. The benchmark scripts accept any directory layout via `--frame-pattern`.

The default pattern for **Ego4D** is `{game_id}_{part}/frame_00_{mm:02d}_{ss:02d}.jpg`, where `{part}` is the 1-indexed video segment within the game (Ego4D records each game as 1-3 contiguous segments). The runner automatically increments `{part}` whenever it detects a backward time-jump within a game, matching the `Game{N}_1`, `Game{N}_2`, `Game{N}_3` convention used in `data/ego4d/transcripts/`.

For **MIDA-YouTube**, pass a pattern that omits `{part}`:

```bash
--frame-pattern '{game_id}/frame_00_{mm:02d}_{ss:02d}.jpg'
```

## 3. Closed-source benchmark (Table 1; closed-source rows of Table 2)

Text-only baseline, all 6 closed-source models on both splits. Model ids are the exact ones used in the paper (see the API snapshot table in the appendix); `gemini-2.5-pro-exp-03-25` has since been retired by Google, so use `gemini-2.5-pro` if the experimental id is no longer served (numbers may then differ slightly).

```bash
for MODEL in gpt-4o-mini gpt-4o gpt-5-nano gpt-5-mini gemini-2.5-pro-exp-03-25 claude-3-5-haiku-20241022; do
  for SPLIT in ego4d youtube; do
    python code/benchmark/run_closed_source.py \
        --data-root data --source $SPLIT --model $MODEL \
        --out predictions/${SPLIT}_${MODEL}_text.csv
  done
done
```

Calls that fail or return unparsable JSON are excluded from scoring; their count is reported as `n_errors` in the `*.metrics.json` file.

Vision-augmented (one frame per utterance):

```bash
python code/benchmark/run_closed_source.py \
    --data-root data --source ego4d --model gpt-4o-mini \
    --frame-dir frames/ego4d \
    --out predictions/ego4d_gpt4o_mini_v1f.csv

python code/benchmark/run_closed_source.py \
    --data-root data --source youtube --model gpt-4o-mini \
    --frame-dir frames/youtube \
    --frame-pattern '{game_id}/frame_00_{mm:02d}_{ss:02d}.jpg' \
    --out predictions/youtube_gpt4o_mini_v1f.csv
```

## 4. Open-source benchmark (open-source rows of Table 2)

```bash
for MODEL_ID in \
    meta-llama/Meta-Llama-3-8B-Instruct \
    OpenGVLab/InternVL3_5-8B-HF \
    Qwen/Qwen2-VL-7B-Instruct \
    Qwen/Qwen2.5-VL-7B-Instruct ; do
  for SPLIT in ego4d youtube; do
    python code/benchmark/run_open_source.py \
        --data-root data --source $SPLIT --model-id $MODEL_ID \
        --out predictions/${SPLIT}_$(echo $MODEL_ID | tr '/' '_').csv
  done
done

for SPLIT in ego4d youtube; do
  python code/benchmark/run_open_source.py \
      --data-root data --source $SPLIT --model-id deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
      --out predictions/${SPLIT}_deepseek-ai_DeepSeek-R1-Distill-Llama-8B.csv

  python code/benchmark/run_open_source.py \
      --data-root data --source $SPLIT --model-id Qwen/Qwen3-8B \
      --max-new-tokens 32768 --use-generation-config \
      --out predictions/${SPLIT}_Qwen_Qwen3-8B.csv
done
```

`<think>` traces are stripped before parsing. Unparsable outputs are scored as the default prediction (all strategies FALSE, veracity Neutral). Add `--frame-dir frames/ego4d` (or `frames/youtube` with the YouTube `--frame-pattern`) for the vision-augmented variant; text-only models print a warning and ignore frames.

## 5. Re-running annotation on a single game (optional)

If you want to regenerate labels from scratch for one game:

```bash
python code/annotation/annotate.py \
    --transcript data/ego4d/transcripts/<recording_id>_Game<N>_1.txt \
    --vote-outcome data/ego4d/metadata/vote_outcome/<recording_id>.json \
    --game-key Game<N> \
    --model gemini-2.5-pro \
    --out predictions/relabelled_<recording_id>_Game<N>.csv
```

The full annotator rulebook lives at `code/prompts/annotation_rules.txt` and is reproduced verbatim in the paper Appendix A.2.

## 6. Compute metrics

`run_closed_source.py` and `run_open_source.py` write a `*.metrics.json` next to each predictions CSV. To recompute metrics for an already-saved predictions CSV (e.g., after editing the parser):

```bash
python -c "
import csv, json
from code.eval.metrics import strategy_metrics, veracity_metrics
rows = list(csv.DictReader(open('predictions/ego4d_gpt-4o-mini_text.csv')))
gold_cat = [r['true_category'] for r in rows]
gold_ver = [r['Truthfulness'] for r in rows]
pred = [{k: r['pred_'+k] for k in ('identity_declaration','accusation','defense','evidence','interrogation','call_for_action')} for r in rows]
pred_ver = [r['pred_veracity'] for r in rows]
print(json.dumps({'strategy': strategy_metrics(gold_cat, pred), 'veracity': veracity_metrics(gold_ver, pred_ver)}, indent=2))"
```
