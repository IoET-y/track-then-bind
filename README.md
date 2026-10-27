# Track, Then Bind

Code for **Ask About the Right Moment: Training-Free Object Identity Reasoning for Video-Language Models**.

Video-language models recognise objects in single frames but lose track of which object is which once identical
objects are shuffled or the identifying evidence disappears. Track, Then Bind separates the two parts of the
problem. The identity of the target is fixed by a frozen VLM at a moment when the evidence for it is visible, and a
promptable video tracker carries that identity afterwards. No component is trained.

1. **Plan.** The VLM fills a fixed schema: the target, its carriers, other subjects, how the target can be
   recognised (by its look, or hidden in a carrier), and whose left and right the question uses.
2. **Perceive.** The VLM boxes every entity on a few frames. SAM2.1 tracks all instances as numbered tracks, and
   tracks them again at a near-native frame rate where they cross or move fast.
3. **Find evidence.** Moments at which the target can be identified: frames on which it is visible, or the moment
   it is hidden.
4. **Bind.** On a short clip with numbered tracks around such a moment, the VLM answers one static question, such
   as "which numbered cup is the ball under right after it is hidden?".
5. **Read out.** The answer is derived from the bound track's final position, stated in text. Questions without a
   target to bind are answered with a visibility timeline built from the tracks.

The same program and configuration ([`track_then_bind/config.py`](track_then_bind/config.py)) run on every
benchmark and with every VLM.

## Installation

```bash
conda create -n ttb python=3.10 -y && conda activate ttb
pip install -r requirements.txt
```

Qwen3-VL-8B and SAM2.1 run on one 80 GB GPU. Qwen3-VL-32B needs two (`--device-map auto`). Videos are decoded with
decord through qwen-vl-utils; if torchcodec is installed as well, set `FORCE_QWENVL_VIDEO_READER=decord`.

Optional environments:
* `scripts/vllm_direct.py` and the InternVL3.5 server need `vllm==0.15.1`.
* `scripts/gemma_server.py` needs a Transformers release with Gemma 4 support (we used 5.20.2), in its own environment.

## Data

All paths are relative to a data root, `./data` by default (`--data` or `TTB_DATA` to change it).

| Path | Content | Source |
|---|---|---|
| `vetbench/` | VET-Bench `cup/` and `card/`, each with `QA.json` and videos | [tiedong/vetbench](https://huggingface.co/datasets/tiedong/vetbench) |
| `shellgame/` | VET-Bench code (answer parser) and the 107 filtered Perception Test cup games | [liutiedong/shellgame](https://github.com/liutiedong/shellgame) at `985ab6b` |
| `toc_bench/` | TOC-Bench code (prompt template, frame sampler, answer extractor) and `answer_final.json` | [cjzcjz666/toc_bench](https://github.com/cjzcjz666/toc_bench) at `e4a10bc` |
| `toc_data/videos/` | TOC-Bench videos | [anonymous-video-benchmark/toc_bench](https://huggingface.co/datasets/anonymous-video-benchmark/toc_bench) at `da30650` |
| `perception_test/` | `mc_question_val.parquet` and `videos/` of the Perception Test validation split | [lmms-lab/PerceptionTest_Val](https://huggingface.co/datasets/lmms-lab/PerceptionTest_Val) |
| `videomme/` | `test.parquet` and `videos/` | [lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) |

```bash
mkdir -p data && cd data
huggingface-cli download tiedong/vetbench --repo-type dataset --local-dir vetbench
git clone https://github.com/liutiedong/shellgame && git -C shellgame checkout 985ab6b
git -C shellgame lfs pull --include "perception_test/cup_games_filtered/*"
git clone https://github.com/cjzcjz666/toc_bench && git -C toc_bench checkout e4a10bc
huggingface-cli download anonymous-video-benchmark/toc_bench --repo-type dataset \
    --revision da3065053d958c38151304e33a42e138b1899eaf --local-dir toc_data
cd .. && python scripts/prepare_data.py --data data
```

`prepare_data.py` writes the item files under `data/items/` from the official releases and the lists in
[`splits/`](splits):

| Split | Items | Use |
|---|---|---|
| VET-Bench cup, card | 50 + 50 | test |
| Perception Test cup games (`pt`, test) | 107 | test; PT-65: its videos with three cups and at least one swap |
| Perception Test cup games (`pt`, dev) | 195 validation videos | development |
| TOC-Bench | 478 dev / 1,845 test questions, split by video | development / test |
| Perception Test, all areas (`pt_val`) | 2,000 validation questions, stratified by area and reasoning type | general video questions |
| Video-MME short and medium (`videomme`) | 1,800 | general video questions |

The Perception Test validation videos are only needed for the development split and `pt_val`, and Video-MME only for
`videomme`. A development set of card-monte videos rendered with the VET-Bench generator can be placed in
`data/card_dev/` (`QA.json` and videos, in the VET-Bench format).

## Running

```bash
python -m track_then_bind.run --bench pt --vlm Qwen/Qwen3-VL-8B-Instruct --out runs/qwen8b/pt.jsonl
```

`--bench` is one of `vet_cup`, `vet_card`, `pt`, `toc`, `pt_val`, `videomme`; `--split` is `test` (default) or
`dev`. Runs resume from an existing output file, and `--shard i --shards n` splits a benchmark by video. Each line of
the output holds the plan, a summary of perception, the bind calls, the facts or timeline given to the read-out,
the answer, and the number of model calls and tokens; `--dump <dir>` also saves the tracks of bound items.

### Other VLMs

```bash
# Qwen3-VL-32B on two GPUs
python -m track_then_bind.run --bench pt --vlm Qwen/Qwen3-VL-32B-Instruct --device-map auto --out runs/qwen32b/pt.jsonl

# InternVL3.5-8B behind a vLLM server
vllm serve OpenGVLab/InternVL3_5-8B --served-model-name internvl3_5-8b --trust-remote-code \
    --max-model-len 40960 --limit-mm-per-prompt '{"image": 160}' --port 8401
python -m track_then_bind.run --bench pt --vlm server:internvl3_5-8b@http://127.0.0.1:8401/v1 --out runs/internvl/pt.jsonl

# Gemma 4 31B behind scripts/gemma_server.py (boxes in [y1, x1, y2, x2], at most 64 frames per request)
python scripts/gemma_server.py --model google/gemma-4-31B-it --port 8501
python -m track_then_bind.run --bench pt --vlm server:gemma-4-31b-it@http://127.0.0.1:8501/v1 \
    --max-frames 64 --box-order yx --out runs/gemma4/pt.jsonl

# OpenAI models (OPENAI_API_KEY); the reasoning effort is the API default unless --reasoning-effort is given
python -m track_then_bind.run --bench pt --vlm openai:gpt-5.5-2026-04-23 --out runs/gpt55/pt.jsonl
```

`--plans-from "runs/qwen8b/vet_card*.jsonl"` reuses the plans of another run, keeping every other step with the
current VLM.

### Baselines and ablations

| Run | Command |
|---|---|
| Direct answer, Transformers | `--config configs/direct.json` |
| Self-consistency, 16 samples | `--config configs/self_consistency.json`; score with `evaluate.py --vote-at k` |
| One answer with more input | `--config configs/more_input.json` |
| Tracks drawn, question asked directly | `--config configs/ablation_tracks_drawn.json` |
| Bind question on the whole video | `--config configs/ablation_whole_video.json` |
| Facts read out with the video | `--config configs/ablation_facts_with_video.json` |

Direct answers in vLLM, as in the main table:

```bash
python scripts/vllm_direct.py --model Qwen/Qwen3-VL-8B-Instruct --benches vet_cup vet_card pt toc --out runs/vllm8b
python scripts/vllm_direct.py --model Qwen/Qwen3-VL-8B-Instruct --benches vet_cup vet_card pt toc --n 16 --out runs/vllm8b_sc
python scripts/vllm_direct.py --model Qwen/Qwen3-VL-8B-Instruct --benches vet_cup vet_card pt toc \
    --fps 16 --max-pixels 288000 --no-resize --toc-frames 96 --max-model-len 65536 --out runs/vllm8b_more
```

## Evaluation

```bash
python scripts/evaluate.py "runs/qwen8b/pt*.jsonl"                                     # accuracy (and PT-65)
python scripts/evaluate.py --compare "runs/vllm8b/pt.jsonl" "runs/qwen8b/pt*.jsonl"   # paired difference, 95% CI, McNemar
python scripts/evaluate.py --vote-at 13 "runs/vllm8b_sc/pt.jsonl"                      # majority vote of 13 samples
python scripts/pt_errors.py "runs/qwen8b/pt*.jsonl"                                    # error breakdown on cup games
```

Accuracy is computed with each benchmark's own answer parser; failed items count as wrong. `--first-letter` reads
replies that omit the required `\boxed{}` by their first option letter.

## Results

Accuracy (%) on the test sets, from the paper.

| VLM | Method | VET cup | VET card | PT | PT-65 | TOC |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | direct answer | 28.0 | 28.0 | 33.6 | 27.7 | 25.1 |
| Qwen3-VL-8B | Track, Then Bind | 100 | 100 | 51.4 | 52.3 | 43.0 |
| Qwen3-VL-32B | direct answer | 32.0 | 38.0 | 35.5 | 33.8 | 37.1 |
| Qwen3-VL-32B | Track, Then Bind | 100 | 100 | 52.3 | 55.4 | 48.6 |
| Gemma 4 31B | direct answer | 36.0 | 28.0 | 36.4 | 32.3 | 40.0 |
| Gemma 4 31B | Track, Then Bind | 100 | 100 | 48.6 | 49.2 | 48.8 |

## Citation

```bibtex
@misc{yang2026ask,
  title  = {Ask About the Right Moment: Training-Free Object Identity Reasoning for Video-Language Models},
  author = {Yang, Di and Li, Xiaochang and Liu, Muyang and Wang, Yufei and Xiong, Yanhai},
  year   = {2026}
}
```

## License

The code is released under the MIT License. The benchmarks and their videos are governed by their own terms.
