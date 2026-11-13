# SubmitChain — code package

Prototype for the paper *SubmitChain: A Privacy-Preserving Consortium Blockchain for Detecting
Paraphrased Duplicate Manuscript Submissions Across Journals*.

## What has been tested, and what has not

| Part | Status |
| --- | --- |
| `chaincode/` smart contract logic | 11 unit tests pass against an in-memory stand-in for the Fabric stub (`npm test`, needs only Node 18+) |
| `gateway/triage.py` | Logic tested with a scripted stand-in for the LLM; **not run** against a real model |
| `experiments/` | Contract microbenchmark and analyses **run**; results in `chaincode_bench.json`, `index_recall.json`, `analysis.json`, `storage_breakdown.json` (these are the numbers in Sections 3.1-3.2) |
| `gateway/` fingerprinting, baselines, evaluation, privacy attack, plots | `python selftest.py` passes on **synthetic** data |
| `gateway/embed.py`, `build_benchmark.py paraphrase` | Written, **not run** (need PyTorch, model downloads and an LLM server) |
| `app/client.js`, `caliper/` | Written, **not run** (need a Fabric network) |
| Deployment on Hyperledger Fabric | **Not done.** Follow the steps below |

Sections 3.1-3.2 of the paper come from `experiments/` (re-run: `node experiments/chaincode_bench.js`,
`node experiments/index_recall.js`, `python experiments/analysis.py`). Sections 3.3-3.5 have no results yet.
`gateway/selftest.py` prints numbers from synthetic data only; they must never appear in the paper.

## 1. Smart contract

    cd chaincode && npm test            # logic tests, no network needed

Deploy on the Fabric test network (Docker and fabric-samples v2.5 required):

    cd fabric-samples/test-network
    ./network.sh up createChannel -ca -s couchdb
    ./network.sh deployCC -ccn submitchain -ccp /path/to/submitchain/chaincode -ccl javascript

    cd /path/to/submitchain/app && npm install
    node client.js config '{"bits":256,"tauH":<from step 4>,"tauB":<from step 4>,"mode":"index","tables":64,"tableBits":16,"seed":"<consortium seed>"}'

The test network has two organisations; `addOrg3/addOrg3.sh up` adds a third. For 5 and 10
organisations generate a network with a tool such as Minifabric or Fablo.

Things to know before measuring:
* `RegisterSubmission` reads a key range and writes in the same transaction. Fabric re-checks the
  range at commit, so two registrations in the same block can invalidate one another (phantom-read
  protection). The gateway must retry. Report the retry rate alongside throughput.
* All channel members can read every record, including author pseudonyms and alerts (paper, Section 2.6). An alert shows that two named
  journals hold a possible duplicate, with distances, and nothing about content. Hiding alerts from
  uninvolved members needs private data collections and is not implemented.
* `SetConfig` is restricted to the first organisation that calls it. A production consortium would
  govern configuration through the chaincode endorsement policy.

## 2. Data

`sources.jsonl`, one paper per line: `{"id","title","abstract","body","category","authors":[...]}`.
Use only openly licensed arXiv papers (check each paper's licence; CC BY / CC0 permit rewriting).

    cd gateway && pip install -r requirements.txt
    python build_benchmark.py paraphrase sources.jsonl texts.jsonl --models <model1> <model2> --base-url http://localhost:11434/v1
    python embed.py texts.jsonl embeddings.npz --model allenai/specter
    python build_benchmark.py pairs sources.jsonl texts.jsonl embeddings.npz pairs.jsonl

Record the exact LLM names and versions, prompts (in `build_benchmark.py`) and temperature in the paper.

## 3. Detection experiment

    python score_pairs.py pairs.jsonl texts.jsonl embeddings.npz scores.csv --bits 128 256 512
    python evaluate.py scores.csv results.json --target-fpr 0.01 \
        --method "Exact hash" exact --method "MinHash" minhash --method "TF-IDF" tfidf \
        --method "Raw embedding (not private)" emb_cos_h emb_cos_b \
        --method "SubmitChain-128" srp128_h srp128_b --method "SubmitChain-256" srp256_h srp256_b \
        --method "SubmitChain-512" srp512_h srp512_b
    python plots.py detection results.json fig5_detection

## 4. Thresholds for the ledger

`results.json` lists, for each method, the thresholds chosen on the validation split. Scores for
SubmitChain are negated Hamming distances, so the on-chain integer thresholds are
`tauH = floor(-threshold_h)` and `tauB = floor(-threshold_b)`.

## 5. Privacy experiment

    python inversion_attack.py texts.jsonl embeddings.npz leakage.json --view header --bits 128 256 512
    python plots.py privacy leakage.json fig7_privacy

The attack is a word-recovery attack (linear probe and MLP; the stronger is reported) trained
directly on the codes. A generative inversion attack (e.g. Vec2Text-style) is a stronger test and is
recommended as an additional experiment if GPU time allows.

## 6. System experiment

    npx caliper launch manager --caliper-workspace caliper --caliper-benchconfig benchconfig.yaml \
        --caliper-networkconfig <your network config> 

Collect mean latency and throughput per (organisations, mode, active submissions) into
`caliper.csv` with columns `orgs,mode,active,latency_ms,tps`, then

    python plots.py system caliper.csv fig6_system

## 7. Triage agent

    python triage.py summarise manuscript.json summary.json --model <model>
    python triage.py compare summary_a.json summary_b.json report.json --model <model>
    python triage.py evaluate flagged_pairs.jsonl summaries.jsonl triage_eval.json --model <model>

`flagged_pairs.jsonl` = the test pairs SubmitChain-256 flags (from `scores.csv`), with their labels.
The exchange of summaries through a Fabric private data collection (Section 2.5) is **not
implemented**; in the experiment both summaries are produced locally.

## Explorer portal (demo)

`docs/index.html` is the role-restricted submission explorer described in Section 2.6 of the paper: look up a
submission identifier and see what the public, a member editor and the author are each allowed to see. It is a
self-contained demonstration — the ledger runs inside the browser tab and resets on reload — so it can be opened
directly from the file or served with GitHub Pages (Settings → Pages → Branch `main`, folder `/docs`).

## Pilot evaluation (CPU only) — Sections 3.3–3.5 of the paper

`experiments/pilot/` holds the scripts and result files of the pilot benchmark.

**What is included:** corpus record identifiers, titles, years and links (`data/corpus_records.json`), the source/split
assignment, all pair lists, LSA embeddings, per-pair scores (`scores.csv`), detection results (`results.json`), leakage
results, index analyses and the projection matrices. `figs.py` redraws Figs 7 and 8 from these files, and
`gateway/evaluate.py` recomputes Table 4 from `scores.csv`.

**What is not included:** the abstract text of the 715 corpus papers and the light, medium and heavy rewrites derived
from it, because the abstracts are the publishers' copyrighted text. To rebuild them, retrieve the abstracts from the
links in `corpus_records.json`, save them as `data/corpus.json` (fields: id, file, title, abstract, year, authors, url),
and run `build_pilot.py`. Light and medium rewrites are regenerated deterministically by `rewrite.py`; the 64 heavy
rewrites (written by Claude Opus 5.5) are available from the authors on request.

```
cd experiments/pilot
python build_pilot.py        # needs data/corpus.json and data/heavy.jsonl
python ../../gateway/score_pairs.py data/pairs.jsonl data/texts.jsonl data/embeddings.npz data/scores.csv --bits 64 128 256 512 --matrix-dir data/matrices
python ../../gateway/evaluate.py data/scores.csv data/results.json --method "MinHash 5-shingle" minhash --method "SubmitChain-256" srp256_h
python figs.py
TABLES=128 TBITS=14 MODES=index node ../chaincode_bench.js 1000,10000,20000 200
```

Encoder: LSA (TF-IDF word and character n-grams, SVD to 256 dimensions), a CPU stand-in for SPECTER. For the full
evaluation, replace `data/embeddings.npz` with the output of `gateway/embed.py`.

## Citation

If you use this code, please cite the paper and the archived release: Shinde, S. C. and Malemath, V. S. (2026) SubmitChain v1.0.0. Zenodo. https://doi.org/10.5281/zenodo.24161026

## Licence

MIT (see `LICENSE`).
