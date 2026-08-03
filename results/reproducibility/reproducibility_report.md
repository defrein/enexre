# Reproducibility Report

## Environment

- Python: 3.10.0
- Platform: Windows-10-10.0.19045-SP0
- PyTorch: 2.12.1
- Transformers: 5.12.1
- Neo4j Python driver: 6.2.0

## Git

- Branch: re-best
- Commit: c1695af79910d26eae34bb04983b068b41506ac0
- Status: dirty

## Key Results

- NER test F1 mean: 0.8897
- RE gold test F1: 0.6802
- Pipeline NER-RE test F1: 0.6458
- Graph: 291 Chemical nodes, 319 Disease nodes, 944 CID relationships
- External PubMed: 5 abstracts, 96 candidates, 0 predicted CID relations

## Reproduction Commands

- `.venv/Scripts/python.exe scripts/validate_bc5cdr.py`
- `.venv/Scripts/python.exe scripts/build_ner_dataset.py`
- `.venv/Scripts/python.exe scripts/train_ner.py --smoke-test --cpu --cleanup-smoke-checkpoint`
- `python scripts/train_ner.py --seed 13 --learning-rate 5e-5 --batch-size 8 --run-name final_seed13_lr5e-5_bs8`
- `python scripts/train_ner.py --seed 42 --learning-rate 5e-5 --batch-size 8 --run-name final_seed42_lr5e-5_bs8`
- `python scripts/train_ner.py --seed 100 --learning-rate 5e-5 --batch-size 8 --run-name final_seed100_lr5e-5_bs8`
- `.venv/Scripts/python.exe scripts/build_re_dataset.py`
- `python scripts/train_re.py --seed 13 --learning-rate 3e-5 --batch-size 8 --run-name re_seed13_lr3e-5_bs8`
- `.venv/Scripts/python.exe scripts/evaluate_re.py --cpu`
- `.venv/Scripts/python.exe scripts/evaluate_pipeline.py --cpu`
- `.venv/Scripts/python.exe scripts/analyze_errors.py`
- `.venv/Scripts/python.exe scripts/build_graph.py`
- `.venv/Scripts/python.exe scripts/query_graph.py --pmid 18801087 --limit 5`
- `.venv/Scripts/python.exe scripts/run_external_pubmed.py --cpu --count 5`

## File Checksums

| File | Exists | SHA256 |
| --- | ---: | --- |
| `data/data_manifest.json` | True | `8a74900daeed0a3b183439cda8b0fdd08af23e290189ef5fcc28ebb8d6e1e716` |
| `configs/config_ner.yaml` | True | `2b47e61dce68a0658404348cbde5801cd871a732a70fcae2287f3a9b755ee19f` |
| `configs/config_re.yaml` | True | `603ee851dd0accefebfbe3154fefb7fcdb6f71b29f4edc98e2d07988d2e79074` |
| `requirements.txt` | True | `683a8480350c3c4d9bedc24fb5a3dbb082c53125b6670482d5d4d0d439b73776` |
| `requirements.lock.txt` | True | `afcdbebeaa1708afd04fb621f9d8a6cc47f3bffd32feef100a3bd6cb265de635` |
| `scripts/validate_bc5cdr.py` | True | `34b4780a58bcf0e78bf5e8b9af9d078165cb08fdae31ed8fe077764718b46db2` |
| `scripts/build_ner_dataset.py` | True | `65ab3cccd1b804acab871a166e7250d6324b961886e36bd9f789793a22fa83f4` |
| `scripts/train_ner.py` | True | `6e1ae7f586dc3a1b71f6472141277e4d0074189d6bf3f1daae4ad912df92ac2b` |
| `scripts/evaluate_ner.py` | True | `c5852c16eb9660f8781ef9656dcfd273dd9ce3cfdf2cef1876e784f6ca9b6ecc` |
| `scripts/build_re_dataset.py` | True | `e0fbd92250a5f23e57278648da92034f9d6fe1c205a773d3d73fc03ef42c95a3` |
| `scripts/train_re.py` | True | `7628bd0e26e2664beaab75964dd66f792b405b99b52d7178121bdbf54fa8bd66` |
| `scripts/evaluate_re.py` | True | `474412d08b436664b2b04bbc9adaedcd43dc13d868beab6bb3cbb25123928638` |
| `scripts/evaluate_pipeline.py` | True | `d6eff96584f80917be50bd35ade21f03ebf3d5c212986a253a28a7f8772edb47` |
| `scripts/analyze_errors.py` | True | `725903c8fa5849305f63d597aa199137f29a1b2531d077d7402d3347a5f407c2` |
| `scripts/build_graph.py` | True | `56c919e227aebb829936593997abe8d3cde724dfffc926297f2a3b100e7a6dd3` |
| `scripts/query_graph.py` | True | `79a211c857c201af811eb0916348b95e4d5eb15eb9afe16f1c841071ec63fd53` |
| `scripts/run_external_pubmed.py` | True | `125930f89466484e2454d21ffdf53ae7f3b0d6c1957b653f4cd5b80ac7a3417a` |
| `results/dataset_validation.json` | True | `ed3c0002669b06daba9e13609ca733f9eab05e5615a1a223da417fa3466ed663` |
| `results/ner_preprocessing_report.json` | True | `2f33f1cbad462f724ce8c8b579ca1fd7f35af13f29535781a4193527cf4e66a6` |
| `results/re_preprocessing_report.json` | True | `2d8d1305e1f73af69584e3ae46e08dd15665a3d37d0403da06ec320384c67228` |
| `results/ner/final_three_seed_test_summary.json` | True | `a2a806423bc44a56f0dc6cf488c2db4e22f6a771bcc8adbcc26b4451cb4c2b19` |
| `results/re/final_re_summary.json` | True | `e2fb28c090f2246c8e6411f33d4ea811fb09514a23deb86ea5144b88ec01abbb` |
| `results/pipeline/best_test_metrics.json` | True | `c8378aed770b1579ab55b7e2610885a02080a4fdb731d94f1d7d43a1af4c819b` |
| `results/error_analysis/error_analysis.json` | True | `80482461d60da9c1122bb56422a3f50a939f08d5f396410207f67898e29ea34c` |
| `results/graph/graph_validation.json` | True | `19700c16e922262085ae5390eb38d0d56111cea5024257e1a7f094ddcff3736a` |
| `results/graph/neo4j_runtime_validation.json` | True | `b18d137b879f87158e622bfdf783e3c5ebc981a6d9b07be03f9f216076e3d0e4` |
| `results/external_pubmed/external_pubmed_summary.json` | True | `e6ebad12b776424d0fc0e698f88eeff5f4b080d8ac7eee539e8fe804a07bc35f` |
| `results/external_pubmed/manual_review_summary.md` | True | `4cd50650ac51eb75473e0ab0b8513c578e8c0a7ef2846cb6d11b273f5449942d` |
