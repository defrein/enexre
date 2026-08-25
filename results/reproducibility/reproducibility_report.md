# Reproducibility Report

## Environment

- Python: 3.10.0
- Platform: Windows-10-10.0.19045-SP0
- PyTorch: 2.12.1
- Transformers: 5.12.1
- Checkpoint Transformers metadata: NER=5.13.1, RE=5.13.1
- Neo4j Python driver: 6.2.0

## Git

- Branch: master
- Commit: 5e13d9c7068b8f6d6f69294cce1cc3662aa1d1b6
- Status: clean

## Key Results

- NER test F1 mean: 0.8897
- RE gold test F1: 0.6802
- Pipeline NER-RE test F1: 0.6458
- Graph: 291 Chemical nodes, 319 Disease nodes, 944 CID relationships
- External PubMed: 5 abstracts, 113 candidates, 15 predicted CID relations, granularity=surface, no gold labels

## Reproduction Commands

- `.venv/Scripts/python.exe scripts/validate_bc5cdr.py`
- `.venv/Scripts/python.exe scripts/build_ner_dataset.py`
- `.venv/Scripts/python.exe scripts/train_ner.py --smoke-test --cpu --cleanup-smoke-checkpoint`
- `python scripts/train_ner.py --seed 13 --learning-rate 5e-5 --batch-size 8 --run-name final_seed13_lr5e-5_bs8`
- `python scripts/train_ner.py --seed 42 --learning-rate 5e-5 --batch-size 8 --run-name final_seed42_lr5e-5_bs8`
- `python scripts/train_ner.py --seed 100 --learning-rate 5e-5 --batch-size 8 --run-name final_seed100_lr5e-5_bs8`
- `.venv/Scripts/python.exe scripts/build_re_dataset.py`
- `python scripts/train_re.py --seed 13 --learning-rate 3e-5 --batch-size 8 --run-name re_seed13_lr3e-5_bs8`
- `python scripts/train_re.py --seed 42 --learning-rate 3e-5 --batch-size 8 --run-name re_seed42_lr3e-5_bs8`
- `python scripts/train_re.py --seed 100 --learning-rate 3e-5 --batch-size 8 --run-name re_seed100_lr3e-5_bs8`
- `.venv/Scripts/python.exe scripts/evaluate_re.py --cpu`
- `.venv/Scripts/python.exe scripts/evaluate_pipeline.py --cpu`
- `.venv/Scripts/python.exe scripts/analyze_errors.py`
- `.venv/Scripts/python.exe scripts/build_graph.py`
- `.venv/Scripts/python.exe scripts/query_graph.py --pmid 18801087 --limit 5`
- `.venv/Scripts/python.exe scripts/prototype_app.py --host 127.0.0.1 --port 8000`
- `.venv/Scripts/python.exe scripts/run_external_pubmed.py --articles-input data/external_pubmed/articles.jsonl --count 5 --candidate-granularity surface --cpu`

## File Checksums

| File | Exists | SHA256 |
| --- | ---: | --- |
| `data/data_manifest.json` | True | `8a74900daeed0a3b183439cda8b0fdd08af23e290189ef5fcc28ebb8d6e1e716` |
| `configs/config_ner.yaml` | True | `2b47e61dce68a0658404348cbde5801cd871a732a70fcae2287f3a9b755ee19f` |
| `configs/config_re.yaml` | True | `603ee851dd0accefebfbe3154fefb7fcdb6f71b29f4edc98e2d07988d2e79074` |
| `requirements.txt` | True | `e4ab3ee33553e5c26caf6017a79722131a0477f8003be1be85650d370c782adc` |
| `requirements.lock.txt` | True | `cd501fbe8749ddf3754b4aff8132e86c57dcfa6c7c40f5aab99db7c94ad33d7d` |
| `scripts/validate_bc5cdr.py` | True | `34b4780a58bcf0e78bf5e8b9af9d078165cb08fdae31ed8fe077764718b46db2` |
| `scripts/build_ner_dataset.py` | True | `65ab3cccd1b804acab871a166e7250d6324b961886e36bd9f789793a22fa83f4` |
| `scripts/train_ner.py` | True | `6e1ae7f586dc3a1b71f6472141277e4d0074189d6bf3f1daae4ad912df92ac2b` |
| `scripts/evaluate_ner.py` | True | `c5852c16eb9660f8781ef9656dcfd273dd9ce3cfdf2cef1876e784f6ca9b6ecc` |
| `scripts/build_re_dataset.py` | True | `e0fbd92250a5f23e57278648da92034f9d6fe1c205a773d3d73fc03ef42c95a3` |
| `scripts/train_re.py` | True | `7628bd0e26e2664beaab75964dd66f792b405b99b52d7178121bdbf54fa8bd66` |
| `scripts/evaluate_re.py` | True | `8567991b6116f5cf01d1ca18324ec61ce3f92155bd712695ec3a889833a0ced6` |
| `scripts/evaluate_pipeline.py` | True | `3cc656a6fd0e9e602074f2a222b5d23d09c3a10eb0486d2c8916b744b1d5984b` |
| `scripts/analyze_errors.py` | True | `e4396455afcdbb37e23b58d16e4ce6752c5ea10c78515b0068015e9bd5875764` |
| `scripts/build_graph.py` | True | `bd346c2c1811a31f99d8cca4d1a6d4c4a1643889b0803d4e7ef92236477021ae` |
| `scripts/query_graph.py` | True | `a25f79ad51fd8e96af7fb8d56ae1cea6f1ace65951a8f7f003e93e2a495acb9c` |
| `scripts/prototype_app.py` | True | `1261a6625cd2fba816a6bf12888176010c34ffa2dfd46d33d567f135f1b684a1` |
| `scripts/run_external_pubmed.py` | True | `f99d8e7b817969e03b31a4c7476b3783e2d3b0a768a5d31b5dd5383c27fabb34` |
| `results/dataset_validation.json` | True | `ed3c0002669b06daba9e13609ca733f9eab05e5615a1a223da417fa3466ed663` |
| `results/ner_preprocessing_report.json` | True | `2f33f1cbad462f724ce8c8b579ca1fd7f35af13f29535781a4193527cf4e66a6` |
| `results/re_preprocessing_report.json` | True | `2d8d1305e1f73af69584e3ae46e08dd15665a3d37d0403da06ec320384c67228` |
| `results/ner/final_three_seed_test_summary.json` | True | `a2a806423bc44a56f0dc6cf488c2db4e22f6a771bcc8adbcc26b4451cb4c2b19` |
| `results/re/final_re_summary.json` | True | `6cf315db4919ce942525d88a455365b35aca16e8764989bffa75186404027325` |
| `results/pipeline/best_test_metrics.json` | True | `c8378aed770b1579ab55b7e2610885a02080a4fdb731d94f1d7d43a1af4c819b` |
| `results/error_analysis/error_analysis.json` | True | `80482461d60da9c1122bb56422a3f50a939f08d5f396410207f67898e29ea34c` |
| `results/graph/graph_validation.json` | True | `19700c16e922262085ae5390eb38d0d56111cea5024257e1a7f094ddcff3736a` |
| `results/graph/neo4j_runtime_validation.json` | True | `5a2ad50241684ad13720d457dfe52c22a6a9bcfbd32aa5990a34bc69521d3975` |
| `results/external_pubmed/external_pubmed_summary.json` | True | `d560088e0bb110caf0a611a71fea0d2f303e44748766b71d2ee7cd15246961c5` |
| `results/external_pubmed/manual_review_summary.md` | True | `5d00b6d87161c447394952fcd658512124c6b6c11cbe1b5dcc875734c9f6e72c` |
