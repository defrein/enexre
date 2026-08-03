# External PubMed Manual Review Summary

Tahap 16 was run on five PubMed abstracts outside BC5CDR using the frozen
NER checkpoint, RE checkpoint, and threshold.

## Configuration

- Query: `("drug-induced"[Title/Abstract] OR "adverse effect"[Title/Abstract]) AND (disease[Title/Abstract] OR toxicity[Title/Abstract]) AND 2020:2026[pdat]`
- Excluded BC5CDR PMIDs: 1500
- NER checkpoint: `checkpoints/ner/final_seed42_lr5e-5_bs8`
- RE checkpoint: `checkpoints/re/re_seed13_lr3e-5_bs8`
- Threshold: `0.70`
- Device: `cpu`

## Aggregate Result

| Metric | Value |
| --- | ---: |
| PubMed abstracts | 5 |
| Predicted entities | 77 |
| Predicted Chemical mentions | 9 |
| Predicted Disease mentions | 68 |
| Candidate Chemical-Disease pairs | 96 |
| Predicted CID relations at threshold 0.70 | 0 |

## Highest-Scored Candidate Pairs

| PMID | Chemical | Disease | RE probability |
| --- | --- | --- | ---: |
| 34278747 | cetuximab | Hypomagnesemia | 0.0082 |
| 34278747 | cisplatin | Hypomagnesemia | 0.0073 |
| 34278747 | aminoglycosides | Hypomagnesemia | 0.0048 |
| 35332071 | glucocorticoids | Drug-induced ILD | 0.0035 |
| 34278747 | Mg | Hypomagnesemia | 0.0034 |
| 34278747 | Magnesium | Hypomagnesemia | 0.0030 |
| 34278747 | amphotericin B | Hypomagnesemia | 0.0026 |

## Interpretation

The model did not predict any external CID relation with the frozen threshold.
This result should be reported as an early external-generalization finding,
not treated as a failure of the gold-entity BC5CDR evaluation.

Likely factors:

- External abstracts are review-style texts rather than BC5CDR-style annotated cases.
- External data does not include MeSH concept IDs, so this prototype evaluates mention pairs.
- The final RE threshold was selected on BC5CDR development data and is conservative on this sample.
- Some high-ranked pairs, such as `cisplatin -> Hypomagnesemia`, are plausible candidates for manual review even though they remain below threshold.
