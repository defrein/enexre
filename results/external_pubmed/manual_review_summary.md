# External PubMed Manual Review Summary

Tahap 16 was rerun on the same five PubMed abstracts outside BC5CDR
using frozen NER and RE checkpoints and the frozen decision threshold.

## Configuration

- Article source: `data/external_pubmed/articles.jsonl`
- Excluded BC5CDR PMIDs: 1500
- NER checkpoint: `checkpoints/ner/final_seed42_lr5e-5_bs8`
- RE checkpoint: `checkpoints/re/re_seed13_lr3e-5_bs8`
- Threshold: `0.70`
- Candidate granularity: repeated case-insensitive surface forms grouped
- Device: `cpu`

## Aggregate Result

| Metric | Initial run | Diagnostic rerun |
| --- | ---: | ---: |
| PubMed abstracts | 5 | 5 |
| Predicted entities | 77 | 77 |
| Candidate Chemical-Disease pairs | 96 | 34 |
| Predicted CID relations at threshold 0.70 | 0 | 4 |
| Maximum CID probability | 0.0082 | 0.9956 |

The initial implementation created one candidate per mention pair. The
BC5CDR RE training data instead creates one candidate per concept pair
and marks all mentions of both target concepts. The diagnostic rerun
groups repeated surface forms and marks all mentions in each group,
which better matches the training input structure without claiming full
concept normalization.

## Predicted CID Relations

| PMID | Chemical | Disease | RE probability |
| --- | --- | --- | ---: |
| 34278747 | aminoglycosides | hypomagnesemia | 0.9956 |
| 34278747 | cisplatin | hypomagnesemia | 0.9942 |
| 34278747 | cetuximab | hypomagnesemia | 0.9560 |
| 34278747 | amphotericin B | hypomagnesemia | 0.9484 |

The abstract explicitly lists these medications as linked to
hypomagnesemia, so all four predictions are textually plausible in
manual review. Magnesium and Mg remain separate surface groups, showing
that synonym and abbreviation normalization is still required.

## Interpretation

This rerun shows that candidate construction, rather than threshold
selection alone, caused the zero-prediction result. It is still not an
external performance evaluation because these five articles do not
have independent gold CID annotations. Three articles also produced no
Chemical entities, so the sample cannot support Precision, Recall, F1,
or broad PubMed generalization claims.
