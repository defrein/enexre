# Error Analysis Summary

## Aggregate Counts

| Evaluation | Candidates | FP | FN | FN missing candidate | FN rejected candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| RE with gold entities | 5405 | 364 | 329 | 0 | 329 |
| Pipeline NER-RE | 4401 | 295 | 417 | 117 | 300 |

## Pipeline Missing-Candidate Breakdown

- chemical_and_disease_missing: 18
- disease_missing: 69
- chemical_missing: 30

## Pipeline False Positive Examples

- PMID 2400986 D002066 -> D020258 p=0.9978; chem=['busulfan', 'Busulfan']; disease=['neurotoxicity', 'neurotoxic']
- PMID 3339945 D013806 -> D012640 p=0.9978; chem=['Theophylline', 'theophylline']; disease=['seizures', 'seizures']
- PMID 2553470 D008094 -> D012640 p=0.9978; chem=['lithium', 'lithium']; disease=['seizure', 'seizures']
- PMID 18801087 D000638 -> D011507 p=0.9978; chem=['Amiodarone', 'Amiodarone']; disease=['proteinuria']
- PMID 1420650 D002220 -> D020820 p=0.9978; chem=['carbamazepine', 'carbamazepine']; disease=['Asterixis', 'asterixis']

## Pipeline False Negative Examples

- PMID 24055495 D005947 -> D000544 p=0.0005; reason=Gold CID candidate was formed, but RE probability was below threshold.
- PMID 8667442 D009853 -> D010437 p=0.0005; reason=Gold CID candidate was formed, but RE probability was below threshold.
- PMID 8667442 D013392 -> D010437 p=0.0005; reason=Gold CID candidate was formed, but RE probability was below threshold.
- PMID 10219427 D003287 -> D003875 p=missing; reason=Gold CID pair was not evaluated because NER-derived candidate was missing.
- PMID 10219427 D003287 -> D004417 p=missing; reason=Gold CID pair was not evaluated because NER-derived candidate was missing.
- PMID 10219427 D003287 -> D010146 p=missing; reason=Gold CID pair was not evaluated because NER-derived candidate was missing.
