# Ampliseq Re-ranking — Ecological Coherence Report

**Samples with detections:** 106  
**Samples with rank changes:** 0  
**Species:** 4

## Interaction data sources
- **BiotXplorer** (primary): text-mined interactions from MEDLINE/PMC via SiBILS
- **GloBI** (fallback): structured interaction database (4.2M species pairs)

## Species rank changes

| Species | Avg Δrank | Samples boosted | Avg BiotXplorer score |
|---|---|---|---|
| autographa gamma | 0.0 | 0/1 | 0.0 |
| castanea mollissima | 0.0 | 0/68 | 0.0 |
| angiosperm mycorrhizal | 0.0 | 0/43 | 0.0 |
| quercus variabilis | 0.0 | 0/42 | 0.0 |

## Interpretation
- **Δrank > 0**: species moved UP (interaction partners co-detected in same sample)
- **Δrank < 0**: species moved DOWN (relatively less community support)
- Score combines BiotXplorer text-mined evidence + GloBI database pairs
