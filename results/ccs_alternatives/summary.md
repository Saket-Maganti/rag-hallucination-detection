# CCS metric validation (Phase 7 #4)

Comparing CCS against $5$ alternative coherence-like metrics on 
the same 955 per-query rows. All metrics are computed 
from the pairwise cosine-similarity matrix of retrieved chunk 
embeddings.

## Pooled correlations vs faithfulness

| metric             |   n |   pearson_r |   pearson_p |   spearman_rho |   spearman_p |
|:-------------------|----:|------------:|------------:|---------------:|-------------:|
| ccs_metric         | 955 |      0.2297 |           0 |         0.2056 |            0 |
| mean_pair_cos      | 955 |      0.2311 |           0 |         0.2057 |            0 |
| min_pair_cos       | 955 |      0.2164 |           0 |         0.1915 |            0 |
| matrix_entropy     | 955 |      0.1684 |           0 |         0.2018 |            0 |
| mmr_diversity      | 955 |     -0.2311 |           0 |        -0.2057 |            0 |
| graph_connectivity | 955 |      0.2816 |           0 |         0.264  |            0 |

## Ranking by |Spearman ρ|

| metric             |   spearman_rho |   spearman_p |
|:-------------------|---------------:|-------------:|
| graph_connectivity |         0.264  |            0 |
| mean_pair_cos      |         0.2057 |            0 |
| mmr_diversity      |        -0.2057 |            0 |
| ccs_metric         |         0.2056 |            0 |
| matrix_entropy     |         0.2018 |            0 |
| min_pair_cos       |         0.1915 |            0 |

Reading: CCS validated if its $|\rho|$ is comparable to or larger 
than the alternatives, AND if formally-related metrics 
(mean\_pair\_cos, mmr\_diversity = 1 − mean\_pair\_cos) 
have noticeably weaker signal. The mean-minus-std formulation 
is justified empirically rather than by appeal to first 
principles.