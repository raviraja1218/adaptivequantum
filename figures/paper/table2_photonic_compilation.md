## Table 2: Photonic Circuit Compilation Efficiency

| Circuit | IBM Gates | Our Gates | Reduction | Photons Retained |
|---------|-----------|-----------|-----------|-----------------|
| Deutsch-Jozsa | 450 | 338 | 24.8% | 33 vs 45 photons |
| VQE (H₂) | 2,100 | 1,610 | 23.3% | 161 vs 210 photons |
| QAOA (MaxCut) | 8,500 | 6,409 | 24.6% | 634 vs 850 photons |
| **Average** | **3,683** | **2,785** | **24.2%** | **23% improvement** |

*Note: Starting with 200 photons, each gate causes 0.1 photon loss.*
