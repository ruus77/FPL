# Eksperymenty Predykcji Punktów FPL — Modele Bazowe, FPLLoss i Meta-Modele

## 1. Uruchomienie eksperymentu
Cały potok eksperymentalny uruchamiany jest jednym poleceniem z katalogu głównego repozytorium:

```bash
PYTHONPATH=src python -m fpl_project.models.run_experiments
```

Ziarno losowe zostało ustalone na `SEED = 77` (`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`).

## 2. Podział danych i liczebności grup cenowych (`tier_counts.csv`)
- **Train (`2022/23`, `2023/24`, `2024/25`)**: `80618` obserwacji (posortowanych globalnie po `(season, gw, code)`).
  - Wewnętrzny chronologiczny zbiór wczesnego zatrzymania dla sieci (`MLP`, `Conv1D`): ostatnie 8 kolejek sezonu `2024/25` (`gw >= 31`, `6085` obserwacji); trening właściwy sieci na wcześniejszych kolejkach (`74533` obserwacji).
- **Valid (`2025/26`)**: `29338` obserwacji.
- **Test (`2026/27`, `is_played == True`)**: `6596` obserwacji (`10` kolejek).
- **Złączenie wewnętrzne (`inner join` po `(code, season, gw)`)**: `0` odrzuconych wierszy na `valid` oraz `0` na `test` (w `train_sub` dla `Conv1D` pominięto `19031` początkowych wierszy rozgrzewki okna `seq_len = 8`).

Progi cenowe na zbiorze `train`: $p_{\text{low}} = 4.5000$ mln £ (kwantyl `0.50`), $p_{\text{high}} = 6.9250$ mln £ (kwantyl `0.955`).

| Zbiór (`split`) | `budget` ($< 4.50$) | `mid` ($[4.50, 6.925)$) | `premium` ($\ge 6.925$) | Razem (`total`) |
| :--- | ---: | ---: | ---: | ---: |
| `train` | 29611 | 47376 | 3631 | 80618 |
| `valid` | 10748 | 17154 | 1436 | 29338 |
| `test` | 1763 | 4583 | 250 | 6596 |

## 3. Funkcja straty `FPLLoss` i stałe konfiguracyjne
Stałe funkcji straty zdefiniowane w `src/fpl_project/scripts/config.py`:
- `EPS_MIN` ($\varepsilon_{\min}$) = `0.1`
- `W_MID` ($w_{\text{mid}}$) = `1.25`
- `W_PREMIUM` ($w_{\text{premium}}$) = `2.0`

Wzór wagi obserwacji (gdzie $m_i = \text{clip}(\text{minutes\_ema\_3} / 90, 0, 1)$):
$$w_i = w_{\text{cena}, i} \cdot (\ln(1 + m_i) + \varepsilon_{\min}), \quad w_{\text{cena}, i} \in \{1.0, 1.25, 2.0\}$$

Ponieważ wagi $w_i$ nie są dzielone przez stałą normalizującą, wartości `FPLLoss` w tabelach są wyrażone w innej skali niż `MSE`.

## 4. Konfiguracja GLM (`glum`) i realizacja wariantu (B)
- **Przestrzeń przeszukiwania GLM (`GLM_PARAM_GRID`)**:
  - `alpha`: `stats.loguniform(1e-5, 0.01)`
  - `l1_ratio`: `stats.uniform(loc=0.0, scale=1.0)`
  - `family`: `["poisson", "negative.binomial", "tweedie", "normal"]`
  - `link`: `["auto", "log"]`
- **Domyślne parametry w bibliotece `glum` (`glum._glm.get_family` oraz `glum._glm.get_link`)**:
  - `family="negative.binomial"` (bez podanego parametru) przyjmuje $\theta = 1.0$ (`NegativeBinomialDistribution(1.0)`),
  - `family="tweedie"` (bez podanego parametru) przyjmuje moc $p = 1.5$ (`TweedieDistribution(1.5)`),
  - `link="auto"` zwraca dla poszczególnych rodzin:
    - `"poisson"` $\rightarrow$ `LogLink` (`log`),
    - `"negative.binomial"` $\rightarrow$ `LogLink` (`log`),
    - `"tweedie"` $\rightarrow$ `LogLink` (`log`),
    - `"normal"` $\rightarrow$ `IdentityLink` (`identity`).
- **Wybrane hiperparametry GLM w `RandomizedSearchCV`**:
  - Wariant `mse`: `{'alpha': np.float64(0.0032362333066577394), 'family': 'normal', 'l1_ratio': np.float64(0.923459926964004), 'link': 'log'}`
  - Wariant `fplloss`: `{'alpha': np.float64(0.005719092164999407), 'family': 'poisson', 'l1_ratio': np.float64(0.35337685610139535), 'link': 'log'}`
- **Wariant (B) dla GLM i XGBoost**:
  - **XGBoost (`objective="reg:squarederror"`)**: przekazanie `sample_weight = w_i` do `fit()` minimalizuje ważony błąd średniokwadratowy $\sum_i w_i (y_i - \hat{y}_i)^2$.
  - **GLM (`glum.GeneralizedLinearRegressor`)**: przekazanie `sample_weight = w_i` maksymalizuje ważoną log-wiarygodność wybranej rodziny rozkładu.
  - W `TimeSeriesSplit` CV (granice foldów pomiędzy kolejkami `(season, gw)`) wagi `w_i` przekazywane są do `estimator.fit()` oraz do ważonego kryterium oceny foldu (`neg_weighted_mse`).

## 5. Wyniki na zbiorze testowym (`metrics_test.csv`)
| Model | Wariant | Poziom | $p$ | MSE | MAE | $R^2$ | Adj $R^2$ | FPLLoss | MSE budget | MSE mid | MSE premium | MAE budget | MAE mid | MAE premium |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `glm` | `mse` | `base` | 194 | 4.6598 | 1.2309 | 0.2597 | 0.2372 | 3.2093 | 4.2467 | 4.3663 | 12.9532 | 1.0026 | 1.2394 | 2.6840 |
| `xgboost` | `mse` | `base` | 194 | 4.4525 | 1.1545 | 0.2926 | 0.2712 | 3.1484 | 3.8280 | 4.2560 | 12.4576 | 0.9276 | 1.1617 | 2.6224 |
| `mlp` | `mse` | `base` | 194 | 4.4597 | 1.1914 | 0.2914 | 0.2700 | 3.1829 | 3.7246 | 4.2694 | 13.1315 | 1.0126 | 1.1800 | 2.6611 |
| `conv1d` | `mse` | `base` | 52 | 4.4700 | 1.2369 | 0.2898 | 0.2842 | 3.1856 | 3.7429 | 4.2887 | 12.9215 | 1.0460 | 1.2346 | 2.6258 |
| `meta_nnls` | `mse` | `meta` | 4 | 4.3253 | 1.1982 | 0.3128 | 0.3124 | 3.0874 | 3.5923 | 4.1622 | 12.4848 | 0.9550 | 1.2095 | 2.7062 |
| `meta_context` | `mse` | `meta` | 44 | 4.3125 | 1.1918 | 0.3148 | 0.3102 | 3.0788 | 3.5749 | 4.1534 | 12.4320 | 0.9430 | 1.2079 | 2.6497 |
| `mean_ensemble` | `mse` | `meta` | 4 | 4.3881 | 1.1816 | 0.3028 | 0.3024 | 3.1304 | 3.6791 | 4.2127 | 12.6030 | 0.9614 | 1.1873 | 2.6304 |
| `best_single(xgboost)` | `mse` | `meta` | 194 | 4.4525 | 1.1545 | 0.2926 | 0.2712 | 3.1484 | 3.8280 | 4.2560 | 12.4576 | 0.9276 | 1.1617 | 2.6224 |
| `glm` | `fplloss` | `base` | 194 | 4.6705 | 1.2453 | 0.2579 | 0.2355 | 3.2058 | 4.2362 | 4.3910 | 12.8578 | 1.0134 | 1.2554 | 2.6963 |
| `xgboost` | `fplloss` | `base` | 194 | 4.4397 | 1.1918 | 0.2946 | 0.2732 | 3.1284 | 3.8407 | 4.2606 | 11.9473 | 0.9709 | 1.2010 | 2.5817 |
| `mlp` | `fplloss` | `base` | 194 | 4.5194 | 1.2003 | 0.2820 | 0.2602 | 3.2048 | 3.8586 | 4.3278 | 12.6913 | 0.9721 | 1.2099 | 2.6321 |
| `conv1d` | `fplloss` | `base` | 52 | 4.3982 | 1.1947 | 0.3012 | 0.2957 | 3.1385 | 3.6568 | 4.2244 | 12.8136 | 0.9646 | 1.2033 | 2.6598 |
| `meta_nnls` | `fplloss` | `meta` | 4 | 4.3307 | 1.2339 | 0.3119 | 0.3115 | 3.0758 | 3.6086 | 4.1824 | 12.1430 | 0.9822 | 1.2514 | 2.6878 |
| `meta_context` | `fplloss` | `meta` | 44 | 4.3134 | 1.1858 | 0.3147 | 0.3101 | 3.0636 | 3.6158 | 4.1603 | 12.0376 | 0.9128 | 1.2129 | 2.6139 |
| `mean_ensemble` | `fplloss` | `meta` | 4 | 4.4008 | 1.1904 | 0.3008 | 0.3004 | 3.1220 | 3.7232 | 4.2311 | 12.2913 | 0.9495 | 1.2049 | 2.6229 |
| `best_single(xgboost)` | `fplloss` | `meta` | 194 | 4.4397 | 1.1918 | 0.2946 | 0.2732 | 3.1284 | 3.8407 | 4.2606 | 11.9473 | 0.9709 | 1.2010 | 2.5817 |

## 6. Pliki wynikowe w katalogu `results/`
- `metrics_test.csv`, `metrics_valid.csv` — metryki na zbiorze testowym i walidacyjnym,
- `tier_counts.csv` — liczebności grup `budget`, `mid`, `premium` dla `train`, `valid`, `test`,
- `table_tiers.csv` — tabela MSE i MAE w podziale na grupy cenowe (zaokrąglona do 2 miejsc po przecinku),
- `meta_weights.csv` — wagi i wyrazy wolne wyuczonych meta-modeli,
- `preds_valid.parquet`, `preds_test.parquet` — predykcje wszystkich modeli na zbiorze walidacyjnym i testowym,
- `experiment_run.log` — pełny zapis przebiegu eksperymentu.
