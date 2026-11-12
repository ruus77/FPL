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
- **Test (`2026/27`, `is_played == True`)**: `4258` obserwacji (`7` kolejek).
- **Złączenie wewnętrzne (`inner join` po `(code, season, gw)`)**: `0` odrzuconych wierszy na `valid` oraz `0` na `test` (w `train_sub` dla `Conv1D` pominięto `19031` początkowych wierszy rozgrzewki okna `seq_len = 8`).

Progi cenowe na zbiorze `train`: $p_{\text{low}} = 4.5000$ mln £ (kwantyl `0.50`), $p_{\text{high}} = 6.9250$ mln £ (kwantyl `0.955`).

| Zbiór (`split`) | `budget` ($< 4.50$) | `mid` ($[4.50, 6.925)$) | `premium` ($\ge 6.925$) | Razem (`total`) |
| :--- | ---: | ---: | ---: | ---: |
| `train` | 29611 | 47376 | 3631 | 80618 |
| `valid` | 10748 | 17154 | 1436 | 29338 |
| `test` | 1239 | 2857 | 162 | 4258 |

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
| `glm` | `mse` | `base` | 219 | 4.9749 | 1.2715 | 0.2355 | 0.1940 | 3.2507 | 5.2160 | 4.3769 | 13.6784 | 1.1693 | 1.2361 | 2.6765 |
| `xgboost` | `mse` | `base` | 219 | 4.7574 | 1.1939 | 0.2689 | 0.2292 | 3.1449 | 4.9715 | 4.2269 | 12.4763 | 1.1135 | 1.1528 | 2.5340 |
| `mlp` | `mse` | `base` | 219 | 4.9194 | 1.1711 | 0.2440 | 0.2030 | 3.2968 | 4.9566 | 4.3861 | 14.0409 | 1.0981 | 1.1224 | 2.5877 |
| `conv1d` | `mse` | `base` | 39 | 5.5094 | 1.2510 | 0.1533 | 0.1455 | 4.0156 | 4.7993 | 5.0015 | 19.8986 | 1.1070 | 1.2241 | 2.8266 |
| `meta_nnls` | `mse` | `meta` | 4 | 4.6830 | 1.2273 | 0.2803 | 0.2796 | 3.1192 | 4.7756 | 4.1907 | 12.6583 | 1.1176 | 1.1954 | 2.6298 |
| `meta_context` | `mse` | `meta` | 44 | 4.6603 | 1.2176 | 0.2838 | 0.2763 | 3.1139 | 4.6996 | 4.1826 | 12.7839 | 1.1122 | 1.1841 | 2.6150 |
| `mean_ensemble` | `mse` | `meta` | 4 | 4.8721 | 1.1982 | 0.2513 | 0.2506 | 3.2872 | 4.9011 | 4.3473 | 13.9061 | 1.1062 | 1.1602 | 2.5710 |
| `best_single(xgboost)` | `mse` | `meta` | 219 | 4.7574 | 1.1939 | 0.2689 | 0.2292 | 3.1449 | 4.9715 | 4.2269 | 12.4763 | 1.1135 | 1.1528 | 2.5340 |
| `glm` | `fplloss` | `base` | 219 | 5.0176 | 1.2808 | 0.2289 | 0.1871 | 3.2481 | 5.2925 | 4.4163 | 13.5180 | 1.1685 | 1.2498 | 2.6876 |
| `xgboost` | `fplloss` | `base` | 219 | 4.7516 | 1.2276 | 0.2698 | 0.2302 | 3.1278 | 4.9650 | 4.2300 | 12.3173 | 1.1458 | 1.1894 | 2.5262 |
| `mlp` | `fplloss` | `base` | 219 | 4.9171 | 1.2227 | 0.2444 | 0.2034 | 3.2665 | 5.0503 | 4.3624 | 13.6808 | 1.1306 | 1.1839 | 2.6115 |
| `conv1d` | `fplloss` | `base` | 39 | 5.3448 | 1.3315 | 0.1786 | 0.1710 | 3.7930 | 4.7136 | 4.8177 | 19.4697 | 1.1903 | 1.3038 | 2.8991 |
| `meta_nnls` | `fplloss` | `meta` | 4 | 4.6874 | 1.2463 | 0.2797 | 0.2790 | 3.1100 | 4.7801 | 4.2000 | 12.5731 | 1.1159 | 1.2244 | 2.6303 |
| `meta_context` | `fplloss` | `meta` | 44 | 4.7168 | 1.2251 | 0.2751 | 0.2676 | 3.1115 | 4.8809 | 4.1893 | 12.7647 | 1.0844 | 1.2071 | 2.6179 |
| `mean_ensemble` | `fplloss` | `meta` | 4 | 4.8280 | 1.2374 | 0.2581 | 0.2574 | 3.2385 | 4.8605 | 4.3150 | 13.6260 | 1.1271 | 1.2081 | 2.5976 |
| `best_single(xgboost)` | `fplloss` | `meta` | 219 | 4.7516 | 1.2276 | 0.2698 | 0.2302 | 3.1278 | 4.9650 | 4.2300 | 12.3173 | 1.1458 | 1.1894 | 2.5262 |

## 6. Pliki wynikowe w katalogu `results/`
- `metrics_test.csv`, `metrics_valid.csv` — metryki na zbiorze testowym i walidacyjnym,
- `tier_counts.csv` — liczebności grup `budget`, `mid`, `premium` dla `train`, `valid`, `test`,
- `table_tiers.csv` — tabela MSE i MAE w podziale na grupy cenowe (zaokrąglona do 2 miejsc po przecinku),
- `meta_weights.csv` — wagi i wyrazy wolne wyuczonych meta-modeli,
- `preds_valid.parquet`, `preds_test.parquet` — predykcje wszystkich modeli na zbiorze walidacyjnym i testowym,
- `experiment_run.log` — pełny zapis przebiegu eksperymentu.
