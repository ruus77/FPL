# Wyniki eksperymentu

Predykcja punktów FPL: cztery modele bazowe (GLM, XGBoost, MLP, Conv1D), każdy uczony z funkcją straty
**MSE** oraz **FPLLoss**, i meta-modele dopasowane na predykcjach ze zbioru walidacyjnego.
Wszystkie metryki dotyczą zbioru testowego (sezon 2026/27).

> Plik generowany automatycznie przez `run_experiments.py`. Zmiany wprowadzaj w
> `src/fpl_project/scripts/model_utils/pipeline/results_readme.py`.

```bash
PYTHONPATH=src python -m fpl_project.models.run_experiments
```

## Wyniki na zbiorze testowym

| Model | MSE (MSE) | MSE (FPLLoss) | MAE (MSE) | MAE (FPLLoss) | R² (MSE) | R² (FPLLoss) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM | 4.660 | 4.671 | 1.231 | 1.245 | 0.260 | 0.258 |
| XGBoost | 4.452 | 4.440 | **1.154** | 1.192 | 0.293 | 0.295 |
| MLP | 4.460 | 4.519 | 1.191 | 1.200 | 0.291 | 0.282 |
| Conv1D | 4.470 | 4.398 | 1.237 | 1.195 | 0.290 | 0.301 |
| NNLS (meta) | 4.325 | 4.331 | 1.198 | 1.234 | 0.313 | 0.312 |
| Ridge z kontekstem (meta) | **4.313** | **4.313** | 1.192 | **1.186** | **0.315** | **0.315** |
| Średnia (meta) | 4.388 | 4.401 | 1.182 | 1.190 | 0.303 | 0.301 |
| Najlepszy pojedynczy | 4.452 | 4.440 | **1.154** | 1.192 | 0.293 | 0.295 |

Pogrubienie oznacza najlepszy wynik w kolumnie. „Najlepszy pojedynczy” to model bazowy o najniższym błędzie na zbiorze walidacyjnym (MSE: XGBoost, FPLLoss: XGBoost).

## Błąd w grupach cenowych

Grupy według ceny zawodnika (`value_ema_3`): **budget** < 4.50 mln £ ≤ **mid** < 6.925 mln £ ≤ **premium**.

| Model | Wariant | MSE budget | MSE mid | MSE premium | MAE budget | MAE mid | MAE premium |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM | MSE | 4.25 | 4.37 | 12.95 | 1.00 | 1.24 | 2.68 |
| XGBoost | MSE | 3.83 | 4.26 | 12.46 | 0.93 | 1.16 | 2.62 |
| MLP | MSE | 3.72 | 4.27 | 13.13 | 1.01 | 1.18 | 2.66 |
| Conv1D | MSE | 3.74 | 4.29 | 12.92 | 1.05 | 1.23 | 2.63 |
| NNLS (meta) | MSE | 3.59 | 4.16 | 12.48 | 0.96 | 1.21 | 2.71 |
| Ridge z kontekstem (meta) | MSE | 3.57 | 4.15 | 12.43 | 0.94 | 1.21 | 2.65 |
| Średnia (meta) | MSE | 3.68 | 4.21 | 12.60 | 0.96 | 1.19 | 2.63 |
| Najlepszy pojedynczy | MSE | 3.83 | 4.26 | 12.46 | 0.93 | 1.16 | 2.62 |
| GLM | FPLLoss | 4.24 | 4.39 | 12.86 | 1.01 | 1.26 | 2.70 |
| XGBoost | FPLLoss | 3.84 | 4.26 | 11.95 | 0.97 | 1.20 | 2.58 |
| MLP | FPLLoss | 3.86 | 4.33 | 12.69 | 0.97 | 1.21 | 2.63 |
| Conv1D | FPLLoss | 3.66 | 4.22 | 12.81 | 0.96 | 1.20 | 2.66 |
| NNLS (meta) | FPLLoss | 3.61 | 4.18 | 12.14 | 0.98 | 1.25 | 2.69 |
| Ridge z kontekstem (meta) | FPLLoss | 3.62 | 4.16 | 12.04 | 0.91 | 1.21 | 2.61 |
| Średnia (meta) | FPLLoss | 3.72 | 4.23 | 12.29 | 0.95 | 1.20 | 2.62 |
| Najlepszy pojedynczy | FPLLoss | 3.84 | 4.26 | 11.95 | 0.97 | 1.20 | 2.58 |

## Dane

| Zbiór | Sezony | Obserwacje | budget | mid | premium |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `train` | 2022/23 – 2024/25 | 80 618 | 29 611 | 47 376 | 3 631 |
| `valid` | 2025/26 | 29 338 | 10 748 | 17 154 | 1 436 |
| `test` | 2026/27 (kolejki 1–10) | 6 596 | 1 763 | 4 583 | 250 |

- Progi cenowe to kwantyle 0.50 i 0.955 ceny na zbiorze treningowym.
- Sieci (MLP, Conv1D) uczą się na 74 533 obserwacjach; ostatnie kolejki sezonu 2024/25
  (`gw ≥ 31`, 6 085 obserwacji) służą do wczesnego zatrzymania.
- Conv1D pomija 19 031 początkowych obserwacji, dla których nie ma pełnego okna 8 kolejek.
- Po złączeniu predykcji wszystkich modeli odrzucono 0 wierszy na `valid` i 0 na `test`.

## Funkcja straty FPLLoss

$$w_i = w_{\text{cena},i} \cdot \left(\ln(1 + m_i) + \varepsilon_{\min}\right), \qquad m_i = \text{clip}\left(\tfrac{\text{minutes\_ema\_3}}{90}, 0, 1\right)$$

| Stała | Wartość |
| :--- | ---: |
| waga `budget` | 1.0 |
| waga `mid` (`W_MID`) | 1.25 |
| waga `premium` (`W_PREMIUM`) | 2.0 |
| $\varepsilon_{\min}$ (`EPS_MIN`) | 0.1 |

Wagi nie są normalizowane, dlatego wartości FPLLoss mają inną skalę niż MSE. Modele tabelaryczne
otrzymują wagi jako `sample_weight` w treningu i w ocenie foldów walidacji krzyżowej.

## Hiperparametry

Strojenie: `RandomizedSearchCV` (20 losowań, 5 foldów `TimeSeriesSplit` z granicami
między kolejkami). Przestrzenie przeszukiwania: `GLM_PARAM_GRID` i `XGB_PARAM_GRID` w `scripts/config.py`.

**GLM**

| Parametr | MSE | FPLLoss |
| :--- | :--- | :--- |
| `alpha` | 0.0032 | 0.0057 |
| `family` | `normal` | `poisson` |
| `l1_ratio` | 0.9235 | 0.3534 |
| `link` | `log` | `log` |

**XGBoost**

| Parametr | MSE | FPLLoss |
| :--- | :--- | :--- |
| `colsample_bytree` | 0.6658 | 0.6071 |
| `gamma` | 0.6578 | 0.0526 |
| `learning_rate` | 0.038 | 0.018 |
| `max_depth` | 3 | 6 |
| `min_child_weight` | 9 | 9 |
| `n_estimators` | 159 | 192 |
| `reg_alpha` | 1.94 | 1.735 |
| `reg_lambda` | 1.915 | 2.416 |
| `subsample` | 0.92 | 0.7996 |

**Sieci neuronowe**

| Parametr | MLP | Conv1D |
| :--- | ---: | ---: |
| warstwy ukryte | `[128, 64]` | – |
| kanały konwolucji | – | `[128, 64]` |
| rozmiar jądra | – | 3 |
| długość okna | – | 8 |
| dropout | 0.2 | 0.2 |
| learning rate | 0.001 | 0.001 |
| weight decay | 0.005 | 0.005 |
| batch size | 128 | 128 |
| liczba epok treningu (MSE) | 13 | 15 |
| liczba epok treningu (FPLLoss) | 17 | 15 |

## Pliki

| Plik | Zawartość |
| :--- | :--- |
| `metrics_test.csv`, `metrics_valid.csv` | wszystkie metryki na zbiorze testowym i walidacyjnym |
| `table_tiers.csv` | MSE i MAE w grupach cenowych (2 miejsca po przecinku) |
| `tier_counts.csv` | liczebności grup cenowych w zbiorach |
| `meta_weights.csv` | wagi i wyrazy wolne meta-modeli |
| `preds_valid.parquet`, `preds_test.parquet` | predykcje wszystkich modeli |
| `preds_predict.parquet` | predykcje na nierozegrane mecze (gdy dostępne są kursy) |
| `../trained_models/` | zapisane modele bazowe i preprocessing |

Ziarno losowe: `SEED = 77`.
