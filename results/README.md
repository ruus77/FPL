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
- Stała normalizująca $c$ (`norm_const` $= \text{mean}(\ln(1 + m_i) + \varepsilon_{\min})$ na `train`, gdzie $m_i = \text{clip}(\text{minutes\_ema\_3} / 90, 0, 1)$): **`0.3169`**

Wzór wagi obserwacji:
$$w_i = \frac{w_{\text{cena}, i} \cdot (\ln(1 + m_i) + \varepsilon_{\min})}{c}, \quad w_{\text{cena}, i} \in \{1.0, 1.25, 2.0\}, \quad c = 0.3169$$

## 4. Realizacja wariantu (B) dla GLM i XGBoost
- **XGBoost (`objective="reg:squarederror"`)**: przekazanie `sample_weight = w_i` do `fit()` minimalizuje ważony błąd średniokwadratowy $\sum_i w_i (y_i - \hat{y}_i)^2$.
- **GLM (`glum.GeneralizedLinearRegressor`)**: przekazanie `sample_weight = w_i` maksymalizuje ważoną log-wiarygodność wybranej rodziny rozkładu (Poisson, Negative Binomial ze strojonym $\theta$, Tweedie z mocą $p \in [1.1, 1.9]$) — analog `FPLLoss`.
- W `TimeSeriesSplit` CV (granice foldów pomiędzy kolejkami `(season, gw)`) wagi `w_i` przekazywane są do `estimator.fit()` oraz do ważonego kryterium oceny foldu (`neg_weighted_mse`).

## 5. Pliki wynikowe w katalogu `results/`
- `metrics_test.csv`, `metrics_valid.csv` — metryki na zbiorze testowym i walidacyjnym,
- `tier_counts.csv` — liczebności grup `budget`, `mid`, `premium` dla `train`, `valid`, `test`,
- `table_tiers.csv` — tabela MSE i MAE w podziale na grupy cenowe (zaokrąglona do 2 miejsc po przecinku),
- `meta_weights.csv` — wagi i wyrazy wolne wyuczonych meta-modeli,
- `preds_valid.parquet`, `preds_test.parquet` — predykcje wszystkich modeli na zbiorze walidacyjnym i testowym,
- `experiment_run.log` — pełny zapis przebiegu eksperymentu.
