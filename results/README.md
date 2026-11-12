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

## 5. Metodologia bootstrapu sparowanego i przedziały ufności (95% CI)
- **Procedura**: bootstrap sparowany blokowy na poziomie całych kolejek (`gw`) zbioru testowego (`7` kolejek, `n = 4258` obserwacji).
- **Parametry**: `2000` powtórzeń losowania ze zwracaniem całych kolejek (`gw`), ziarno generatora `SEED = 77`, identyczna próba bootstrapowa dla porównywanych wariantów i modeli.
- **Przedział ufności**: metoda percentylowa (`2.5%` i `97.5%` rozkładu bootstrapowego różnicy).

### 5.1. Porównanie wariantów `FPLLoss − MSE` (`variant_comparison.csv`)
Przypadki (model, grupa, metryka), w których 95% przedział ufności dla różnicy $\text{Metryka}(\text{fplloss}) - \text{Metryka}(\text{mse})$ **nie obejmuje zera**:
- `glm` (grupa `all`, `MSE`, n=4258): różnica = `-0.0693`, 95% CI = `[-0.1265, -0.0208]`
- `glm` (grupa `all`, `MAE`, n=4258): różnica = `+0.0701`, 95% CI = `[+0.0457, +0.0879]`
- `glm` (grupa `budget`, `MAE`, n=1239): różnica = `+0.0722`, 95% CI = `[+0.0267, +0.1529]`
- `glm` (grupa `mid`, `MAE`, n=2857): różnica = `+0.0762`, 95% CI = `[+0.0681, +0.0846]`
- `glm` (grupa `premium`, `MSE`, n=162): różnica = `-1.2389`, 95% CI = `[-2.9958, -0.0526]`
- `xgboost` (grupa `all`, `MAE`, n=4258): różnica = `+0.0389`, 95% CI = `[+0.0170, +0.0540]`
- `xgboost` (grupa `mid`, `MAE`, n=2857): różnica = `+0.0458`, 95% CI = `[+0.0405, +0.0498]`
- `xgboost` (grupa `premium`, `MAE`, n=162): różnica = `+0.0533`, 95% CI = `[+0.0027, +0.1018]`
- `mlp` (grupa `all`, `MAE`, n=4258): różnica = `+0.0774`, 95% CI = `[+0.0411, +0.1017]`
- `mlp` (grupa `budget`, `MAE`, n=1239): różnica = `+0.0508`, 95% CI = `[+0.0038, +0.1359]`
- `mlp` (grupa `mid`, `MAE`, n=2857): różnica = `+0.0903`, 95% CI = `[+0.0757, +0.1015]`
- `mlp` (grupa `premium`, `MSE`, n=162): różnica = `-0.4532`, 95% CI = `[-0.9397, -0.0016]`
- `conv1d` (grupa `all`, `MAE`, n=4258): różnica = `+0.0811`, 95% CI = `[+0.0657, +0.0973]`
- `conv1d` (grupa `budget`, `MAE`, n=1239): różnica = `+0.0829`, 95% CI = `[+0.0604, +0.1308]`
- `conv1d` (grupa `mid`, `MSE`, n=2857): różnica = `-0.0970`, 95% CI = `[-0.1760, -0.0348]`
- `conv1d` (grupa `mid`, `MAE`, n=2857): różnica = `+0.0797`, 95% CI = `[+0.0659, +0.0932]`
- `conv1d` (grupa `premium`, `MSE`, n=162): różnica = `+0.6828`, 95% CI = `[+0.2524, +1.1368]`
- `conv1d` (grupa `premium`, `MAE`, n=162): różnica = `+0.0919`, 95% CI = `[+0.0618, +0.1269]`
- `meta_nnls` (grupa `all`, `MSE`, n=4258): różnica = `+0.0522`, 95% CI = `[+0.0050, +0.1276]`
- `meta_nnls` (grupa `mid`, `MAE`, n=2857): różnica = `+0.0332`, 95% CI = `[+0.0233, +0.0407]`
- `meta_context` (grupa `mid`, `MAE`, n=2857): różnica = `+0.0255`, 95% CI = `[+0.0149, +0.0332]`

### 5.2. Porównanie modeli z `best_single` w wariancie `mse` (`model_comparison.csv`)
Przypadki (model, wariant, grupa), w których 95% przedział ufności dla różnicy $\text{MSE}(\text{model}) - \text{MSE}(\text{best\_single\_mse})$ **nie obejmuje zera**:
- `glm` (wariant `mse`, grupa `all`, `MSE`, n=4258): różnica = `+0.3198`, 95% CI = `[+0.1947, +0.4794]`
- `glm` (wariant `mse`, grupa `mid`, `MSE`, n=2857): różnica = `+0.1657`, 95% CI = `[+0.1166, +0.2196]`
- `glm` (wariant `mse`, grupa `premium`, `MSE`, n=162): różnica = `+2.7331`, 95% CI = `[+0.8987, +5.5563]`
- `mlp` (wariant `mse`, grupa `all`, `MSE`, n=4258): różnica = `+0.1620`, 95% CI = `[+0.0699, +0.2489]`
- `mlp` (wariant `mse`, grupa `mid`, `MSE`, n=2857): różnica = `+0.1592`, 95% CI = `[+0.0662, +0.2373]`
- `mlp` (wariant `mse`, grupa `premium`, `MSE`, n=162): różnica = `+1.5645`, 95% CI = `[+0.7796, +2.9317]`
- `conv1d` (wariant `mse`, grupa `all`, `MSE`, n=4258): różnica = `+0.7520`, 95% CI = `[+0.2109, +1.1461]`
- `conv1d` (wariant `mse`, grupa `mid`, `MSE`, n=2857): różnica = `+0.7746`, 95% CI = `[+0.5871, +0.9547]`
- `conv1d` (wariant `mse`, grupa `premium`, `MSE`, n=162): różnica = `+7.4223`, 95% CI = `[+3.8531, +12.9340]`
- `meta_nnls` (wariant `mse`, grupa `budget`, `MSE`, n=1239): różnica = `-0.1812`, 95% CI = `[-0.2712, -0.0025]`
- `meta_nnls` (wariant `mse`, grupa `premium`, `MSE`, n=162): różnica = `+0.3841`, 95% CI = `[+0.1123, +0.6006]`
- `meta_context` (wariant `mse`, grupa `budget`, `MSE`, n=1239): różnica = `-0.2310`, 95% CI = `[-0.3425, -0.0043]`
- `meta_context` (wariant `mse`, grupa `premium`, `MSE`, n=162): różnica = `+0.5684`, 95% CI = `[+0.3048, +0.8119]`
- `mean_ensemble` (wariant `mse`, grupa `all`, `MSE`, n=4258): różnica = `+0.1444`, 95% CI = `[+0.0324, +0.2459]`
- `mean_ensemble` (wariant `mse`, grupa `mid`, `MSE`, n=2857): różnica = `+0.1256`, 95% CI = `[+0.0481, +0.1981]`
- `mean_ensemble` (wariant `mse`, grupa `premium`, `MSE`, n=162): różnica = `+2.0036`, 95% CI = `[+0.6015, +4.3734]`
- `glm` (wariant `fplloss`, grupa `all`, `MSE`, n=4258): różnica = `+0.2506`, 95% CI = `[+0.1642, +0.3872]`
- `glm` (wariant `fplloss`, grupa `budget`, `MSE`, n=1239): różnica = `+0.3560`, 95% CI = `[+0.0219, +0.5403]`
- `glm` (wariant `fplloss`, grupa `mid`, `MSE`, n=2857): różnica = `+0.1343`, 95% CI = `[+0.1097, +0.1574]`
- `glm` (wariant `fplloss`, grupa `premium`, `MSE`, n=162): różnica = `+1.4942`, 95% CI = `[+0.5560, +2.6423]`
- `mlp` (wariant `fplloss`, grupa `all`, `MSE`, n=4258): różnica = `+0.1594`, 95% CI = `[+0.1042, +0.2117]`
- `mlp` (wariant `fplloss`, grupa `mid`, `MSE`, n=2857): różnica = `+0.1258`, 95% CI = `[+0.0876, +0.1666]`
- `mlp` (wariant `fplloss`, grupa `premium`, `MSE`, n=162): różnica = `+1.1113`, 95% CI = `[+0.3543, +2.1417]`
- `conv1d` (wariant `fplloss`, grupa `all`, `MSE`, n=4258): różnica = `+0.7047`, 95% CI = `[+0.0776, +1.1928]`
- `conv1d` (wariant `fplloss`, grupa `mid`, `MSE`, n=2857): różnica = `+0.6776`, 95% CI = `[+0.4896, +0.8266]`
- `conv1d` (wariant `fplloss`, grupa `premium`, `MSE`, n=162): różnica = `+8.1051`, 95% CI = `[+4.4483, +14.0037]`
- `meta_nnls` (wariant `fplloss`, grupa `budget`, `MSE`, n=1239): różnica = `-0.0577`, 95% CI = `[-0.1198, -0.0107]`
- `meta_nnls` (wariant `fplloss`, grupa `premium`, `MSE`, n=162): różnica = `+0.6670`, 95% CI = `[+0.2846, +1.1723]`
- `meta_context` (wariant `fplloss`, grupa `premium`, `MSE`, n=162): różnica = `+0.8123`, 95% CI = `[+0.3149, +1.3296]`
- `mean_ensemble` (wariant `fplloss`, grupa `all`, `MSE`, n=4258): różnica = `+0.1257`, 95% CI = `[+0.0220, +0.2178]`
- `mean_ensemble` (wariant `fplloss`, grupa `mid`, `MSE`, n=2857): różnica = `+0.1023`, 95% CI = `[+0.0462, +0.1525]`
- `mean_ensemble` (wariant `fplloss`, grupa `premium`, `MSE`, n=162): różnica = `+1.7782`, 95% CI = `[+0.4133, +3.7957]`

## 6. Pliki wynikowe w katalogu `results/`
- `metrics_test.csv`, `metrics_valid.csv` — metryki na zbiorze testowym i walidacyjnym,
- `tier_counts.csv` — liczebności grup `budget`, `mid`, `premium` dla `train`, `valid`, `test`,
- `variant_comparison.csv` — różnice `FPLLoss − MSE` dla MSE i MAE z 95% CI z bootstrapu sparowanego,
- `model_comparison.csv` — różnice MSE względem `best_single` (wariant `mse`) z 95% CI z bootstrapu sparowanego,
- `table_tiers.csv` — tabela MSE i MAE w podziale na grupy cenowe (zaokrąglona do 2 miejsc po przecinku),
- `meta_weights.csv` — wagi i wyrazy wolne wyuczonych meta-modeli,
- `preds_valid.parquet`, `preds_test.parquet` — predykcje wszystkich modeli na zbiorze walidacyjnym i testowym.
