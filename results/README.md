# Eksperymenty Predykcji Punktów FPL — Modele Bazowe, FPLLoss i Meta-Modele

## 1. Uruchomienie eksperymentu
Cały potok eksperymentalny (weryfikacja testami jednostkowymi, przygotowanie danych, strojenie GLM i XGBoost, trening sieci MLP i Conv1D, konstrukcja meta-modeli oraz ewaluacja na zbiorze testowym) uruchamiany jest jednym poleceniem z katalogu głównego repozytorium:

```bash
PYTHONPATH=src python -m fpl_project.models.run_experiments
```

Ziarno losowe zostało ustalone na `SEED = 77` (`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`), co gwarantuje pełną powtarzalność wyników.

## 2. Podział danych i brak wycieku informacji
- **Train (`2022/23`, `2023/24`, `2024/25`)**: `80618` obserwacji (posortowanych globalnie po `(season, gw, code)`).
  - Dla sieci neuronowych (`MLP`, `Conv1D`) wydzielono wewnętrzny chronologiczny zbiór wczesnego zatrzymania (`early_stopping_val`): ostatnie 8 kolejek sezonu `2024/25` (`gw >= 31`, `6085` obserwacji), a trening właściwy przebiegał na wcześniejszych kolejkach (`74533` obserwacji). Zbiór `valid` nie był używany do wczesnego zatrzymania ani doboru hiperparametrów.
- **Valid (`2025/26`)**: `29338` obserwacji — użyty wyłącznie do wygenerowania predykcji modeli bazowych uczących meta-modele oraz wyboru najlepszego pojedynczego modelu odniesienia.
- **Test (`2026/27`, rozegrane kolejki `is_played == True`)**: `4258` obserwacji — użyty wyłącznie do końcowej oceny wszystkich modeli.
- **Złączenie wewnętrzne (`inner join` po `(code, season, gw)`)**:
  - Liczba wierszy odrzuconych przy łączeniu `features_engineering.parquet` z `eda.parquet` i predykcjami `Conv1D`: `0` na `valid` oraz `0` na `test` (w `train_sub` dla `Conv1D` pominięto `19031` początkowych wierszy rozgrzewki okna `seq_len = 8`).

## 3. Funkcja straty `FPLLoss` i wartość stałej normalizującej $c$
Wagi obserwacji wyliczane są z **nieskalowanych** kolumn wiersza:
- Cena: `price = value_ema_3` (w mln funtów),
- Minuty: `m = clip(minutes_ema_3 / 90.0, 0.0, 1.0)`.

Na zbiorze treningowym (`train`) wyznaczono:
- Kwantyl `0.50` ceny ($p_{\text{low}}$): **`4.500000` mln £**
- Kwantyl `0.955` ceny ($p_{\text{high}}$): **`6.925000` mln £**
- Stała normalizująca $c$ (`norm_const` $= \text{mean}(\ln(1 + \max(m_i, 0)) + \varepsilon_{\min})$ przy $\varepsilon_{\min} = 0.1$): **`0.316909`**

Waga pojedynczej obserwacji:
$$w_i = \frac{w_{\text{cena}, i} \cdot (\ln(1 + \max(m_i, 0)) + 0.1)}{0.316909}, \quad w_{\text{cena}, i} \in \{1.0, 1.25, 2.0\}$$

## 4. Realizacja wariantu (B) dla GLM i XGBoost
- **XGBoost (`objective="reg:squarederror"`)**: przekazanie `sample_weight = w_i` do `fit()` minimalizuje dokładnie ważony błąd średniokwadratowy $\sum_i w_i (y_i - \hat{y}_i)^2$, czyli dokładnie tę samą funkcję celu co `FPLLoss`.
- **GLM (`glum.GeneralizedLinearRegressor`)**: przekazanie `sample_weight = w_i` maksymalizuje **ważoną log-wiarygodność** (weighted log-likelihood / ważoną dewiancję wybranej rodziny rozkładu: Poisson, Negative Binomial z estymowanym/strojonym $\theta$ lub Tweedie z mocą $p \in [1.1, 1.9]$). Dla uogólnionych modeli liniowych z funkcją wiążącą `log` i niegaussowską funkcją wariancji jest to **analog** `FPLLoss` (ważenie obserwacji tymi samymi wagami decyzyjnymi $w_i$), a nie algebraicznie identyczna ważona strata kwadratowa MSE.
- W `TimeSeriesSplit` CV (z granicami foldów wypadającymi ściśle pomiędzy kolejkami `(season, gw)`) wagi `w_i` przekazywane są zarówno do `estimator.fit()`, jak i do ważonego kryterium oceny foldu (`neg_weighted_mse`).

## 5. Wygenerowane pliki w katalogu `results/`
- `metrics_test.csv` — pełne zestawienie metryk na zbiorze testowym (`2026/27`) wraz z liczbą cech wejściowych `p` użytych w $\text{Adj } R^2$,
- `metrics_valid.csv` — zestawienie metryk na zbiorze walidacyjnym (`2025/26`),
- `preds_valid.parquet`, `preds_test.parquet` — identyfikatory zawodników, `y` oraz predykcje wszystkich modeli bazowych i meta-modeli,
- `meta_weights.csv` — wagi i wyrazy wolne (`intercept`) wyuczonych meta-modeli (`meta_nnls` oraz `meta_context`).
