<p align="center">
  <h1 align="center">Detekcja Prawdy i Kłamstwa z EEG</h1>
</p>

## Tech Stack

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)](https://www.python.org/) [![MNE-Python](https://img.shields.io/badge/MNE--Python-1.5+-8DD6F9?logo=python)](https://mne.tools/) [![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy)](https://numpy.org/) [![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas)](https://pandas.pydata.org/)

[![SciPy](https://img.shields.io/badge/SciPy-1.10+-8CAAE6?logo=scipy)](https://scipy.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn)](https://scikit-learn.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c)](https://matplotlib.org/) [![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-9cf)](https://seaborn.pydata.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/) [![Jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626?logo=jupyter)](https://jupyter.org/)

[![Status](https://img.shields.io/badge/Status-Alpha-yellow)]() [![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

</div>

## Start

Automatyczne uruchomienie potoku:

```bash
python run.py
```

Polecenie wykonuje wszystkie skrypty i notebooki:

1. Ekstrakcja cech (`extract_features.py`)
2. Analiza według płci (`sex_analysis.py`)
3. Analiza według wieku (`age_analysis.py`)
4. Notebook EDA (`eda.ipynb`)
5. Notebook modeli ML (`baseline_models.ipynb`)
6. Notebook sieci neuronowych (`neural_networks.ipynb`)

## Notebooki

### Eksploracyjna Analiza Danych (eda.ipynb)

Analiza eksploracyjna obejmująca ocenę jakości sygnału, wizualizację składowej P300, analizę oscylacji neuronalnych, porównania według płci przez 4 zintegrowane wykresy, analizę korelacji z wiekiem przez 1 zintegrowany wykres oraz mapy topograficzne wzorców aktywności mózgu.

### Bazowe Modele Uczenia Maszynowego (baseline_models.ipynb)

Walidacja krzyżowa niezależna od uczestnika przy użyciu strategii GroupKFold z progiem wariancji i selekcją cech opartą na F-score, w tym macierze pomyłek i wykresy porównania wydajności dla wszystkich klasyfikatorów.

**Trening i ewaluacja:**

- Random Forest (RF)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Eksperymenty z Sieciami Neuronowymi (neural_networks.ipynb)

Podejścia głębokiego uczenia do detekcji kłamstwa przy użyciu wielowarstwowych perceptronów (MLP) z walidacją niezależną od uczestnika.

**Eksperymenty obejmują:**

- porównanie architektur (1–3 warstwy ukryte)
- testowanie funkcji aktywacji (ReLU, Tanh, Sigmoid)
- analiza regularyzacji (kara L2)
- wizualizacja wydajności i metryki

## Potok analityczny

```mermaid
graph TD
    A[Surowe dane EEG<br/>pliki .fif] --> B[Ładowanie danych<br/>data_loader.py]
    B --> C[Preprocessing<br/>preprocessing.py]
    C --> D[Ekstrakcja cech<br/>extract_features.py]
    D --> E[Baza cech<br/>real_features.pkl]
    E --> F[Analiza płci<br/>sex_analysis.py]
    E --> G[Analiza wieku<br/>age_analysis.py]
    E --> H[Notebook EDA<br/>eda.ipynb]
    E --> I[Modele ML<br/>baseline_models.ipynb]
    C --> J[Cache ML<br/>ml_features_cache.pkl]
    J --> I
    J --> K[Sieci neuronowe<br/>neural_networks.ipynb]
    F --> L[Charts/<br/>Wizualizacje płci]
    G --> L
    H --> L
    I --> M[Wydajność modeli<br/>Metryki i wykresy]
    K --> M
```

## Szczegóły techniczne

### Architektura analizy

- **15 uczestników** z kompletnymi danymi (wszystkie 4 bloki)
- **4 bloki eksperymentalne** na uczestnika:
  - szczera odpowiedź na prawdziwą tożsamość
  - kłamliwa odpowiedź na prawdziwą tożsamość
  - szczera odpowiedź na fałszywą tożsamość
  - kłamliwa odpowiedź na fałszywą tożsamość
- **21 kanałów EEG** @ 250 Hz częstotliwość próbkowania
- **~60 epok** na uczestnika (po odrzuceniu artefaktów)

### Przetwarzanie sygnałów

1. **Filtracja**: pasmowo-przepustowy 0,1–30 Hz, notch 50 Hz
2. **Epokowanie**: −0,2 do 0,8 s względem bodźca
3. **Korekcja linii bazowej**: −0,2 do 0 s przed bodźcem
4. **Odrzucanie artefaktów**: automatyczne, progowe

```mermaid
graph TD
    A[Surowy sygnał EEG<br/>21 kanałów] --> B[Filtr pasmowy<br/>0.1-30 Hz]
    B --> C[Filtr notch<br/>50 Hz]
    C --> D[Epokowanie<br/>-0.2 do 0.8s]
    D --> E[Korekcja linii bazowej<br/>-0.2 do 0s]
    E --> F[Odrzucanie artefaktów<br/>Próg automatyczny]
    F --> G[Czyste epoki<br/>Gotowe do analizy]

```

## Wyekstrahowane cechy

### 1. Składowa P300 (300–500 ms po bodźcu)

Składowa P300 służy jako marker przetwarzania autoreferencyjnego i wykazuje wyższe amplitudy przy szczerych odpowiedziach na prawdziwą tożsamość i zredukowane amplitudy podczas kłamstwa.

### 2. Oscylacje neuronalne (PSD Welcha)

- **Theta (4–8 Hz)**: pamięć robocza i kontrola poznawcza
- **Alpha (8–13 Hz)**: uwaga i hamowanie
- **Beta (13–30 Hz)**: przygotowanie ruchowe i przetwarzanie poznawcze
- **Gamma (30–100 Hz)**: poznanie wysokiego poziomu (filtr notch 50/100 Hz)

### Ekstrakcja cech

- **dziedzina czasu**: amplituda P300 (średnia 300–500 ms)
- **dziedzina częstotliwości**: PSD Welcha (okno 256-punktowe)
- **behawioralne**: czas reakcji na podstawie adnotacji (zakres 200–3000 ms)

```mermaid
graph LR
    A[Czyste epoki] --> B[Analiza<br/>dziedziny czasu]
    A --> C[Analiza<br/>dziedziny częstotliwości]
    A --> D[Analiza<br/>behawioralna]

    B --> E[Amplituda P300<br/>okno 300-500ms]

    C --> F[PSD Welcha<br/>FFT 256-punktowy]
    F --> G[Theta 4-8Hz]
    F --> H[Alpha 8-13Hz]
    F --> I[Beta 13-30Hz]
    F --> J[Gamma 30-100Hz]

    D --> K[Czas reakcji<br/>200-3000ms]

    E --> L[Wektor cech<br/>na próbę]
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L
```

## Projekt eksperymentu

```mermaid
graph TD
    A[Uczestnik] --> B[Blok 1:<br/>Szczera odpowiedź<br/>Prawdziwa tożsamość]
    A --> C[Blok 2:<br/>Kłamliwa odpowiedź<br/>Prawdziwa tożsamość]
    A --> D[Blok 3:<br/>Szczera odpowiedź<br/>Fałszywa tożsamość]
    A --> E[Blok 4:<br/>Kłamliwa odpowiedź<br/>Fałszywa tożsamość]

    B --> F[Zapis EEG<br/>21 kanałów @ 250Hz]
    C --> F
    D --> F
    E --> F

    F --> G[~60 epok/blok<br/>po odrzuceniu artefaktów]

```

## Struktura projektu

```
IMPLEMENTATION/
├── run.py                       # automatyczny runner potoku
├── extract_features.py          # ekstrakcja cech (P300, Theta, Alpha, Beta, Gamma, RT)
├── sex_analysis.py              # analiza neuronalna według płci
├── age_analysis.py              # analiza korelacji z wiekiem
├── data_loader.py               # narzędzia do wczytywania danych EEG
├── preprocessing.py             # przetwarzanie sygnałów i ekstrakcja cech
├── analysis_utils.py            # funkcje pomocnicze do analizy danych
├── visualization_config.py      # ujednolicona konfiguracja stylów wykresów
├── eda.ipynb                    # notebook eksploracyjnej analizy danych
├── baseline_models.ipynb        # notebook modeli uczenia maszynowego
├── neural_networks.ipynb        # notebook eksperymentów głębokiego uczenia
├── results/
│   ├── real_features.pkl        # wyekstrahowane cechy
│   └── ml_features_cache.pkl    # buforowane cechy ML
└── charts/                      # wygenerowane wizualizacje
```

## Wnioski

### Demografia

- **Płeć**: Mężczyźni wolniejsi o 70 ms od kobiet (712 ms vs 643 ms)
- **Wiek**: Słaba korelacja dodatnia (r=0,127, p=0,368, nieistotna)

### Amplituda P300

| Warunek             | Szczery | Kłamliwy | Różnica |
| ------------------- | ------- | -------- | ------- |
| Prawdziwa tożsamość | 4,73 µV | 3,98 µV  | −16,0%  |
| Fałszywa tożsamość  | 3,64 µV | 3,18 µV  | −12,8%  |

### Czas reakcji

| Warunek             | Szczery | Kłamliwy | Koszt RT |
| ------------------- | ------- | -------- | -------- |
| Prawdziwa tożsamość | 658 ms  | 663 ms   | 5 ms     |
| Fałszywa tożsamość  | 643 ms  | 746 ms   | 103 ms   |

### Wyniki modeli

#### Sieć neuronowa MLP

| Metryka  | CV GroupKFold | Zbiór treningowy |
| -------- | ------------- | ---------------- |
| Accuracy | 46,0%         | 78,2%            |
| F1-Score | 46,9%         | 77,8%            |

**Konfiguracja:** architektura (250→100→50→25→1), aktywacja `tanh`, regularyzacja L2 = 0,0001, 6 463 próbek, 15 uczestników.

Duża różnica między wynikami CV a treningowymi wskazuje na przeuczenie przy tak małej kohortie. Wyniki CV (~47% F1) są poniżej progu 60%, co jest typowe dla zadań detekcji kłamstwa z EEG przy walidacji niezależnej od uczestnika (GroupKFold).
