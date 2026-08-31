Scoring kredytowy z porównaniem podejść klasycznych i sieci neuronowej

Idea: predykcja prawdopodobieństwa niespłacenia kredytu na podstawie danych o kliencie.

Dane: Kaggle "Give Me Some Credit" lub "Home Credit Default Risk", ewentualnie UCI "Statlog German Credit Data".
EDA: rozkłady dochodu/wieku/DTI, analiza braków (np. luki w historii kredytowej), outliery w dochodach, feature engineering (debt-to-income, liczba zapytań kredytowych w czasie).
ML klasyczne: regresja logistyczna, Random Forest, XGBoost/LightGBM + tuning (Optuna/GridSearch).
Sieć neuronowa: MLP z embeddingami dla cech kategorycznych (tabularna sieć neuronowa) — realne porównanie z modelami klasycznymi na tych samych danych.
Implementacja: klasy DataLoader, Preprocessor, ModelTrainer, Predictor; wzorzec Strategy do przełączania modeli; testy jednostkowe transformacji danych.
Wdrożenie: FastAPI z endpointem scoringowym, Docker, CI/CD (GitHub Actions), deploy na Cloud Run/GKE.