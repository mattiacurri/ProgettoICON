# Corporate Credit Rating Prediction System

## Descrizione del Progetto

Questo progetto mira a prevedere il **Corporate Credit Rating**, che rappresenta l'opinione di un'agenzia indipendente sulla capacità di un'azienda di adempiere ai propri obblighi finanziari. Utilizzando tecniche di **apprendimento supervisionato** e **ragionamento probabilistico**, il sistema automatizza la valutazione del rischio aziendale, classificando le società in base alla loro solidità finanziaria.

Il progetto utilizza modelli di **Machine Learning** come Decision Tree, RandomForest, XGBoost e LightGBM per predire il rating, e include una **rete bayesiana** per fare inferenze probabilistiche sui dati.

## Struttura del Progetto

1. **Preprocessing dei dati**
   - Pulizia e preparazione del dataset con tecniche come normalizzazione e one-hot encoding.
   - Riduzione delle classi di rating da 22 a 4 principali categorie rappresentanti rischio minimo-basso, medio, medio-alto, alto-default.

2. **Apprendimento Supervisionato**
   - Utilizzo di modelli come Decision Tree, RandomForest, XGBoost e LightGBM.
   - Test di varie tecniche di riequilibrio delle classi, come SMOTE e ADASYN.

3. **Rete Bayesiana**
   - Creazione di una rete bayesiana per gestire casi di dati mancanti e generare nuove previsioni basate su evidenze probabilistiche.

## Risultati

I risultati mostrano che l'approccio basato su **SMOTE** ha prodotto le migliori prestazioni complessive, in particolare con i modelli **RandomForest** e **XGBoost**. Le metriche utilizzate per la valutazione includono Balanced Accuracy, Cohen's Kappa e Geometric Mean.

## Sviluppi Futuri

- Aumentare il numero di dati per migliorare l'accuratezza delle previsioni.
- Approfondire l'utilizzo di reti neurali per la classificazione.
- Integrare altre variabili per un'analisi più completa e accurata.
