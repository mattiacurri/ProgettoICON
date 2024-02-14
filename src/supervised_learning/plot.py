import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import learning_curve


# Funzione che mostra la curva di apprendimento per ogni modello
def plot_learning_curves(model, X, y, differentialColumn, model_name, method_name, cv, scoring='balanced_accuracy'):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
                                                            random_state=42)
    #for train_size, cv_train_scores, cv_test_scores in zip(train_sizes, train_scores, test_scores):
        #print(f"{train_size} samples were used to train the model")
        #print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
        #print(f"The average test accuracy is {cv_test_scores.mean():.2f}")

    # Calcola gli errori su addestramento e test
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Salva su file i valori numerici della deviazione standard e della varianza
    file = open(f'../../plots/learning_curve_{model_name}_{method_name}.txt', 'w')
    file.write(f"Train Error Std: {train_errors_std[-1]}\n")
    file.write(f"Test Error Std: {test_errors_std[-1]}\n")
    file.write(f"Train Error Var: {train_errors_var[-1]}\n")
    file.write(f"Test Error Var: {test_errors_var[-1]}\n")
    file.close()

    # Stampare su terminale la deviazione standard e la varianza
    print(
        f"\033[94m{model_name} - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}\033[0m")

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    # Visualizza la curva di apprendimento
    plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {model_name} con {method_name}')
    # plt.ylim(0, 1)
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()

    # save plot to file
    plt.savefig(f'../../plots/learning_curve_{model_name}_{method_name}.png')

    plt.show()


def visualizeAspectRatioChart(dataSet, differentialColumn, title):
    # Conta le occorrenze per ciascun valore unico di differentialColumn
    counts = dataSet[differentialColumn].value_counts()

    # Etichette e colori per il grafico
    labels = counts.index.tolist()
    colors = [
        "lightcoral",
        "lightskyblue",
        "lightgreen",
        "gold",
        "lightsteelblue",
        "lightpink",
        "lightgrey",
        "lightblue",
        "lightgreen",
        "lightcoral",
        "lightpink",
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
    )
    ax.legend(labels, loc="lower left", fontsize="small")
    plt.title(title)
    plt.show()


# Funzione che visualizza i grafici delle metriche per ogni modello
def visualizeMetricsGraphs(model, title):
    models = list(model.keys())

    # Creazione di un array numpy per ogni metrica
    accuracy = np.array([model[clf]["accuracy_list"] for clf in models])
    precision = np.array([model[clf]["precision_list"] for clf in models])
    recall = np.array([model[clf]["recall_list"] for clf in models])
    f1 = np.array([model[clf]["f1"] for clf in models])

    # Calcolo delle medie per ogni modello e metrica
    mean_accuracy = np.mean(accuracy, axis=1)
    mean_precision = np.mean(precision, axis=1)
    mean_recall = np.mean(recall, axis=1)
    mean_f1 = np.mean(f1, axis=1)

    # Creazione del grafico a barre
    bar_width = 0.2
    index = np.arange(len(models))
    # add the number of the value in the plot
    for i in range(len(models)):
        plt.text(
            i, mean_accuracy[i], f"{mean_accuracy[i]:.2f}", ha="center", va="bottom"
        )
        plt.text(
            i + bar_width,
            mean_precision[i],
            f"{mean_precision[i]:.2f}",
            ha="center",
            va="bottom",
        )
        plt.text(
            i + 2 * bar_width,
            mean_recall[i],
            f"{mean_recall[i]:.2f}",
            ha="center",
            va="bottom",
        )
        plt.text(
            i + 3 * bar_width, mean_f1[i], f"{mean_f1[i]:.2f}", ha="center", va="bottom"
        )

    plt.bar(index, mean_accuracy, bar_width, label="Balanced Accuracy")
    plt.bar(index + bar_width, mean_precision, bar_width, label="Precision (Weighted)")
    plt.bar(index + 2 * bar_width, mean_recall, bar_width, label="Recall (Weighted)")
    plt.bar(index + 3 * bar_width, mean_f1, bar_width, label="F1 (Weighted)")
    # Aggiunta di etichette e legenda
    plt.xlabel("Punteggio medio per ogni modello")
    plt.ylabel("Punteggi medi")
    plt.title(title)
    plt.xticks(index + 1.5 * bar_width, models)
    plt.legend()

    # allontana la legenda
    plt.legend(loc="lower left", fontsize="small")

    # save to file
    plt.savefig(f'../../plots/metrics_{title}.png')

    # Visualizzazione del grafico
    plt.show()
    return mean_accuracy, mean_precision, mean_recall, mean_f1


def stampa_metriche(pred_train, pred_test, y_train, y_test, model):
    # Calcola e stampa l'accuracy per il train set e il test set
    accuracy_train = accuracy_score(y_train, pred_train)
    accuracy_test = accuracy_score(y_test, pred_test)
    print("----------CONFRONTO PERFORMANCE TRAIN VS TEST-----------")
    print(f"Modello valutato: {model}")
    print(f'Accuracy (Train): {accuracy_train}')
    print(f'Accuracy (Test): {accuracy_test}')

    # Calcola e stampa precision, recall, e F1-score per il train set e il test set
    precision_train = precision_score(y_train, pred_train, average='macro')
    recall_train = recall_score(y_train, pred_train, average='macro')
    f1_train = f1_score(y_train, pred_train, average='macro')

    precision_test = precision_score(y_test, pred_test, average='macro')
    recall_test = recall_score(y_test, pred_test, average='macro')
    f1_test = f1_score(y_test, pred_test, average='macro')

    print(f'Precision (Train): {precision_train}')
    print(f'Precision (Test): {precision_test}')
    print(f'Recall (Train): {recall_train}')
    print(f'Recall (Test): {recall_test}')
    print(f'F1-Score (Train): {f1_train}')
    print(f'F1-Score (Test): {f1_test}')

    # Stampa classification report per il train set e il test set
    print('Classification Report (Train):\n', classification_report(y_train, pred_train))
    print('Classification Report (Test):\n', classification_report(y_test, pred_test))
