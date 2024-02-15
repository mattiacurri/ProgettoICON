import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


# Funzione che mostra la curva di apprendimento per ogni modello
def plot_learning_curves(model, X, y, differentialColumn, model_name, method_name, cv, scoring='balanced_accuracy', smote=False):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
                                                            random_state=42)

    # Calcola gli errori su addestramento e test
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Salva su file i valori numerici della deviazione standard e della varianza
    if smote:
        file = open(f'../../plots/SMOTEPipeline/learning_curve_{model_name}_{method_name}.txt', 'w')
        file.write(f"Train Error Std: {train_errors_std[-1]}\n")
        file.write(f"Test Error Std: {test_errors_std[-1]}\n")
        file.write(f"Train Error Var: {train_errors_var[-1]}\n")
        file.write(f"Test Error Var: {test_errors_var[-1]}\n")
        file.close()
    else:
        file = open(f'../../plots/learning_curve_{model_name}_{method_name}.txt', 'w')
        file.write(f"Train Error Std: {train_errors_std[-1]}\n")
        file.write(f"Test Error Std: {test_errors_std[-1]}\n")
        file.write(f"Train Error Var: {train_errors_var[-1]}\n")
        file.write(f"Test Error Var: {test_errors_var[-1]}\n")
        file.close()

    print(
        f"\033[94m{model_name} - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}\033[0m")
    if smote:
        f = open(f'../../plots/SMOTEPipeline/var_std_{model_name}_{method_name}.txt', 'w')
        f.write(f"Train Error Std: {train_errors_std[-1]}\n")
        f.write(f"Test Error Std: {test_errors_std[-1]}\n")
        f.write(f"Train Error Var: {train_errors_var[-1]}\n")
        f.write(f"Test Error Var: {test_errors_var[-1]}\n")
        f.close()
    else:
        f = open(f'../../plots/var_std_{model_name}_{method_name}.txt', 'w')
        f.write(f"Train Error Std: {train_errors_std[-1]}\n")
        f.write(f"Test Error Std: {test_errors_std[-1]}\n")
        f.write(f"Train Error Var: {train_errors_var[-1]}\n")
        f.write(f"Test Error Var: {test_errors_var[-1]}\n")
        f.close()

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
    if smote:
        plt.savefig(f'../../plots/SMOTEPipeline/learning_curve_{model_name}_{method_name}.png')
    else:
        plt.savefig(f'../../plots/learning_curve_{model_name}_{method_name}.png')

    plt.show()

# Funzione che visualizza i grafici delle metriche per ogni modello
def visualizeMetricsGraphs(model, title, smote=False):
    models = list(model.keys())

    # Creazione di un array numpy per ogni metrica
    accuracy = np.array([model[clf]["accuracy_list"] for clf in models])
    cohen = np.array([model[clf]["cohen_kappa"] for clf in models])
    f1 = np.array([model[clf]["geometric_mean"] for clf in models])

    # Calcolo delle medie per ogni modello e metrica
    mean_accuracy = np.mean(accuracy, axis=1)
    mean_cohen = np.mean(cohen, axis=1)
    mean_geo = np.mean(f1, axis=1)

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
            mean_cohen[i],
            f"{mean_cohen[i]:.2f}",
            ha="center",
            va="bottom",
        )
        plt.text(
            i + 2 * bar_width, mean_geo[i], f"{mean_geo[i]:.2f}", ha="center", va="bottom"
        )

    plt.bar(index, mean_accuracy, bar_width, label="Balanced Accuracy")
    plt.bar(index + bar_width, mean_cohen, bar_width, label="Cohen's Kappa")
    plt.bar(index + 2 * bar_width, mean_geo, bar_width, label="Geometric Mean")
    # Aggiunta di etichette e legenda
    plt.xlabel("Punteggio medio per ogni modello")
    plt.ylabel("Punteggi medi")
    plt.title(title)
    plt.xticks(index + 1.5 * bar_width, models)
    plt.legend()

    # allontana la legenda
    plt.legend(loc="lower left", fontsize="small")

    # save to file
    if smote:
        plt.savefig(f'../../plots/SMOTEPipeline/metrics_{title}.png')
    else:
        plt.savefig(f'../../plots/metrics_{title}.png')

    # Visualizzazione del grafico
    plt.show()
    return mean_accuracy, mean_cohen, mean_geo

def visualizeAspectRatioChart(dataSet, differentialColumn, title):
    # Count the occurrences for each unique value of differentialColumn
    counts = dataSet[differentialColumn].value_counts()

    # Labels for the chart
    labels = [f'{label} (n={count})' for label, count in zip(counts.index, counts.values)]
    colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'gold', 'mediumorchid', 'lightsteelblue', 'lightpink','lightgrey','lightblue']

    # Long list of colors to avoid repetitions in case of many unique values
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.legend(labels, loc='lower left', fontsize='small')

    # Modify the title to include the number of classes
    plt.title(f"{title} - Number of classes: {len(labels)}")
    # save to file

    plt.savefig(f'../../plots/aspect_ratio_unbalanced.png')
    plt.show()

def sturgeRule(n):
    return int(1 + np.log2(n)) // 3