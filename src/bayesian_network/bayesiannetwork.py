import pickle
import time

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.metrics import correlation_score, log_likelihood_score
from pgmpy.models import BayesianNetwork
from sklearn.metrics import balanced_accuracy_score

pd.set_option('display.max_columns', 100)


# Funzione che visualizza il grafo del Bayesian Network
def visualizeBayesianNetwork(bayesianNetwork: BayesianNetwork):
    G = nx.MultiDiGraph(bayesianNetwork.edges())
    pos = nx.spring_layout(G, iterations=100, k=2,
                           threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=250, node_color="#ff574c")
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=8,
        arrowstyle="->",
        edge_color="blue",
        connectionstyle="arc3,rad=0.2",
        min_source_margin=1.2,
        min_target_margin=1.5,
        edge_vmin=2,
        edge_vmax=2,
    )

    plt.title("BAYESIAN NETWORK GRAPH")
    plt.show()
    plt.clf()


def visualizeInfo(bayesianNetwork: BayesianNetwork):
    for cpd in bayesianNetwork.get_cpds():
        print(f'CPT of {cpd.variable}:')
        print(cpd, '\n')


# Funzione che crea la rete bayesiana
def bNetCreation(df):
    # Ricerca della struttura ottimale
    hc_k2 = HillClimbSearch(df)
    k2_model = hc_k2.estimate(scoring_method='k2score', max_iter=100)
    # Creazione della rete bayesiana
    model = BayesianNetwork(k2_model.edges())
    model.fit(df, estimator=MaximumLikelihoodEstimator, n_jobs=-1)
    # Salvo la rete bayesiana su file
    with open('modellol.pkl', 'wb') as output:
        pickle.dump(model, output)
    visualizeBayesianNetwork(model)
    # visualizeInfo(model)
    return model


# Funzione che carica la rete bayesiana da file
def loadBayesianNetwork():
    with open('modello2.pkl', 'rb') as input:
        model = pickle.load(input)
    visualizeBayesianNetwork(model)
    # visualizeInfo(model)
    return model


# Predico il valore di differentialColumn per l'esempio
def predict(bayesianNetwork: BayesianNetwork, example, differentialColumn):
    inference = VariableElimination(bayesianNetwork)
    result = inference.query(variables=[differentialColumn], evidence=example, elimination_order='MinFill')
    print(result)


# genera un esempio randomico
def generateRandomExample(bayesianNetwork: BayesianNetwork):
    return bayesianNetwork.simulate(n_samples=1).drop(columns=['clusterIndex'])


# take from file dataset

df = pd.read_csv("../../data/dataset_preprocessed_bayesian.csv")


def markov_blanket_of(node):
    print(f'Markov blanket of \'{node}\' is {set(bayesianNetwork.get_markov_blanket(node))}')


def generateRandomExample(bayesianNetwork: BayesianNetwork):
    exp = bayesianNetwork.simulate(n_samples=1)
    exprating = exp['Rating']
    expr = exp.drop(columns=['Rating'])
    return exprating, expr


def multi_predict(bayesianNetwork: BayesianNetwork, n_iter):
    for n in range(n_iter):
        exprating, exp = generateRandomExample(bayesianNetwork)
        print("", exp)
        print("  Rating: ", exprating.values[0])
        print(predict(bayesianNetwork, exp.to_dict('records')[0], 'Rating'))
        print("--------- ORA SENZA DEBT/EQUITY RATIO -----------")
        del exp["Debt/Equity Ratio"]
        print(predict(bayesianNetwork, exp.to_dict('records')[0], 'Rating'))


def query_report(infer, variables, evidence=None, elimination_order="MinFill", show_progress=False, desc=""):
    if desc:
        print(desc)
    start_time = time.time()
    # evidence = {key: df[key].values[0] for key in evidence.keys()} if evidence else None
    print(infer.query(variables=variables,
                      evidence=evidence,
                      elimination_order=elimination_order,
                      show_progress=show_progress))
    print(f'--- Query executed in {time.time() - start_time:0,.4f} seconds ---\n')


from sklearn.preprocessing import LabelEncoder

# Initialize a label encoder
label_encoder = LabelEncoder()

# bayesianNetwork = bNetCreation(df)
bayesianNetwork = loadBayesianNetwork()
print(correlation_score(bayesianNetwork, df, score=balanced_accuracy_score))
# multi_predict(bayesianNetwork, 10)

# print(multi_predict(bayesianNetwork, 10))
infer = VariableElimination(bayesianNetwork)
query_report(infer, variables=['Debt/Equity Ratio'], evidence={'Rating': 3},
             desc='Data la osservazione che una azienda è molto rischiosa qual è la distribuzione di probabilità '
                  'per Debt/Equity Ratio')
query_report(infer, variables=['Debt/Equity Ratio', 'Operating Cash Flow Per Share'], evidence={'Rating': 3},
             desc='Data la osservazione che una azienda è molto rischiosa qual è la distribuzione di probabilità '
                  'congiunta di Debt/Equity Ratio e Operating Cash Flow Per Share')
