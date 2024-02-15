import pickle
import time

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

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
    # Get the CPDs
    print(f'Check model: {bayesianNetwork.check_model()}\n')
    for cpd in bayesianNetwork.get_cpds():
        print(f'CPT of {cpd.variable}:')
        print(cpd, '\n')


# Funzione che crea la rete bayesiana
def bNetCreation(df):
    #Ricerca della struttura ottimale
    hc_k2=HillClimbSearch(df)
    k2_model=hc_k2.estimate(scoring_method='k2score', max_iter=100)
    #Creazione della rete bayesiana
    model = BayesianNetwork(k2_model.edges())
    model.fit(df, estimator=MaximumLikelihoodEstimator, n_jobs=-1)
    #Salvo la rete bayesiana su file
    with open('modelloSMOTE.pkl', 'wb') as output:
        pickle.dump(model, output)
    visualizeBayesianNetwork(model)
    #visualizeInfo(model)
    return model

# Funzione che carica la rete bayesiana da file
def loadBayesianNetwork():
    with open('modello2.pkl', 'rb') as input:
        model = pickle.load(input)
    # visualizeBayesianNetwork(model)
    # visualizeInfo(model)
    return model

#Predico il valore di differentialColumn per l'esempio
def predict(bayesianNetwork: BayesianNetwork, example, differentialColumn):
    inference = VariableElimination(bayesianNetwork)
    result = inference.query(variables=[differentialColumn], evidence=example, elimination_order='MinFill')
    print(result)

#genera un esempio randomico
def generateRandomExample(bayesianNetwork: BayesianNetwork):
    return bayesianNetwork.simulate(n_samples=1).drop(columns=['clusterIndex'])


# take from file dataset

df = pd.read_csv("../../data/dataset_preprocessed_bayesian.csv")

def markov_blanket_of(node):
    print(f'Markov blanket of \'{node}\' is {set(bayesianNetwork.get_markov_blanket(node))}')

def generateRandomExample(bayesianNetwork: BayesianNetwork):
    return bayesianNetwork.simulate(n_samples=1).drop(columns=['Rating'])

def multi_predict(bayesianNetwork: BayesianNetwork, n_iter):
    for n in range(n_iter):
        exp = generateRandomExample(bayesianNetwork)
        for d in df.columns:
            print(exp[d] if d != "Rating" else "")
        print(predict(bayesianNetwork, exp.to_dict('records')[0], 'Rating'))
        print("--------------------")

def query_report(infer, variables, evidence=None, elimination_order="MinFill", show_progress=False, desc=""):
    if desc:
        print(desc)
    start_time = time.time()
    #evidence = {key: df[key].values[0] for key in evidence.keys()} if evidence else None
    print(infer.query(variables=variables,
                      evidence=evidence,
                      elimination_order=elimination_order,
                      show_progress=show_progress))
    print(f'--- Query executed in {time.time() - start_time:0,.4f} seconds ---\n')

#bayesianNetwork = bNetCreation(df)
bayesianNetwork = loadBayesianNetwork()
exp = generateRandomExample(bayesianNetwork)
for d in df.columns:
    print(exp[d] if d != "Rating" else "")
predict(bayesianNetwork, exp.to_dict('records')[0], 'Rating')

del exp["Debt/Equity Ratio"]
print(exp)
predict(bayesianNetwork, exp.to_dict('records')[0], 'Rating')


# print(multi_predict(bayesianNetwork, 10))
infer = VariableElimination(bayesianNetwork)
query_report(infer, variables=['Rating'], evidence={'Current Ratio': 1, 'Debt/Equity Ratio': 1})
#query_report(infer, variables=['Debt/Equity Ratio', 'ROI - Return On Investment'], evidence={"Rating": 1})
#query_report(infer, variables=['Debt/Equity Ratio', 'ROI - Return On Investment'], evidence={"Rating": 2})
#query_report(infer, variables=['Debt/Equity Ratio', 'ROI - Return On Investment'], evidence={"Rating": 3})
#query_report(infer, variables=['Rating'], evidence={'Rating': 1}, desc="Probabilità che sia stata una certa agenzia a dare il rating 1")
#query_report(infer, variables=['Rating'], evidence={'Rating': 2}, desc="Probabilità che sia stata una certa agenzia a dare il rating 1")
#query_report(infer, variables=['Rating'], evidence={'Rating': 3}, desc="Probabilità che sia stata una certa agenzia a dare il rating 1")
