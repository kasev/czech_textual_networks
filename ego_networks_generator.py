'''PREREQUISITIES'''

### these should go easy
import sys
import pandas as pd
import numpy as np
import os
import string
import collections
import regex as re


### this requires installation
import nltk
from nltk.collocations import *

### for network analysis
import networkx as nx

### for visualization
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio
###init_notebook_mode(connected=True)


def network_formation_df(data_frame, prezident, lexicon_size, threshold):
    '''From a dataframe with rows corresponding to individual documents,
    to be subsellected on the basis of author's name column, for instance'''
    lemmata_list = []
    ### for each year's speech:
    for element in data_frame[data_frame["rok_prezident"].str.endswith(prezident)]["lemmata_filtered"].tolist():        
        lemmata_list.extend(element)
    lexicon = [word_tuple[0] for word_tuple in nltk.FreqDist(lemmata_list).most_common(lexicon_size)]
    bigrams_list = []
    for element in data_frame[data_frame["rok_prezident"].str.endswith(prezident)]["lemmata_filtered"].tolist():
        lemmatized_text = element
        for bigram in nltk.bigrams(lemmatized_text):
            if ((bigram[0] in lexicon) & (bigram[1] in lexicon)):
                if bigram[0] != bigram[1]:
                    bigrams_list.append(tuple(sorted(bigram)))
    bigrams_counts = list((collections.Counter(bigrams_list)).items())
    bigrams_counts = sorted(bigrams_counts, key=lambda x: x[1], reverse=True)
    ### create a NetworkX object
    G = nx.Graph()
    G.clear()
    ### form the network from tuples of this form: (node1, node2, number of co-occurrences / lenght of the document)
    G.add_weighted_edges_from(np.array([(bigram_count[0][0], bigram_count[0][1],  int(bigram_count[1])) for bigram_count in bigrams_counts if bigram_count[1] >= threshold]))
    ### add distance attribute
    for (u, v, wt) in G.edges.data('weight'):
        G.add_edge(u,v,distance=round(1/ int(wt), 5))
    document_lenght = len(lemmata_list)
    for (u, v, wt) in G.edges.data('weight'):
        G.add_edge(u,v,norm_weight=round(int(wt)/document_lenght, 5))
    return G
def ego_network_drawing_reduced(network, term, num_of_neighbours, title, mode):
    '''derrive ego network from a preexisting network
    specify source term and number of neighbors
    includes only shortest paths from the source'''
    length, path = nx.single_source_dijkstra(network, term, target=None, weight="distance")
    shortest_nodes = list(length.keys())[0:num_of_neighbours+1]
    path_values_sorted = [dict_pair[1] for dict_pair in sorted(path.items(), key=lambda pair: list(length.keys()).index(pair[0]))]
    path_edges = []
    for path_to_term in path_values_sorted[1:num_of_neighbours+1]:
        path_edges.extend([tuple(sorted(bigram)) for bigram in nltk.bigrams(path_to_term)])
    shortest_edges = list(set(path_edges))
    ego_network = network.copy(as_view=False)
    nodes_to_remove = []
    for node in ego_network.nodes:
        if node not in shortest_nodes:
            nodes_to_remove.append(node)
    for element in nodes_to_remove:
        ego_network.remove_node(element)    
    edges_to_remove = []
    for edge in ego_network.edges:
        if edge not in shortest_edges:
            if (edge[1],edge[0]) not in shortest_edges:
                edges_to_remove.append(edge)
    for element in edges_to_remove:
        ego_network.remove_edge(element[0], element[1])
    return draw_3d_network(ego_network, title, mode)

def draw_3d_network(networkx_object, file_name, mode):
    '''take networkX object and draw it in 3D'''
    Edges = list(networkx_object.edges)
    L=len(Edges)
    distance_list = [distance[2] for distance in list(networkx_object.edges.data("distance"))]
    weight_list = [int(float(weight[2])) for weight in list(networkx_object.edges.data("weight"))]
    labels= list(networkx_object.nodes)
    N = len(labels)
    adjc= [len(one_adjc) for one_adjc in list((nx.generate_adjlist(networkx_object)))] ### instead of "group"
    pos_3d=nx.spring_layout(networkx_object, weight="weight", dim=3)
    nx.set_node_attributes(networkx_object, pos_3d, "pos_3d")
    layt = [list(array) for array in pos_3d.values()]
    N= len(networkx_object.nodes)
    Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[layt[k][1] for k in range(N)]# y-coordinates
    Zn=[layt[k][2] for k in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for Edge in Edges:
        Xe+=[networkx_object.nodes[Edge[0]]["pos_3d"][0],networkx_object.nodes[Edge[1]]["pos_3d"][0], None]# x-coordinates of edge ends
        Ye+=[networkx_object.nodes[Edge[0]]["pos_3d"][1],networkx_object.nodes[Edge[1]]["pos_3d"][1], None]
        Ze+=[networkx_object.nodes[Edge[0]]["pos_3d"][2],networkx_object.nodes[Edge[1]]["pos_3d"][2], None]

        ### to get the hover into the middle of the line
        ### we have to produce a node in the middle of the line
        ### based on https://stackoverflow.com/questions/46037897/line-hover-text-in-plotly

    middle_node_trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            opacity=0,
            text=weight_list,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                opacity=0
            )
        )

    for Edge in Edges:

        x0,y0,z0 = networkx_object.nodes[Edge[0]]["pos_3d"]
        x1,y1,z1 = networkx_object.nodes[Edge[1]]["pos_3d"]
        ###trace3['x'] += [x0, x1, None]
        ###trace3['y'] += [y0, y1, None]
        ###trace3['z'] += [z0, z1, None]
        ###trace3_list.append(trace3)
        middle_node_trace['x'] += tuple([(x0+x1)/2])
        middle_node_trace['y'] += tuple([(y0+y1)/2])#.append((y0+y1)/2)
        middle_node_trace['z'] += tuple([(z0+z1)/2])#.append((z0+z1)/2)
        

    ### edge trace    
    edge_trace1 = go.Scatter3d(
        x=[], y=[], z=[],
        #hoverinfo='none',
        mode='lines',
        line=dict(width=0.8,color="#000000"),
        )
    edge_trace2 = go.Scatter3d(
        x=[],y=[], z=[],
        #hoverinfo='none',
        mode='lines',
        line=dict(width=0.5,color="#404040"),
        )
    edge_trace3 = go.Scatter3d(
        x=[], y=[], z=[],
        #hoverinfo='none',
        mode='lines',
        line=dict(width=0.3,color="#C0C0C0"),
        )
    best_5percent_norm_weight = sorted(list(networkx_object.edges.data("norm_weight")), key=lambda x: x[2], reverse=True)[int((len(networkx_object.edges.data("norm_weight")) / 100) * 5)][2]
    best_20percent_norm_weight = sorted(list(networkx_object.edges.data("norm_weight")), key=lambda x: x[2], reverse=True)[int((len(networkx_object.edges.data("norm_weight")) / 100) * 20)][2]
    for edge in networkx_object.edges.data():
        if edge[2]["norm_weight"] >= best_5percent_norm_weight:
            x0, y0, z0 = networkx_object.node[edge[0]]['pos_3d']
            x1, y1, z1 = networkx_object.node[edge[1]]['pos_3d']
            edge_trace1['x'] += tuple([x0, x1, None])
            edge_trace1['y'] += tuple([y0, y1, None])
            edge_trace1['z'] += tuple([z0, z1, None])
        else:
            if edge[2]["norm_weight"] >= best_20percent_norm_weight:
                x0, y0, z0 = networkx_object.node[edge[0]]['pos_3d']
                x1, y1, z1 = networkx_object.node[edge[1]]['pos_3d']
                edge_trace1['x'] += tuple([x0, x1, None])
                edge_trace1['y'] += tuple([y0, y1, None])
                edge_trace1['z'] += tuple([z0, z1, None])
            else:
                x0, y0, z0 = networkx_object.node[edge[0]]['pos_3d']
                x1, y1, z1 = networkx_object.node[edge[1]]['pos_3d']
                edge_trace1['x'] += tuple([x0, x1, None])
                edge_trace1['y'] += tuple([y0, y1, None])
                edge_trace1['z'] += tuple([z0, z1, None])
    

    ### node trace
    node_trace=go.Scatter3d(x=Xn,
                       y=Yn,
                       z=Zn,
                       mode='markers+text',
                       ###name=labels,
                       marker=dict(symbol='circle',
                                     size=6,
                                     color=adjc,
                                     colorscale='Earth',
                                     reversescale=True,
                                     line=dict(color='rgb(50,50,50)', width=0.5)
                                     ),
                       text=[],
                       #textposition='bottom center',
                       #hovertext=adjc,
                       #hoverinfo='text'
                       )
    for node in networkx_object.nodes():
        node_trace["text"] += tuple([node])
    
    axis=dict(showbackground=False,
                  showline=False,
                  zeroline=False,
                  showgrid=False,
                  showticklabels=False,
                  title=''
                  )
    layout = go.Layout(
                 title="",
                 width=1200,
                 height=800,
                 showlegend=False,
                 scene=dict(
                     xaxis=dict(axis),
                     yaxis=dict(axis),
                     zaxis=dict(axis),
                ),
             margin=dict(
                t=100
            ),
            hovermode='closest',
            annotations=[
                   dict(
                   showarrow=False,
                    text="",
                    xref='paper',
                    yref='paper',
                    x=0,
                    y=0.1,
                    xanchor='left',
                    yanchor='bottom',
                    font=dict(
                    size=14
                    )
                    )
                ],    )
    data=[edge_trace1, edge_trace2, edge_trace3, node_trace, middle_node_trace]
    fig=go.Figure(data=data, layout=layout)
    if mode=="offline":
        return plot(fig, filename="../" + file_name+"_3D.html")
    if mode=="online":
        return py.iplot(fig, filename=file_name)
    if mode=="eps":
        return pio.write_image(fig, "../" + file_name + "_3D.eps" , scale=1)

def ego_network_standard(dataframe, prezident_name, term, neighbours):
    network = network_formation_df(dataframe, prezident_name, 200, 1)
    if "term" not in network.nodes():
        print("This term is not part of given network. Try another one.")
    else:
        ego_network_drawing_reduced(network, term, neighbours, prezident_name + " - " + term, "offline")