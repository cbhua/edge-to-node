# E2N: an edge classificaton model
**E2N**, which is Edges to Nodes, is a strategy for edge classification. 

## Introduction

TODO:

- [x] There are many node classification methods, but not too many edge classification ones;
- [x] Intro to some edge to vertige methods

With the development of node embedding, there are more and more models with excellent performances. However, in the meanwhile, little effort has been taken on the problem of edge classification. The problem of edge classification is generally harder than that of vertex classification. This is because vertex classification methods are primarily based on the notion of homophily [E davidand], while it is generally much more diffcult to apply such homophily principle to edges. Some methods based on behavior relation interplay (BIR) has been used in social network link prediction, [35 36 39]such as friends network and trust network. There methods are mostly unsupervised and rely on features on exact domain in graph. [41] also used domain features to infer edge types with supervised models. These domain-feature-baed methods require particular networks and network relationships, so can not handle more general networks' edge classification problem. 

Since there are many excellent models for nodes classification, which can handle more general problems, also the constrains for edge classification methods so far. We considered the method about transferring edges to nodes in graph, and then apply such node classification methods on transfered graph. 

## Background

TODO:

- [x] The shortcut for previous edge to node methods

Gao et al. proposed an expectation maximization approach on edge-type trainsition matrix, which works as transferring edges in graph to nodes based on adjancy matrix of the graph. They applied this method on biomedical problems, about gene co-expression, biochemical regulation, and biomolecular inhibition or activation. 

However, the methods here emphasis more on ragion features, not for exact edge classification, which not match what we need for TCP attack connection detecting. Inspire by this, we tried to create a new edge to node strategy, E2N, to handle this problem. 

## Proposed Algorithms

TODO:

- [x] Two ways to transfer edges to nodes
- [x] Feature we choose and reason 
- [x] Model we applied and reason

We create two strategies to convert edges to nodes. To do so, we firstly directly set edges in original graph $e_{ori}^i$ to transformed nodes $v_{trans}^i$ with several features, which will be shown later. Then connect transformed nodes $v_{trans}^i$ by two methods. Firstly, we create directed edges $e_{trans}^{(i, j)}$ from transformed node $v_{trans}^i$ to transformed node $v_{trans}^j$, if the destination node of the original edge for $v_{trans}^i$  is the same with the source noe of the original edge for $v_{trans}^j$. Secondly, for those edges in orignal graph with the same source node, we make them fully connected in transformed graph.  By this way we get a converted graph where nodes in it are edges in original graph. Then we will assign features to this new graph. 

Features we chose for transformed nodes are source node's in-degree, source node's out-degree, destinate node's in-degree, destinate node's out-degree, port used number, averate connection time and connection times. The explaination for those features are as following. 

Source node's in-degree and out-degree: we have one conception that there are some attackers in the network. Such attackers have the features that they have many out connections while fewer in connections, so if we detect an edge with it's source node having high out-degrees, it is reasonable for us to suspect that this node may be an attacker, and give a higer weight for edges from it to be anormaly connections.

Destinate node's in-degree and out-degree: similar with souce node's in-degree and out-degree features, we also hold on conception that there are some servers easily being attacked in the network. For example, in real world, some server with poor firewall system will be easily attacked, at this time, there will be a high level of in-degree for this server node. So, if we detect an edge with it's destination node haveing high in-degree, it would be also reasonable for us to suspect that this node is being attacked and this edge would be an anormaly connection.  

Port used number: since for each node, i.e. each edge in original graph, there maybe more than once connected within a time window. Such connection may use different ports to connect, we record the type of ports those connections used as a feature, which would be useful to show the pattern of attack connections. 

Average connection time: claculate the average connection times for connections on one edge as a feature. 

Connection times: one edge in original graph will be actived several times, we record this number as a feature. 

We completed the creation of transformed graph so far and going to apply some node classification methods on this graph.We applied multi layers graph convolutional network (GCN) to process the graph. In which model, graph convolutional layers are responsible for exacting high-level node representations, graph pooling layers play the role of downsampling. Based on the strategy of transforming graph, we know that for those edges from one node will be transformed to a community, and using GCN can easily detect them in graph, which match our conceptions about nodes type in graph. 

Beside the GCN layers we applied, we also applied graph attention network (GAT) on the graph. The point of using GAT is to emphasis the connection chain within the graph. This model will give more importance on connection routes. 

## Experiment

TODO:

- [x] Explain some hyper parameters 
- [x] Give the loss decrease plot
- [x] Give the F1 score plot

We create our graph transformer functions to generate edge to node graph from original graph. And then applied DGL packages to analysis and train the model. For GCN model, we set the number of hidden layer to 16. During the experiment, we got to know that by this way the model will give much more weight to exact attack types. To deal with this problem, we added another balance weigh matrix to model to control the unbalance of trinning. 

The unbalance of dataset has much influence on the trainning, since in GCN model, it would be easy to ignore some attack connections within many in-connections or out-connections. We also applied average strategy during assigning features, which will also lost some informations about exact attack connections. 

## Conclusions

## Reference

E.DavidandK.Jon,*Networks,Crowds,andMarkets:ReasoningAbout a Highly Connected World*. Cambridge University Press, 2010.



