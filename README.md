# elasticNN
Experiments for dynamically growing and pruning NN architectures. 

## Goal
The goal is to develop rules which allow a neural net to add neurons dynamically during supervised training to encompass the complexity of the problem. This may involve adding, removing, splitting, or merging neurons within a layer, and perhaps creating entirely new layers. Rules will be informed by information theoretic principles as to avoid being overly heuristicy and emperical.

## Methods
We will begin by training MNIST classifiers with varying depths and widths. We will try to characterize underrepresentative and overrepresentative layers within the network.

Then we will use the insight from this study to develop rules for modifying the network, and assess the affects of these changes on our networks. Onc we have developed useful rules, we will implement these rules such that they will run dynamically during the training process.
