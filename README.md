# Kohonen neural network

This is an application that showcases how different cluster can be identified using a Kohonen neural network.  
Clusters of data are represented as black dots and cluster leaders are represented by red and green dots. Red being the starting coordinates and green being the final coordinates.



## How to use
The params.txt file contains the essentials for the Kohonen neural network. The number of units represents the how many potential cluster one might assume the dataset has.  
The learning rate sets the initial learning rate for the network, as each epoch is completed this learning rate is reduced to be more towards the center of a cluster.  
Normalisation is a boolean of 0/1 to enable or disable normalisation for the weights and datapoints.  
  
The training.txt file contains all the datapoints to train the network with. The file contains a 100 different coordinate datapoints which will be shown as black dots in the processing application. These black dots will be the clusters.  

Note: The starting max epochs is 1 which means upon running the application the network only runs for 1 cycle. If learning rate is lower then the number of epochs must increase. To do this change the maxEpochs value in the Processing code to your desired value.
