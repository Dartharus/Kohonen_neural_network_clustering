//Arrays to store x,y values for data points and initial weights for with and without normalization
float[] datasetX = new float[100];
float[] datasetY = new float[100];
float[] classesX = new float[41];
float[] classesY = new float[41];
float[] normDatasetX = new float[100];
float[] normDatasetY = new float[100];
float[] normClassesX = new float[41];
float[] normClassesY = new float[41];
float[] classesStartX = new float[41];
float[] classesStartY = new float[41];
float[] classesEndX = new float[41];
float[] classesEndY = new float[41];

//parameters associated with the Kohonen network
int numOfUnits;
int counter = 0;
int maxEpochs = 1;
float learningRate;
boolean normalize;

//function to round the decimal to a certain number of decimal places to solve rounding issue
float roundDecimal(float decimalplaces, int decimal) 
{
  return Float.parseFloat(String.format("%." + decimal + "f", decimalplaces));
}

void settings()
{
  size(1000,1000);
}

void setup()
{
  //setup of the axis lines
  background(200);
  translate(width*.5,height*.5);
  strokeWeight(1);
  stroke(0);
  line(0,-500,0,500);
  line(-500,0,500,0);
  strokeWeight(7);
  
  //store training data x,y values into arrays
  String[] training = loadStrings("training.txt");
  String[] params = loadStrings("params.txt");
  for(int i = 0; i < 100; i++)
  {
    String[]list = split(training[i],',');
    datasetX[i] = Float.parseFloat(list[0]);
    datasetY[i] = Float.parseFloat(list[1]);
  }
  //store weights x,y values into arrays
  for(int i = 3; i < params.length; i++)
  {
    String[]list = split(params[i],',');
    classesX[i-3] = Float.parseFloat(list[0]);
    classesY[i-3] = Float.parseFloat(list[1]);
  }
  //assign number of units variable
  String[]NOU = split(params[0],',');
  numOfUnits = Integer.parseInt(NOU[1]);
  //assign learning rate variable
  String[]LR = split(params[1],',');
  learningRate = Float.parseFloat(LR[1]);
  //assign normalize boolean variable
  String[]N = split(params[2],',');
  int normcheck = Integer.parseInt(N[1]);
  if(normcheck == 0)
  {
    normalize = false;
  }
  else
  {
    normalize = true;
  }
  //normalize x,y values
  if(normalize == true)
  {
    for(int i = 0; i < datasetX.length; i++)
    {
      PVector v = new PVector(datasetX[i],datasetY[i]);
      v.normalize();
      normDatasetX[i] = v.x;
      normDatasetY[i] = v.y;
    }
    for(int i = 0; i < classesX.length; i++)
    {
      PVector v = new PVector(classesX[i],classesY[i]);
      v.normalize();
      normClassesX[i] = v.x;
      normClassesY[i] = v.y;
    }
  }
  //print params
  println("number of units: " + numOfUnits);
  println("learning rate: " + learningRate);
  println("normalize: " + normalize);
  if(normalize == true)
  {
    for(int i = 0; i < normDatasetX.length; i++)
    {
      stroke(0,0,0);
      point(normDatasetX[i]*200,normDatasetY[i]*200);  //draw points to show the data points
    }
    for(int i = 0; i < numOfUnits; i++)
    {
      stroke(255,0,0);
      classesStartX[i] = normClassesX[i];
      classesStartY[i] = normClassesY[i];
      point(classesStartX[i]*200,classesStartY[i]*200);  //draw points to show the initial weights
    }
    while(counter != maxEpochs)
    {
      for(int i = 0; i < normDatasetX.length; i++)  //unit updating procedure for each data point
      {
        PVector dataset = new PVector(normDatasetX[i],normDatasetY[i]);
        float net = 0.0;
        int winner = 0;
        //println("dataset: " + dataset);
        for(int j = 0; j < numOfUnits; j++)
        {   
          PVector normClass = new PVector(normClassesX[j],normClassesY[j]);
          float curNet = dataset.dot(normClass);  //compute net value
          if(curNet > net)
          {
            winner = j;
            net = curNet;
          }
        }
        //println("net: " + net);
        //println("winner: " + winner);
        PVector winnerClassVector = new PVector(normClassesX[winner],normClassesY[winner]);
        PVector updatedClassVector = winnerClassVector.add(dataset.sub(winnerClassVector).mult(learningRate));
        PVector normUpdatedClassVector = updatedClassVector.normalize();
        //println("old weight: " + normClassesX[winner],normClassesY[winner]);
        normClassesX[winner] = normUpdatedClassVector.x;  //update the current weight in the arrays with the updated weights
        normClassesY[winner] = normUpdatedClassVector.y;
        //println("new weight: " + normClassesX[winner],normClassesY[winner]);
      }
      //println("epoch: " + counter);
      //println("learning rate: " + learningRate);
      counter++;
      //learningRate -= 0.001;  //reduce learning rate by a constant
      //learningRate = learningRate/2;  //or reduce learning rate by division
      
      //calculate silhouette coefficient
      ArrayList<PVector>[] listOfClusters = new ArrayList[numOfUnits]; //create an array of arraylist with the size of number of units that holds vectors to store the data points
      for(int i = 0; i < listOfClusters.length; i++)
      {
        listOfClusters[i] = new ArrayList<PVector>();  //populating the array with an empty arraylist
      }
      ArrayList<PVector>[] listOfClustersCohesionSeperationMetrics = new ArrayList[numOfUnits]; //create an array of arraylist with the size of number of units that holds vectors to store the cohesion and seperation metrics
      for(int i = 0; i < listOfClusters.length; i++)
      {
        listOfClustersCohesionSeperationMetrics[i] = new ArrayList<PVector>();
      }
      for(int i = 0; i < normDatasetX.length; i++)
      {
        PVector dataset = new PVector(normDatasetX[i],normDatasetY[i]);
        float tempDist = 1000000.0;
        int cluster = 0;
        for(int j = 0; j < numOfUnits; j++)  //determine which data point is associated with which cluster based on the distance between the data point and the cluster leader from the Kohonen network
        {
          PVector clusterLeader = new PVector(normClassesX[j],normClassesY[j]);
          float dist = dataset.dist(clusterLeader);
          if(dist < tempDist)
          {
            cluster = j;
            tempDist = dist;
          }
        }
        //println(cluster);
        listOfClusters[cluster].add(dataset);  //add the data points into the array of arraylist by which cluster has which data points
      }
      for(int i = 0; i < numOfUnits; i++)  //calculate the cohesion metric by looping through every data point in each cluster and calcutating the inter cluster distance for each point
      {
        ArrayList cluster = listOfClusters[i];
        //println(cluster);
        if(cluster.size() != 0)
        {
          ArrayList cohesionSeperationCluster = listOfClustersCohesionSeperationMetrics[i];
          for(int j = 0; j < cluster.size(); j++)
          {
            PVector datapoint = (PVector)cluster.get(j);
            float sumOfInterClusterDist = 0.0;
            float cohesionMetric = 0.0;
            for(int k = 0; k < cluster.size(); k++)
            {
              PVector otherDatapoint = (PVector)cluster.get(k);
              if(j != k)
              {
                float interClusterDist = datapoint.dist(otherDatapoint);
                sumOfInterClusterDist += interClusterDist;
              }
            }
            cohesionMetric = sumOfInterClusterDist/(cluster.size()-1);
            //println("cluster: " + i + " datapoint: " + j + " cohesionMetric is: " + cohesionMetric);
            PVector tempV = new PVector(cohesionMetric,1000000);  //dummy value for for the y component of the vector for the seperation metric to replace
            cohesionSeperationCluster.add(tempV);
          }
        }
      }
      for(int i = 0; i < numOfUnits; i++)  //calculate the seperation metric by checking a data point in a cluster and calculating the distance with another point in another cluster
      {
        ArrayList cluster = listOfClusters[i];
        if(cluster.size() != 0)
        {
          ArrayList cohesionSeperationCluster = listOfClustersCohesionSeperationMetrics[i];
          for(int a = 0; a < numOfUnits; a++)
          {
            ArrayList otherCluster = listOfClusters[a];
            if(otherCluster.size() != 0)
            {
              float seperationMetric = 0.0;
              if(i != a)
              {
                for(int b = 0; b < cluster.size(); b++)
                {
                  float distToOtherClusterPoints = 0.0;
                  float sumOfDistToOtherClusterPoints = 0.0;
                  PVector datapoint = (PVector)cluster.get(b);
                  for(int c = 0; c < otherCluster.size(); c++)
                  {
                    PVector otherClusterDatapoint = (PVector)otherCluster.get(c);
                    //println("comparing datapoint: " + b + " from cluster: " + i + " with datapoint: " + c + " from cluster: " + a);
                    distToOtherClusterPoints = datapoint.dist(otherClusterDatapoint);
                    sumOfDistToOtherClusterPoints += distToOtherClusterPoints;
                  }
                  seperationMetric = sumOfDistToOtherClusterPoints/(otherCluster.size()-1);
                  //println("seperationMetric for datapoint: " + b + " from cluster: " + i + " to cluster: " + a + " is " + seperationMetric);
                  PVector tempV = (PVector)cohesionSeperationCluster.get(b);
                  float tempf = tempV.x;
                  float tempSM = tempV.y;
                  //println(tempSM);
                  //println(seperationMetric);
                  if(seperationMetric < tempSM)  //take the lowest seperation metric
                  {
                    PVector tempV2 = new PVector(tempf,seperationMetric,0);
                    cohesionSeperationCluster.set(b,tempV2);
                  }
                }
              }
            }
          }
          //println("cluster: " + i + cohesionSeperationCluster);
        }
      }
      float finalSilhouetteCoefficient;
      float sumOfSilhouetteCoefiicient = 0.0;
      for(int i = 0; i < numOfUnits; i++)
      {
        ArrayList cohesionSeperationCluster = listOfClustersCohesionSeperationMetrics[i];
        if(cohesionSeperationCluster.size() != 0)
        {
          for(int j = 0; j < cohesionSeperationCluster.size(); j++) //calculate the silhouette coefficient for the entire run by taking the average of the silhouette coefficient for every data point
          {
            PVector cohesionSeperationMetric = (PVector)cohesionSeperationCluster.get(j);
            //println(cohesionSeperationMetric);
            float a = cohesionSeperationMetric.x;
            float b = cohesionSeperationMetric.y;
            //println(a + "," + b);
            float silhouetteCoefficient = (b-a)/max(b,a);
            sumOfSilhouetteCoefiicient += silhouetteCoefficient;
            //println(silhouetteCoefficient);
          }
        }
      }
      finalSilhouetteCoefficient = sumOfSilhouetteCoefiicient/datasetX.length;
      finalSilhouetteCoefficient = roundDecimal(finalSilhouetteCoefficient,4);
      println("silhouette coefficient: " + finalSilhouetteCoefficient);
    }
    for(int i = 0; i < numOfUnits; i++)
    {
      stroke(0,255,0);
      point(normClassesX[i]*200,normClassesY[i]*200);  //draw a point for the updated weights after training
      //println("updated weight " + i + ": " + normClassesX[i] + ", " + normClassesY[i]);
    }
  }
  //check cohesionMetric
    /*for(int i = 0; i < cluster3.size(); i++)
    {
      PVector datapoint = (PVector)cluster3.get(i);
      float sumOfInterClusterDist = 0.0;
      float cohesionMetric = 0.0;
      for(int j = 0; j < cluster3.size(); j++)
      {
        PVector otherDataPoint = (PVector)cluster3.get(j);
        if(i != j)
        {
          //println("datapoint: " + i + " compared with datapoint: " + j);
          float interClusterDist = datapoint.dist(otherDataPoint);
          sumOfInterClusterDist += interClusterDist;
        }
      }
      cohesionMetric = sumOfInterClusterDist/(cluster3.size()-1);
      println("datapoint: " + i + " cohesionMetric is: " + cohesionMetric);
    }*/
    //check seperationMetric between points from cluster 1 and cluster 2
    /*for(int i = 0; i < cluster1.size(); i++)
    {
      float distToOtherClusterPoints = 0.0;
      float sumOfdistToOtherClusterPoints = 0.0;
      float seperationMetric = 0.0;
      PVector datapoint = (PVector)cluster1.get(i);
      for(int j = 0; j < cluster2.size(); j++)
      {
        distToOtherClusterPoints = datapoint.dist((PVector)cluster2.get(j));
        sumOfdistToOtherClusterPoints += distToOtherClusterPoints;
        //println("comparing datapoint: " + i + datapoint + " from cluster 1 to datapoint: " + j + (PVector)cluster2.get(j) + " from cluster 2, distance is : " + distToOtherClusterPoints);
      }
      seperationMetric = sumOfdistToOtherClusterPoints/(cluster2.size()-1);
      //println("seperationMetric for datapoint: " + i + " is " + seperationMetric);
    }*/
    //check seperation metric between one point to other points in another cluster
    /*float dist = 0.0;
    float sumOfDist = 0.0;
    for(int i = 0; i < cluster2.size(); i++)
    {
      PVector datapoint = cluster2.get(i);
      dist = datapoint.dist(cluster1.get(0));
      sumOfDist += dist;
    }
    float seperationMetric = sumOfDist/(cluster2.size()-1);
    println(seperationMetric);*/
  else  //unit updating and silhouette coefficient calculation are the same as with normalized values, values below are for a run without normalization
  {
    for(int i = 0; i < datasetX.length; i++)
    {
      stroke(0,0,0);
      point(datasetX[i]*200,datasetY[i]*200);
    }
    for(int i = 0; i < numOfUnits; i++)
    {
      stroke(255,0,0);
      classesStartX[i] = classesX[i];
      classesStartY[i] = classesY[i];
      println(classesStartX[i]);
      point(classesStartX[i]*200,classesStartY[i]*200);
    }
    while(counter != maxEpochs)
    {
      for(int i = 0; i < datasetX.length; i++)
      {
        PVector dataset = new PVector(datasetX[i],datasetY[i]);
        float net = 10000.0;
        int winner = 0;
        //println("dataset: " + dataset);
        for(int j = 0; j < numOfUnits; j++)
        {
          PVector classes = new PVector(classesX[j],classesY[j]);
          float curNet = dataset.dist(classes);
          if(curNet < net)
          {
            winner = j;
            net = curNet;
          }
        }
        //println("net: " + net);
        //println("winner: " + winner);
        PVector winnerClassVector = new PVector(classesX[winner],classesY[winner]);
        PVector updatedClassVector = winnerClassVector.add(dataset.sub(winnerClassVector).mult(learningRate));
        //println("old weight: " + classesX[winner],classesY[winner]);
        classesX[winner] = updatedClassVector.x;
        classesY[winner] = updatedClassVector.y;
        //println("new weight: " + classesX[winner],classesY[winner]);
      }
      println("epoch: " + counter);
      //println("learning rate: " + learningRate);
      counter++;
      //learningRate -= 0.005;
      learningRate = learningRate/2;
      //calculate silhouette coefficient
    ArrayList<PVector>[] listOfClusters = new ArrayList[numOfUnits];
    for(int i = 0; i < listOfClusters.length; i++)
    {
      listOfClusters[i] = new ArrayList<PVector>();
    }
    ArrayList<PVector>[] listOfClustersCohesionSeperationMetrics = new ArrayList[numOfUnits];
    for(int i = 0; i < listOfClusters.length; i++)
    {
      listOfClustersCohesionSeperationMetrics[i] = new ArrayList<PVector>();
    }
    for(int i = 0; i < normDatasetX.length; i++)
    {
      PVector dataset = new PVector(datasetX[i],datasetY[i]);
      float tempDist = 1000000.0;
      int cluster = 0;
      for(int j = 0; j < numOfUnits; j++)
      {
        PVector clusterLeader = new PVector(classesX[j],classesY[j]);
        float dist = dataset.dist(clusterLeader);
        if(dist < tempDist)
        {
          cluster = j;
          tempDist = dist;
        }
      }
      //println(cluster);
      listOfClusters[cluster].add(dataset);
    }
    for(int i = 0; i < numOfUnits; i++)
    {
      ArrayList cluster = listOfClusters[i];
      //println(cluster);
      if(cluster.size() != 0)
      {
        ArrayList cohesionSeperationCluster = listOfClustersCohesionSeperationMetrics[i];
        for(int j = 0; j < cluster.size(); j++)
        {
          PVector datapoint = (PVector)cluster.get(j);
          float sumOfInterClusterDist = 0.0;
          float cohesionMetric = 0.0;
          for(int k = 0; k < cluster.size(); k++)
          {
            PVector otherDatapoint = (PVector)cluster.get(k);
            if(j != k)
            {
              float interClusterDist = datapoint.dist(otherDatapoint);
              sumOfInterClusterDist += interClusterDist;
            }
          }
          cohesionMetric = sumOfInterClusterDist/(cluster.size()-1);
          //println("cluster: " + i + " datapoint: " + j + " cohesionMetric is: " + cohesionMetric);
          PVector tempV = new PVector(cohesionMetric,1000000);
          cohesionSeperationCluster.add(tempV);
        }
      }
    }
    for(int i = 0; i < numOfUnits; i++)
    {
      ArrayList cluster = listOfClusters[i];
      if(cluster.size() != 0)
      {
        ArrayList cohesionSeperationCluster = listOfClustersCohesionSeperationMetrics[i];
        for(int a = 0; a < numOfUnits; a++)
        {
          ArrayList otherCluster = listOfClusters[a];
          if(otherCluster.size() != 0)
          {
            float seperationMetric = 0.0;
            if(i != a)
            {
              for(int b = 0; b < cluster.size(); b++)
              {
                float distToOtherClusterPoints = 0.0;
                float sumOfDistToOtherClusterPoints = 0.0;
                PVector datapoint = (PVector)cluster.get(b);
                for(int c = 0; c < otherCluster.size(); c++)
                {
                  PVector otherClusterDatapoint = (PVector)otherCluster.get(c);
                  //println("comparing datapoint: " + b + " from cluster: " + i + " with datapoint: " + c + " from cluster: " + a);
                  distToOtherClusterPoints = datapoint.dist(otherClusterDatapoint);
                  sumOfDistToOtherClusterPoints += distToOtherClusterPoints;
                }
                //println(otherCluster.size());
                seperationMetric = sumOfDistToOtherClusterPoints/(otherCluster.size()-1);
                //println("seperationMetric for datapoint: " + b + " from cluster: " + i + " to cluster: " + a + " is " + seperationMetric);
                PVector tempV = (PVector)cohesionSeperationCluster.get(b);
                float tempf = tempV.x;
                float tempSM = tempV.y;
                //println(tempSM);
                //println(seperationMetric);
                if(seperationMetric < tempSM)
                {
                  PVector tempV2 = new PVector(tempf,seperationMetric,0);
                  cohesionSeperationCluster.set(b,tempV2);
                }
              }
            }
          }
        }
        //println("cluster: " + i + cohesionSeperationCluster);
      }
    }
    float finalSilhouetteCoefficient;
    float sumOfSilhouetteCoefiicient = 0.0;
    for(int i = 0; i < numOfUnits; i++)
    {
      ArrayList cohesionSeperationCluster = listOfClustersCohesionSeperationMetrics[i];
      if(cohesionSeperationCluster.size() != 0)
      {
        for(int j = 0; j < cohesionSeperationCluster.size(); j++)
        {
          PVector cohesionSeperationMetric = (PVector)cohesionSeperationCluster.get(j);
          //println(cohesionSeperationMetric);
          float a = cohesionSeperationMetric.x;
          float b = cohesionSeperationMetric.y;
          //println(a + "," + b);
          float silhouetteCoefficient = (b-a)/max(b,a);
          sumOfSilhouetteCoefiicient += silhouetteCoefficient;
          //println(silhouetteCoefficient);
        }
      }
    }
    finalSilhouetteCoefficient = sumOfSilhouetteCoefiicient/datasetX.length;
    finalSilhouetteCoefficient = roundDecimal(finalSilhouetteCoefficient,4);
    println("silhouette coefficient: " + finalSilhouetteCoefficient);
    }
    for(int i = 0; i < numOfUnits; i++)
    {
      stroke(0,255,0);
      point(classesX[i]*200,classesY[i]*200);
    }
  }
}
