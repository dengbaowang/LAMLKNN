/*    This program is implenment as a extension of Mulan package, which can be download in Mulan website.
 *    This implement is based on ML-kNN algorithm
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.classifier.lazy;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans; 

/**
 <!-- globalinfo-start -->
 * Class implementing the LAML-kNN (Multi-Label k Nearest Neighbours) algorithm.<br>
 * <br>
 * For more information, see<br>
 * 
 */
@SuppressWarnings("serial")
public class LAMLkNN extends MultiLabelKNN {

    /**
     * Smoothing parameter controlling the strength of uniform prior <br>
     * (Default value is set to 1 which yields the Laplace smoothing).
     */
    protected double smooth;
    /**
     * A table holding the prior probability for an instance to belong in each
     * class
     */
    protected DistanceFunction dfunc = new EuclideanDistance();
    private Clusterer clusterer;
    protected int numOfClusters;
    private double[] CentersK;
    private double[] PriorK;
    private double[][] PriorProbabilities;
    /**
     * A table holding the prior probability for an instance not to belong in
     * each class
     */
    private double[][] PriorNProbabilities;
    /**
     * A table holding the probability for an instance to belong in each
     * class<br> given that i:0..k of its neighbors belong to that class
     */
    private double[][][] CondProbabilities;
    /**
     * A table holding the probability for an instance not to belong in each
     * class<br> given that i:0..k of its neighbors belong to that class
     */
    private double[][][] CondNProbabilities;

    /**
     * @param numOfNeighbors : the number of neighbors
     * @param smooth : the smoothing factor
     */
    public LAMLkNN(int numOfNeighbors,int numOfClusters,double smooth) {
        super(numOfNeighbors);
        this.numOfClusters = numOfClusters;
        this.smooth = smooth;
        SimpleKMeans kmeans = new SimpleKMeans();
        try {
			kmeans.setNumClusters(numOfClusters);
			((NormalizableDistance) dfunc).setDontNormalize(false);
			kmeans.setDistanceFunction(dfunc);
			kmeans.setPreserveInstancesOrder(true);
	        clusterer = kmeans;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
    }

    /**
     * The default constructor
     */
    public LAMLkNN() {
        super();
        this.smooth = 1.0;
    }

    public String globalInfo() {
        return "Class implementing the ML-kNN (Multi-Label k Nearest Neighbours) algorithm." + "\n\n" + "For more information, see\n\n" + getTechnicalInformation().toString();
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Deng-Bao Wang");
        result.setValue(Field.TITLE, "A locally adaptive k Nearest Neighbour algorithm");
        result.setValue(Field.JOURNAL, "Pattern Recogn.");
        result.setValue(Field.VOLUME, "");
        result.setValue(Field.NUMBER, "");
        result.setValue(Field.YEAR, "2018");
        result.setValue(Field.ISSN, "");
        result.setValue(Field.PAGES, "");
        result.setValue(Field.PUBLISHER, "");
        result.setValue(Field.ADDRESS, "");

        return result;
    }

    @Override
    protected void buildInternal(MultiLabelInstances train) throws Exception {
        super.buildInternal(train);
        CentersK = new double[numOfClusters];
        PriorK = new double[numOfClusters];
        PriorProbabilities = new double[numLabels][numOfClusters];
        PriorNProbabilities = new double[numLabels][numOfClusters];
        CondProbabilities = new double[numLabels][numOfNeighbors + 1][numOfClusters];
        CondNProbabilities = new double[numLabels][numOfNeighbors + 1][numOfClusters]; 
        Instances removedInstances = RemoveAllLabels.transformInstances(train);
        clusterer.buildClusterer(removedInstances);
        ComputePriorK();
        ComputePrior();
        ComputeCond();

        if (getDebug()) {
            System.out.println("Computed Prior Probabilities");
            for (int i = 0; i < numLabels; i++) {
                System.out.println("Label " + (i + 1) + ": " + PriorProbabilities[i]);
            }
            System.out.println("Computed Posterior Probabilities");
            for (int i = 0; i < numLabels; i++) {
                System.out.println("Label " + (i + 1));
                for (int j = 0; j < numOfNeighbors + 1; j++) {
                    System.out.println(j + " neighbours: " + CondProbabilities[i][j]);
                    System.out.println(j + " neighbours: " + CondNProbabilities[i][j]);
                }
            }
        }
    }

    /**
     * Computing Prior and PriorN Probabilities for each class of the training
     * set
     * @throws Exception 
     */
    //private void ComputeCentersK() {
    	//CentersK=
    //}
    private void ComputePriorK() {
    	//System.out.println(Arrays.toString(((SimpleKMeans) clusterer).getClusterSizes()));
    	//System.out.println(train.numInstances());
    	for(int c=0;c<numOfClusters;c++) {
    		//System.out.println(c);
    		PriorK[c]=(double)((SimpleKMeans) clusterer).getClusterSizes()[c]/train.numInstances();
    		//System.out.println(PriorK[c]);
    	}
    }
    private void ComputePrior() throws Exception {
    	int[] whichK=((SimpleKMeans) clusterer).getAssignments();
    	for (int c = 0; c < numOfClusters; c++) {
	        for (int i = 0; i < numLabels; i++) {
	            int temp_Ci = 0;
	            for (int j = 0; j < train.numInstances(); j++) {
	            	if (whichK[j]==c) {
		                double value = Double.parseDouble(train.attribute(labelIndices[i]).value(
		                        (int) train.instance(j).value(labelIndices[i])));
		                if (Utils.eq(value, 1.0)) {
		                    temp_Ci++;
		                }
	            	}
	            }
	            PriorProbabilities[i][c] = (smooth + temp_Ci) / (smooth * 2 + ((SimpleKMeans) clusterer).getClusterSizes()[c]);
	            //System.out.print(PriorProbabilities[i][c]+" ");
	            PriorNProbabilities[i][c] = 1 - PriorProbabilities[i][c];
	        }
	        //System.out.println();
    	}
    }

    /**
     * Computing Cond and CondN Probabilities for each class of the training set
     *
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private void ComputeCond() throws Exception {
    	int[] whichK=((SimpleKMeans) clusterer).getAssignments();
    	for (int c=0; c < numOfClusters; c++) {
	        int[][] temp_Ci = new int[numLabels][numOfNeighbors + 1];
	        int[][] temp_NCi = new int[numLabels][numOfNeighbors + 1];
	        //System.out.println(temp_Ci[0][0]+"ffdfdfd");
	        for (int i = 0; i < train.numInstances(); i++) {
	        	if (whichK[i]==c) {
		            Instances knn = new Instances(lnn.kNearestNeighbours(train.instance(i), numOfNeighbors));
		
		            // now compute values of temp_Ci and temp_NCi for every class label
		            for (int j = 0; j < numLabels; j++) {
		
		                int aces = 0; // num of aces in Knn for j
		                for (int k = 0; k < numOfNeighbors; k++) {
		                    double value = Double.parseDouble(train.attribute(labelIndices[j]).value(
		                            (int) knn.instance(k).value(labelIndices[j])));
		                    if (Utils.eq(value, 1.0)) {
		                        aces++;
		                    }
		                }
		                // raise the counter of temp_Ci[j][aces] and temp_NCi[j][aces] by 1
		                if (Utils.eq(Double.parseDouble(train.attribute(labelIndices[j]).value(
		                        (int) train.instance(i).value(labelIndices[j]))), 1.0)) {
		                    temp_Ci[j][aces]++;
		                } else {
		                    temp_NCi[j][aces]++;
		                }
		            }
	        	}
	        }
	
	        // compute CondProbabilities[i][..] for labels based on temp_Ci[]
	        for (int i = 0; i < numLabels; i++) {
	            int temp1 = 0;
	            int temp2 = 0;
	            for (int j = 0; j < numOfNeighbors + 1; j++) {
	                temp1 += temp_Ci[i][j];
	                temp2 += temp_NCi[i][j];
	            }
	            for (int j = 0; j < numOfNeighbors + 1; j++) {
	                CondProbabilities[i][j][c] = (smooth + temp_Ci[i][j]) / (smooth * (numOfNeighbors + 1) + temp1);
	                CondNProbabilities[i][j][c] = (smooth + temp_NCi[i][j]) / (smooth * (numOfNeighbors + 1) + temp2);
	            }
	        }
    	}
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] confidences = new double[numLabels];
        boolean[] predictions = new boolean[numLabels];
        Instance newInstance = RemoveAllLabels.transformInstance(instance, labelIndices);
        int cluster = clusterer.clusterInstance(newInstance);
        Instances knn = null;
        try {
            knn = new Instances(lnn.kNearestNeighbours(instance, numOfNeighbors));
        } catch (Exception ex) {
            Logger.getLogger(MLkNN.class.getName()).log(Level.SEVERE, null, ex);
        }

        for (int i = 0; i < numLabels; i++) {
            // compute sum of aces in KNN
            int aces = 0; // num of aces in Knn for i
            for (int k = 0; k < numOfNeighbors; k++) {
                double value = Double.parseDouble(train.attribute(labelIndices[i]).value(
                        (int) knn.instance(k).value(labelIndices[i])));
                if (Utils.eq(value, 1.0)) {
                    aces++;
                }
            }
            double Prob_in =PriorK[cluster] * PriorProbabilities[i][cluster] * CondProbabilities[i][aces][cluster];
            double Prob_out =PriorK[cluster] * PriorNProbabilities[i][cluster] * CondNProbabilities[i][aces][cluster];
            if (Prob_in > Prob_out) {
                predictions[i] = true;
            } else if (Prob_in < Prob_out) {
                predictions[i] = false;
            } else {
                Random rnd = new Random();
                predictions[i] = (rnd.nextInt(2) == 1) ? true : false;
            }
            // ranking function
            confidences[i] = Prob_in / (Prob_in + Prob_out);
        }
        MultiLabelOutput mlo = new MultiLabelOutput(predictions, confidences);
        return mlo;
    }
}
