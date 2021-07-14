/*
 *    kNN.java
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.lazy;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import com.github.javacliparser.FlagOption;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.lazy.neighboursearch.EuclideanDistance;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import scala.util.parsing.combinator.testing.Str;

/**
 * k Nearest Neighbor.<p>
 *
 * Valid options are:<p>
 *
 * -k number of neighbours <br> -m max instances <br> 
 *
 * @author Jesse Read (jesse@tsc.uc3m.es)
 * @version 03.2012
 */
public class kNN extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

	public IntOption kOption = new IntOption( "k", 'k', "The number of neighbors", 10, 1, Integer.MAX_VALUE);

	public IntOption limitOption = new IntOption( "limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

	public FlagOption weightedVote = new FlagOption("weighted", 'W', "Use weighted votes");

	public FlagOption standardizeData = new FlagOption("standardize", 's', "Standardize the streaming data");

	public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption(
            "nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{
                "LinearNN", "KDTree"},
            new String[]{"Brute force search algorithm for nearest neighbour search. ",
                "KDTree search algorithm for nearest neighbour search"
            }, 0);

	int C = 0;

    @Override
    public String getPurposeString() {
        return "kNN: special.";
    }

    protected Instances window;
    
	// Counter
	int counter = 0;
	// Sum array.
	double sumArray[];
	// Sum of squares array.
	double sumSquareArray[];
	// Mean array.
	double meanArray[];
	// Variance array.
	double varianceArray[];
	// Standardized array.
	double standardArray[];

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			this.window = new Instances(context,0); //new StringReader(context.toString())
			this.window.setClassIndex(context.classIndex());
		} catch(Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

    @Override
    public void resetLearningImpl() {
		this.window = null;
    }

	/**
	 *	The method to standardize the streaming data.
	 * @param inst The instance to be processed.
	 */
	protected void standardizeData(Instance inst){
		// Check if the two sum arrays are null/empty.
		if (sumArray == null) {								// If NOT executed then number of instances > 0.
			// Fill the sum arrays with the first entry.
			sumArray = inst.toDoubleArray();
			sumSquareArray = inst.toDoubleArray();
			// INITIALIZE THESE ARRAYS WITH SUITABLE DIMENSIONS: DO NOT USE IN CONDITION.
			meanArray = inst.toDoubleArray();					// Last element redundant.
			varianceArray = inst.toDoubleArray();				// Last element redundant.
			standardArray = inst.toDoubleArray();				// Last element redundant.
			// Loop through the sum square array (minus class).
			for (int sq = 0; sq < sumSquareArray.length - 1; sq++) {
				// Fill the array with square values.
				sumSquareArray[sq] = Math.pow(sumSquareArray[sq], 2.0);
			}
		}
		// Check if there is already an instance stored.
		if (this.window.numInstances() > 0) {				// If executed, then at least one instance has appeared.
			// Loop through each feature/attribute (minus class).
			for (int att = 0; att < inst.numAttributes() - 1; att++){
				// Get the sum & sum of squares.
				sumArray[att] += inst.value(att);
				sumSquareArray[att] += Math.pow(inst.value(att), 2);
			}
			// Print increment message.
			System.out.println("Standardizing instance...");
			// Loop through the array dimensions (minus class).
			for (int m = 0; m < meanArray.length - 1; m++){
				// Set the mean array with the mean values.
				meanArray[m] = sumArray[m] / (counter);
				// Set the variance array with the variance values.
				varianceArray[m] = (sumSquareArray[m] - (Math.pow(sumArray[m], 2.0) / counter)) / (counter - 1);
				// Set the standardized array.
				standardArray[m] = (inst.value(m) - meanArray[m]) / Math.sqrt(varianceArray[m]);
				// Set the new instance value.
				inst.setValue(m, standardArray[m]);
			}
		}
		// Otherwise the instance is the first to appear.
		else {
			// Loop through each feature/attribute (minus class).
			for (int att = 0; att < inst.numAttributes() - 1; att++){
				// Set the first instance values to have 0.
				inst.setValue(att, 0);
			}
		}
	}

    @Override
    public void trainOnInstanceImpl(Instance inst) {
		if (inst.classValue() > C)
			C = (int)inst.classValue();
		if (this.window == null) {
			this.window = new Instances(inst.dataset());
		}
		if (this.limitOption.getValue() <= this.window.numInstances()) {
			this.window.delete(0);
		}
		this.window.add(inst);
    }

	@Override
    public double[] getVotesForInstance(Instance inst) {
		// Increment counter.
		counter++;
		System.out.println("Instance number: " + counter);
		// Check if the standardize data option has been set.
		if (standardizeData.isSet()){
			// Call the method.
			/////////// NB: This method replaces the current instance feature values with standardized feature values.
			/////////// The standardized instances will automatically be applied to trainOnInstanceImpl.
			standardizeData(inst);
		}
		// Votes.
		double v[] = new double[C+1];
		// Instance values.
		double instValues[] = new double[inst.numAttributes()];
		// Neighbour values.
		double neighbourValues[] = new double[inst.numAttributes()];
		try {
			NearestNeighbourSearch search;
			if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
				search = new LinearNNSearch(this.window);  
			} else {
				search = new KDTree();
				search.setInstances(this.window);
			}
			// Check if there is an instance stored and the weighted vote is set.
			if (this.window.numInstances()>0 && weightedVote.isSet()) {
				// Store the current instance values into an array.
				instValues = inst.toDoubleArray();
				// Search the k-nearest neighbours.
				Instances neighbours = search.kNearestNeighbours(inst,Math.min(kOption.getValue(),this.window.numInstances()));
				// Loop through the number of instances in the neighbourhood.
				for(int i = 0; i < neighbours.numInstances(); i++) {
					// Get the current neighbour value.
					neighbourValues = neighbours.instance(i).toDoubleArray();
					// Calculate the euclidean distance between neighbour and instance.
					double sum = 0.0;
					double dist = 0.0;
					// Loop through each attribute (minus class) to be used in calculating the euclidean distance.
					for (int e = 0; e < instValues.length - 1; e++) {
						// Add to the current sum.
						sum = sum + Math.pow((neighbourValues[e] - instValues[e]), 2.0);
					}
					// Distance.
					dist = Math.sqrt(sum);
					// Store the distance based on class value into the v array.
					v[(int)neighbours.instance(i).classValue()] += dist;
				}
				// Take 1 / distance.
				for (int i = 0; i < v.length; i++) {
					if (v[i] > 0) {
						v[i] = 1 / v[i];
					}
				}
			}
			else if (this.window.numInstances()>0 && !weightedVote.isSet()) {
				Instances neighbours = search.kNearestNeighbours(inst,Math.min(kOption.getValue(),this.window.numInstances()));
				for(int i = 0; i < neighbours.numInstances(); i++) {
					v[(int)neighbours.instance(i).classValue()]++;
				}
			}
		} catch(Exception e) {
			//System.err.println("Error: kNN search failed.");
			//e.printStackTrace();
			//System.exit(1);
			return new double[inst.numClasses()];
		}
		return v;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    public boolean isRandomizable() {
        return false;
    }
}