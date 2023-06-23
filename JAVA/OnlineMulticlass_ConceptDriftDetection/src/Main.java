package src;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import moa.classifiers.AbstractClassifier;
import moa.evaluation.ClassificationPerformanceEvaluator;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.streams.ArffFileStream;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	/**
	 * @param args
	 * @throws Exception 
	 */
	//data stream properties
	public static double[][] classPercentage;// class percentage of each class (time decayed) at each time step. 1st index - number of total time steps; 2nd index - number of classes in data streams
	public static boolean imbalance;// whether the current data stream is imbalanced
	public static int numClasses;//number of classes
	public static int[] numInstances;//number of instances of each class
	public static ArrayList<Integer> classIndexMinority = new ArrayList<Integer>();//class indice of current minority classes
	public static ArrayList<Integer> classIndexMajority = new ArrayList<Integer>();//class indice of current majority classes
	public static ArrayList<Integer> classIndexNormal = new ArrayList<Integer>();//class indice of other classes
	// performance at current time step
	public static double[] currentClassRecall_decay;//time decayed recall value of each class at current time step at current run
	public static double[][][] classRecall_window;// recall value of each class within current window at each time step at each run
	public static double[][] gmean_window;//gmean of window recalls at each time step at each run
	public static double[][] PMAUC_window;//PMAUC within current window at each time step at each run
	public static double[][] WAUC_window;//Provost's weighted multiclass AUC, within current window at each time step at each run
	public static double[][] EWAUC_window;//Provost's weighted multiclass AUC but with Equal Weights for all classes, within current window at each time step at each run

	public static double[][] gmean_window1;//gmean of window recalls at each time step at each run
	public static double[][] PMAUC_window1;//PMAUC within current window at each time step at each run
	public static double[][] WAUC_window1;//Provost's weighted multiclass AUC, within current window at each time step at each run
	public static double[][] EWAUC_window1;
	public static double[][] gmean_window2;//gmean of window recalls at each time step at each run
	public static double[][] PMAUC_window2;//PMAUC within current window at each time step at each run
	public static double[][] WAUC_window2;//Provost's weighted multiclass AUC, within current window at each time step at each run
	public static double[][] EWAUC_window2;
	public static double[] MAUC;//Hand and Til's MAUC at each run, over all data
	
	public static int[] numDrifts; // number of detected drifts at each run
	public static int[][] driftLocations;//time steps of where drift is detected at each run



	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		int window_size = 200;
		String folder = "C:\\Users\\joseg\\Desktop\\MCCD\\Real-World Data\\";
		String datafile = folder + "Gas Sensor Array Drift Data\\batches1-4_8-10.arff";
		String resultfile = folder + "Result\\ChangingEveryWindowPasses_HYBRID_EWAUC_window200_lastbatchp_average_perrun.txt";
		String sampleMode = "HYBRID";
		int whichDDM = 2;//1: PMAUC-PH, 2: EWAUC-PH, 3: WAUC-PH and 4: GM-PH; other numbers: no drift detection.
		int numRun = 100;
		int usedMetric = 2;
		
		//ins: just for setting data properties and initialising performance arrays
		System.out.println(datafile);
		DataSource source = new DataSource(datafile);
		Instances ins = source.getDataSet();
		if (ins.classIndex() == -1)
			ins.setClassIndex(ins.numAttributes() - 1);
		int numTimeStep = ins.numInstances();   
		numClasses = ins.numClasses(); 
		Instance fistInst = ins.instance(0);// get the first instance of the data stream, for initializing MLP
		
		//initialise performance metrics
		classRecall_window = new double[numTimeStep][numClasses][numRun];
	    gmean_window = new double[numTimeStep][numRun];
	    PMAUC_window = new double[numTimeStep][numRun];
	    WAUC_window = new double[numTimeStep][numRun];
	    EWAUC_window = new double[numTimeStep][numRun];
		MAUC = new double[numRun];

		gmean_window1 = new double[numTimeStep][numRun];
		PMAUC_window1 = new double[numTimeStep][numRun];
		WAUC_window1 = new double[numTimeStep][numRun];
		EWAUC_window1 = new double[numTimeStep][numRun];

		gmean_window2 = new double[numTimeStep][numRun];
		PMAUC_window2 = new double[numTimeStep][numRun];
		WAUC_window2  = new double[numTimeStep][numRun];
		EWAUC_window2 = new double[numTimeStep][numRun];
		
		//initialise variables for concept drift dection
		numDrifts = new int[numRun];
	    driftLocations = new int[numRun][PHT.MAX_DRIFTNUM];	    
	    for(int i = 0; i < numRun; i++) {
	    	for(int j = 0; j < PHT.MAX_DRIFTNUM; j++)
	    		driftLocations[i][j] = -1;
	    }
	    
		
		for(int run = 0; run < numRun; run++){
			System.out.println("Run " + (run+1));

			// Obtain data stream
			ArffFileStream data = new ArffFileStream(datafile,-1);
			data.prepareForUse();

			classPercentage = new double[numTimeStep][numClasses];
			numInstances = new int[numClasses];
			currentClassRecall_decay = new double[numClasses];

			// local variables
			double delta1 = 0.4;
			double delta2 = 0.3;
			double sizedecayfactor = 0.9;//theta
			double recalldecayfactor = 0.9;//theta'
			int numSamples_Total = 0; // number of processed samples from the beginning
			boolean isCorrect = true;
			int predictedLabel, realLabel;

			// initialize online models
			OzaBag model = (OzaBag) initializeOnlineModel(run, data, fistInst);
			OzaBag model2 = (OzaBag) initializeOnlineModel(run, data, fistInst);
			
			// initialise drift detection
			PHT phtdrifter = new PHT(ins);
			//PageHinkleyDM phtdrifter2 = new PageHinkleyDM();//this is equivalent to PHT 

			// choose an evaluator for performance assessment
			ClassificationPerformanceEvaluator evaluator = new WindowClassificationPerformanceEvaluator();
			
			// initialize the PMAUC (2 of em) calculation class
			AUCClassificationPerformanceEvaluator_mclass pmauc = new AUCClassificationPerformanceEvaluator_mclass();
			pmauc.numClasses = numClasses;
			pmauc.widthOption = window_size;
			pmauc.aucEstimator = pmauc.new Estimator(pmauc.widthOption);

			AUCClassificationPerformanceEvaluator_mclass pmauc1 = new AUCClassificationPerformanceEvaluator_mclass();
			pmauc1.numClasses = numClasses;
			pmauc1.widthOption = window_size;
			pmauc1.aucEstimator = pmauc1.new Estimator(pmauc1.widthOption);

			AUCClassificationPerformanceEvaluator_mclass pmauc2 = new AUCClassificationPerformanceEvaluator_mclass();
			pmauc2.numClasses = numClasses;
			pmauc2.widthOption = window_size;
			pmauc2.aucEstimator = pmauc2.new Estimator(pmauc2.widthOption);

			int end_of_last_window = window_size;
			int justChanged=0; // variable for testing purposes
			// online training loop: test the current instance first, then used to update the learner (prequential)
			while(data.hasMoreInstances()){

				Instance trainInst = data.nextInstance();

				double[] prediction = model.getVotesForInstance(trainInst);
				if(end_of_last_window>window_size) {

					int last_window_choice = get_last_window_choice(end_of_last_window, window_size, usedMetric, run);
					if(justChanged==1) {
						//System.out.println(last_window_choice);
						justChanged=0;
					}
					if (last_window_choice == 2)
						prediction = model2.getVotesForInstance(trainInst);
				}
				evaluator.addResult(trainInst, prediction);
				pmauc.addResult(trainInst,prediction);
				predictedLabel = Utils.maxIndex(prediction);
				
				prediction = model.getVotesForInstance(trainInst);
				pmauc1.addResult(trainInst,prediction);

				prediction = model2.getVotesForInstance(trainInst);
				pmauc2.addResult(trainInst,prediction);

				realLabel = (int)trainInst.classValue();
				numInstances[realLabel]++;
				if(predictedLabel==realLabel) isCorrect = true;
				else isCorrect = false;
				numSamples_Total ++;
/*				for(int i = 0; i < prediction.length; i++) {
					System.out.print(prediction[i] + ", ");
				}
				System.out.println(isCorrect);*/

				// update class percentages
				updateClassPercentage(realLabel, numSamples_Total, sizedecayfactor);

				// train online model
				if(sampleMode.equals("MOOB"))
					MOOB_adaptive(trainInst, model, numSamples_Total);
				else if(sampleMode.equals("MUOB"))
					MUOB_adaptive(trainInst, model, numSamples_Total);
				else if(sampleMode.equals("HYBRID")){
					MOOB_adaptive(trainInst, model, numSamples_Total);
					MUOB_adaptive(trainInst, model2, numSamples_Total);
				}
				else
					model.trainOnInstance(trainInst);


				// update time decayed recall
				updateDecayRecall(realLabel, isCorrect, recalldecayfactor);	
				// class imbalance detection
				imbalanceStatus(delta1, delta2, numSamples_Total);
				
				// Output metrics
				double pm = pmauc.aucEstimator.getPMAUC();
				double gm = pmauc.aucEstimator.getGmean();
				double wa = pmauc.aucEstimator.getWeightedAUC();
				double ewa = pmauc.aucEstimator.getEqualWeightedAUC();
				PMAUC_window[numSamples_Total-1][run] = pm;
				gmean_window[numSamples_Total-1][run] = gm;
				WAUC_window[numSamples_Total-1][run] = wa;
				EWAUC_window[numSamples_Total-1][run] = ewa;

				pm = pmauc1.aucEstimator.getPMAUC();
				gm = pmauc1.aucEstimator.getGmean();
				wa = pmauc1.aucEstimator.getWeightedAUC();
				ewa = pmauc1.aucEstimator.getEqualWeightedAUC();
				PMAUC_window1[numSamples_Total-1][run] = pm;
				gmean_window1[numSamples_Total-1][run] = gm;
				WAUC_window1[numSamples_Total-1][run] = wa;
				EWAUC_window1[numSamples_Total-1][run] = ewa;

				pm = pmauc2.aucEstimator.getPMAUC();
				gm = pmauc2.aucEstimator.getGmean();
				wa = pmauc2.aucEstimator.getWeightedAUC();
				ewa = pmauc2.aucEstimator.getEqualWeightedAUC();
				PMAUC_window2[numSamples_Total-1][run] = pm;
				gmean_window2[numSamples_Total-1][run] = gm;
				WAUC_window2[numSamples_Total-1][run] = wa;
				EWAUC_window2[numSamples_Total-1][run] = ewa;

				for(int c = 0; c < numClasses; c++)
					classRecall_window[numSamples_Total-1][c][run] = pmauc.aucEstimator.getRecall(c);
				
				//////////////////////Apply Drift detection method//////////////////////////
				if(numSamples_Total>window_size) {
					if(numSamples_Total%window_size==0)
						end_of_last_window=numSamples_Total;
					switch(whichDDM){
					case 1:
						applyPHT(model, phtdrifter, pmauc, trainInst, pm, numSamples_Total);
						applyPHT(model2, phtdrifter, pmauc, trainInst, pm, numSamples_Total);
						/*
						phtdrifter2.input(1-pm);
						if (phtdrifter2.isChangeDetected) {
							model.resetLearning();
							driftLocations[run][numDrifts[run]]=numSamples_Total;
							numDrifts[run]++;
						}*/
						break;
					case 2:
						applyPHT(model, phtdrifter, pmauc, trainInst, ewa, numSamples_Total);
						applyPHT(model2, phtdrifter, pmauc, trainInst, pm, numSamples_Total);
						/*phtdrifter2.input(1-ewa);
						if (phtdrifter2.isChangeDetected) {
							model.resetLearning();
							driftLocations[run][numDrifts[run]]=numSamples_Total;
							numDrifts[run]++;
						}*/
						break;
					case 3: 
						applyPHT(model, phtdrifter, pmauc, trainInst, wa, numSamples_Total);
						applyPHT(model2, phtdrifter, pmauc, trainInst, pm, numSamples_Total);
						/*phtdrifter2.input(1-wa);
						if (phtdrifter2.isChangeDetected) {
							model.resetLearning();
							driftLocations[run][numDrifts[run]]=numSamples_Total;
							numDrifts[run]++;
						}*/
						break;
					case 4:
						applyPHT(model, phtdrifter, pmauc, trainInst, gm, numSamples_Total);
						applyPHT(model2, phtdrifter, pmauc, trainInst, pm, numSamples_Total);
						/*phtdrifter2.input(1-gm);
						if (phtdrifter2.isChangeDetected) {
							model.resetLearning();
							driftLocations[run][numDrifts[run]]=numSamples_Total;
							numDrifts[run]++;
						}*/
					default: break;
					}
				}
			}//while
			numDrifts[run] = phtdrifter.numDrift;
			driftLocations[run] = phtdrifter.driftLocation;
			System.out.print(numDrifts[run] + "\t");
			for(int j = 0; j < driftLocations[run].length; j++){
				if(driftLocations[run][j] != -1) System.out.print(driftLocations[run][j] + "\t");
				else break;
			}
			System.out.println();
			phtdrifter = null;
			evaluator = null;
			
		}//for numRun
		
		System.out.println(average(numDrifts));
		//printPerformance(resultfile,numTimeStep);
		printPerformancePerRun(resultfile, numRun);
	}

	/**Initialize Online Bagging*/
	public static AbstractClassifier initializeOnlineModel(int seed, ArffFileStream data, Instance fistInst){
		OzaBag model = new OzaBag();
		//model.baseLearnerOption.setValueViaCLIString("functions.Perceptron");//default learning rate is 1
		model.baseLearnerOption.setValueViaCLIString("src.OnlineMultilayerPerceptron");//default learning rate is 0.3
		//model.baseLearnerOption.setValueViaCLIString("bayes.NaiveBayes");
		//model.baseLearnerOption.setValueViaCLIString("trees.HoeffdingTree");//default of OzaBag

		model.ensembleSizeOption.setValue(11);
		model.randomSeedOption.setValue(seed);//model.randomSeedOption.setValue((int)System.currentTimeMillis());
		if(model.baseLearnerOption.getValueAsCLIString().equals("src.OnlineMultilayerPerceptron")){
			model.firtInst = fistInst;
		}
		model.setModelContext(data.getHeader());
		model.prepareForUse();
		return model;
	}

	//Multi-class Oversampling Online Bagging using adaptive sampling rates
	public static void MOOB_adaptive(Instance currentInstance, OzaBag model, int numSamplesTotal){
		Integer classLabel = (int)currentInstance.classValue();
		double lambda = 1.0;
		int cp_max = Utils.maxIndex(classPercentage[numSamplesTotal-1]);
		model.trainOnInstanceImpl(currentInstance, (double)lambda*cp_max/classPercentage[numSamplesTotal-1][classLabel]);
	}

	// Multi-class Oversampling Online Bagging using fixed sampling rates
	public static void MOOB_fix(Instance currentInstance, OzaBag model, int numSamplesTotal){
		Integer classLabel = (int)currentInstance.classValue();
		double lambda = 1.0;
		double[] samplingRates = {76/70, 76/76, 76/17, 76/13, 76/9, 76/29};
		model.trainOnInstanceImpl(currentInstance, (double)lambda*classPercentage[numSamplesTotal-1][classLabel]*samplingRates[classLabel]);
	}

	//Multi-class Undersampling Online Bagging using adaptive sampling rates
	public static void MUOB_adaptive(Instance currentInstance, OzaBag model, int numSamplesTotal){
		Integer classLabel = (int)currentInstance.classValue();//the class label index
		double lambda = 1.0;
		int cp_min = Utils.minIndex(classPercentage[numSamplesTotal-1]);
		double rate = (double)classPercentage[numSamplesTotal-1][cp_min]/classPercentage[numSamplesTotal-1][classLabel];
		if(rate < 0.01)
			model.trainOnInstanceImpl(currentInstance, (double)lambda*0.01);
		else
			model.trainOnInstanceImpl(currentInstance, (double)lambda*rate);
	}

	//Multi-class Undersampling Online Bagging using fixed sampling rates
	public static void MUOB_fix(Instance currentInstance, OzaBag model, int numSamplesTotal){
		Integer classLabel = (int)currentInstance.classValue();
		double lambda = 1.0;
		double[] samplingRates = {9/70, 9/76, 9/17, 9/13, 9/9, 9/29};
		model.trainOnInstanceImpl(currentInstance, (double)lambda*classPercentage[numSamplesTotal-1][classLabel]*samplingRates[classLabel]);
	}

	/**class imbalance detection method*/
	public static void imbalanceStatus(double delta1, double delta2, int numSamplesTotal){
		int[] classIndexAscend = Utils.sort(classPercentage[numSamplesTotal-1]);
		classIndexMinority.clear();
		classIndexMajority.clear();
		classIndexNormal.clear();

		for(int m = 0; m < numClasses-1; m++){
			if(numInstances[classIndexAscend[m]]==0) continue;//start from the non-zero size class
			for(int n = m+1; n < numClasses; n++){
				if((classPercentage[numSamplesTotal-1][classIndexAscend[n]]-classPercentage[numSamplesTotal-1][classIndexAscend[m]] > delta1) &&
						(currentClassRecall_decay[classIndexAscend[n]]-currentClassRecall_decay[classIndexAscend[m]] > delta2)){
					//classIndexAscend[m] is the minority and classIndexAscend[n] is the majority
					if(!classIndexMinority.contains(classIndexAscend[m])){
						classIndexMinority.add(classIndexAscend[m]);
						//System.out.println("Class "+classIndexAscend[m]+" is added to the minority");
					}
					if(!classIndexMajority.contains(classIndexAscend[n])){
						classIndexMajority.add(classIndexAscend[n]);
						//System.out.println("Class "+classIndexAscend[n]+" is added to the majority");
					}
				}
			}
		}
		for(int k = 0; k < numClasses; k++){
			if(numInstances[classIndexAscend[k]]==0) continue;//start from the non-zero size class
			while(classIndexMinority.contains(classIndexAscend[k]) && 
					classIndexMajority.contains(classIndexAscend[k])){
				classIndexMajority.remove(classIndexAscend[k]);
			}
			if((!classIndexMinority.contains(classIndexAscend[k])) && 
					(!classIndexMajority.contains(classIndexAscend[k])))
				classIndexNormal.add(classIndexAscend[k]);
		}
		if(classIndexMinority.isEmpty() && classIndexMajority.isEmpty())
			imbalance = false;
		else
			imbalance = true;
	}


	/**update percentage of classes at each time step with time decay*/
	public static void updateClassPercentage(int realLabel, int numSamplesTotal, double sizedecayfactor){
		if(numSamplesTotal >1){
			for(int t = 0; t < numClasses; t++){
				if(t==realLabel)
					classPercentage[numSamplesTotal-1][t] = classPercentage[numSamplesTotal-2][t]*sizedecayfactor+(1-sizedecayfactor);
				else
					classPercentage[numSamplesTotal-1][t] = classPercentage[numSamplesTotal-2][t]*sizedecayfactor;
			}
		}
		else{
			classPercentage[numSamplesTotal-1][realLabel] = 1;
		}
	}

	/**update time decayed recall of classes at each time step*/
	public static void updateDecayRecall(int realLabel, boolean isCorrect, double recalldecayfactor){
		if(isCorrect && numInstances[realLabel]==1)
			currentClassRecall_decay[realLabel] = 1;
		else if(isCorrect)
			currentClassRecall_decay[realLabel] = currentClassRecall_decay[realLabel]*recalldecayfactor+(1-recalldecayfactor);
		else if(!isCorrect)
			currentClassRecall_decay[realLabel] = currentClassRecall_decay[realLabel]*recalldecayfactor;
	}

	/**check whether the number is an element of the array
	 * return true if it is.*/
	public static boolean inArray(int[] array, int number){
		for(int i = 0; i < array.length; i++){
			if (number==array[i])
				return true;
		}
		return false;
	}

	/**
	 * Calculate consistency and discriminacy of PMAUC compared to G-mean. Definition is in Brzezinski's paper 
	 * "Prequential AUC: properties of the area under the ROC curve for data streams with concept drift".
	 * Random sample 2 pairs of PMAUC and Gmean at time steps a and b for 10000 times. 
	 * @param f mean_PMAUC
	 * @param g mean_gmean
	 * @param window window size, such as 500.
	 * @return consistency r/(r+s) and discriminancy p/q*/
	public static double[] calculateConsistencyDiscriminancy(double[] f, double[] g, int window){
		double cst = 0.0, dcn = 0.0;//initialise consistency and discriminancy
		int max = f.length;
		Random rand = new Random(); 
		
		int r = 0, s = 0, p = 0, q = 0;
		
		for(int i = 0; i < 10000; i++) {
			//generate 2 random integers in range [window,length of whole data stream].
			int rand_int1 = rand.nextInt(max-window)+window; 
	        int rand_int2 = rand.nextInt(max-window)+window;
	        //'a' is the larger integer.
	        if(rand_int1 == rand_int2) continue;
	        int a, b;
	        if(rand_int1 > rand_int2) {
	        	a = rand_int1;
	        	b = rand_int2;
	        }
	        else {
	        	b = rand_int1;
	        	a = rand_int2;
	        }//if
	        
	        //consistency
	        if((f[a] > f[b] && g[a] > g[b]) || (f[a] < f[b] && g[a] < g[b])) r++;
	        if((f[a] > f[b] && g[a] < g[b]) || (f[a] < f[b] && g[a] > g[b])) s++;
	        
	        //discriminancy
	        if(f[a] != f[b] && g[a] == g[b]) p++;
	        if(g[a] != g[b] && f[a] == f[b]) q++;
		}
		System.out.println("r = " + r + ", s = " + s + ", p = " + p + ", q = " + q);
		
		if((r+s) == 0) cst = Double.POSITIVE_INFINITY;
		else cst = (double) r/(r+s);
		
		if(p == 0 && q == 0) dcn = -1;
		else if(q == 0) dcn = Double.POSITIVE_INFINITY;
		else dcn = (double)p/q;
		
		double[] result = new double[2];
		result[0] = cst;
		result[1] = dcn;
		
		return result;
	}

	public static void printPerformance(String filename, int numTimeStep) throws IOException {
		
		BufferedWriter writer = new BufferedWriter(new FileWriter(filename));

		/*Print out performance into the result file*/
		double[] mean_pmauc = new double[numTimeStep];
		double[] mean_wauc = new double[numTimeStep];
		double[] mean_ewauc = new double[numTimeStep];
		double[] mean_gm = new double[numTimeStep];
		writer.append("Mean at each time step \n");
		writer.append("PMAUC, WAUC, EWAUC, G-mean \n");
		for(int i = 0; i < numTimeStep; i++) {
			mean_pmauc[i] = Utils.mean(PMAUC_window[i]);
			mean_wauc[i] = Utils.mean(WAUC_window[i]);
			mean_ewauc[i] = Utils.mean(EWAUC_window[i]);
			mean_gm[i] = Utils.mean(gmean_window[i]);
			writer.append(mean_pmauc[i] + ", " + mean_wauc[i] + ", "+ mean_ewauc[i] + ", "+ mean_gm[i] + ", " + 
//					Utils.mean(classRecall_window[i][0]) + ", " + Utils.mean(classRecall_window[i][1]) + ", "+ 
//					Utils.mean(classRecall_window[i][2]) + ", "+ Utils.mean(classRecall_window[i][3]) + 
					"\n");
		}
		writer.append("Standard Deviation at each time step\n");
		writer.append("PMAUC, WAUC, EWAUC, G-mean\n");
		for(int i = 0; i < numTimeStep; i++) {
			writer.append(Math.sqrt(Utils.variance(PMAUC_window[i])) + ", " + Math.sqrt(Utils.variance(WAUC_window[i])) + ", "+ 
					Math.sqrt(Utils.variance(EWAUC_window[i])) + ", "+ Math.sqrt(Utils.variance(gmean_window[i])) + ", " + 
//					Math.sqrt(Utils.variance(classRecall_window[i][0])) + ", " + Math.sqrt(Utils.variance(classRecall_window[i][1])) + ", "+ 
//					Math.sqrt(Utils.variance(classRecall_window[i][2])) + ", "+ Math.sqrt(Utils.variance(classRecall_window[i][3])) + 
					"\n");
		}
		
		//print drift detector performance from artificial data
		//double[] detectorPerf = getDriftDectorPerformance();
		//writer.append("Drift detection performance \n");
		//writer.append("True Detection Rate, False Alarm Rate, Delay of Detection\n");
		//writer.append(detectorPerf[0] + ", " + detectorPerf[1] + ", " + detectorPerf[2]+"\n");
		

		//print number of detected drift for real data
		double driftnum = getDriftNumber4Realdata();
		writer.append("The averge number of detected drift\n");
		writer.append(driftnum+"\n");
				
		writer.close();
	}
	
	//Instead print the average performance per time step, this function prints the average performance on the last batch of real-world data per run.
	//It is for the wilcoxon sign rank test. 
	public static void printPerformancePerRun(String filename, int numRun) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(filename));

		/*Print out performance into the result file*/
		double[] lastbatch_pmauc = new double[numRun];
		double[] lastbatch_wauc = new double[numRun];
		double[] lastbatch_ewauc = new double[numRun];
		double[] lastbatch_gm = new double[numRun];
		writer.append("Mean performance on the last batch at each run \n");
		writer.append("PMAUC, WAUC, EWAUC, G-mean \n");
		for(int i = 0; i < numRun; i++) {
			lastbatch_pmauc[i] = row_average(PMAUC_window, 3600, 5288, i);
			lastbatch_wauc[i] = row_average(WAUC_window, 3600, 5288, i);
			lastbatch_ewauc[i] = row_average(EWAUC_window, 3600, 5288, i);
			lastbatch_gm[i] = row_average(gmean_window, 3600, 5288, i);
			writer.append(lastbatch_pmauc[i] + ", " + lastbatch_wauc[i] + ", "+ lastbatch_ewauc[i] + ", "+ lastbatch_gm[i] + ", " + 
//					Utils.mean(classRecall_window[i][0]) + ", " + Utils.mean(classRecall_window[i][1]) + ", "+ 
//					Utils.mean(classRecall_window[i][2]) + ", "+ Utils.mean(classRecall_window[i][3]) + 
					"\n");
		}
		
		writer.close();
	}

	public static int get_last_window_choice(int endOfWindow,int window_size,int usedMetric,int i) throws IOException {

		/*Gets performance from last window*/
		double result1 = 0.00;
		double result2 = 0.00;

		switch(usedMetric) {
			case 1:
				result1 = row_average(PMAUC_window1, endOfWindow - window_size, endOfWindow, i);
				result2 = row_average(PMAUC_window2, endOfWindow - window_size, endOfWindow, i);
				break;
			case 2:
				result1 = row_average(EWAUC_window1, endOfWindow - window_size, endOfWindow, i);
				result2 = row_average(EWAUC_window2, endOfWindow - window_size, endOfWindow, i);
				break;
			case 3:
				result1 = row_average(WAUC_window1, endOfWindow - window_size, endOfWindow, i);
				result2 = row_average(WAUC_window2, endOfWindow - window_size, endOfWindow, i);
				break;
			case 4:
				result1 = row_average(gmean_window1, endOfWindow - window_size, endOfWindow, i);
				result2 = row_average(gmean_window2, endOfWindow - window_size, endOfWindow, i);
				break;
			default:
				break;
		}

		if(result1>result2)
			return 1;
		return 2;
	}

	/**Apply PHT (Page-Hinckley Test) monitoring the drop of AUC-based metrics or G-mean */
	  public static int applyPHT(OzaBag model, PHT detector, AUCClassificationPerformanceEvaluator_mclass metric, Instance currentIns, double val, int time){ 
	    int isDrift = 0;
	    isDrift = detector.input_PAUC(val);
	    if(isDrift==1){
	      detector.storedInstances.add(currentIns);
	      detector.store = true;
	    }
	    else if (isDrift==2){
	      resetImbalanceStatus();
	      resetModel(model, detector.storedInstances);
	      resetWindowPerformance(metric);
	      detector.store = false;
	      detector.driftLocation[detector.numDrift] = time;
	      detector.numDrift++;
	      
	    }
	    else{
	      detector.storedInstances.delete();
	      detector.store = false;
	    }
	    return isDrift;
	  }
	  
	  public static void resetModel(OzaBag model, Instances storedInstances){
		  model.resetLearning();
		  for(int i = 0; i < storedInstances.numInstances(); i++){
			  model.trainOnInstance(storedInstances.instance(i));
		  }
	  }
	  
	  public static void resetImbalanceStatus() {
		  imbalance = false;
		  for(int i = 0; i < numClasses; i++){
			  numInstances[i] = 0;
		  }
	  }
	  
	  public static void resetWindowPerformance(AUCClassificationPerformanceEvaluator_mclass metric) {
		  metric.aucEstimator = metric.new Estimator(metric.widthOption);
	  }
	  
	  /*print True Detection Rate (TDR), the False Alarm Rate (FAR) and the Delay of Detection (DoD) for detectors over artificial data*/
	  public static double[] getDriftDectorPerformance() {
		  
		  double[] result = new double[3];
		  int numRun = driftLocations.length;
		  double tdr = 0, far = 0, dod = 0;
		  
		  int drift_time = 2500;
		  int far_i = 0;
		  
		  for(int i = 0; i < numRun; i++) {
			  for(int j = 0; j < driftLocations[i].length; j++){
				  if(driftLocations[i][j] != -1) {
					  if(driftLocations[i][j]>=drift_time && j == 0) {
						  tdr = tdr + 1;
						  dod+=driftLocations[i][j]-drift_time;
					  }
					  else if(driftLocations[i][j]>=drift_time && driftLocations[i][j-1]<drift_time) {
						  tdr = tdr + 1;
						  dod+=driftLocations[i][j]-drift_time;
					  }
					  else
						  far_i = far_i +1;
				  }
				  else break;
			  }
			  far += (double)far_i/(far_i+1);
			  far_i = 0;
		  }
		  dod = (double)dod/tdr;
		  tdr = (double)tdr/numRun;
		  far = (double)far/numRun;
		  result[0] = tdr;
		  result[1] = far;
		  result[2] = dod;
		  
		  return result;
	  }
	  
	  public static double getDriftNumber4Realdata() {
		  double driftnum = 0;
		  int numRun = driftLocations.length;
		  for(int i = 0; i < numRun; i++) {
			  for(int j = 0; j < driftLocations[i].length; j++){
				  if(driftLocations[i][j] != -1) 
					  driftnum = driftnum+1;
			  }
		  }
		  driftnum = (double)driftnum/numRun;
		  return driftnum;
	  }
	  
	  public static double average(int[] values) {  
		    int sum = 0;
		    double average;

		    for(int i=0; i < values.length; i++){
		        sum = sum + values[i];
		    }
		    average = (double)sum/values.length;
		    return average;    
		}
	  
	  public static double row_average(double[][] values, int startTime, int endTime, int run) {  
		    double sum = 0.0;
		    double average;

		    for(int i=startTime; i <= endTime; i++){
		        sum = sum + values[i][run];
		    }
		    average = sum/(endTime-startTime+1);
		    return average;    
		}
}
