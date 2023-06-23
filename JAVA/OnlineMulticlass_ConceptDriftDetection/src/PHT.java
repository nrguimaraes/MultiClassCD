package src;

import weka.core.Instances;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;
import moa.classifiers.core.driftdetection.*;

/**
 *
 * <p>
 * E. S. Page. Continuous inspection schemes. Biometrika, 41(1/2):100-115, June
 * 1954. URL http://www.jstor.org/stable/2333009.
 * </p>
 *
 * @author Paulo Goncalves (paulomgj at gmail dot com)
 *
 */
public class PHT extends AbstractOptionHandler implements DriftDetectionMethod {

    /**
   * 
   */
    private static final long serialVersionUID = 1L;
    public int minNumInstancesOption = 30; // Minimum number of instances.
    public double driftLevelOption = 1.5;//for input() function. It monitors the increase of error. Detection threshold Lambda: depends on the admissible false alarm rate.  Increasing lambda will entail fewer false alarms, but might miss or delay some changes.
    public double warningLevelOption = 0.5;//for input() function. Warning threshold.
    public double driftLevel = 50;//for input_PAUC() function. It monitors the drop of AUC and G-mean metrics.
    public double warningLevel = 30;//for input_PAUC() function.
    public double deltaOption = 0.005;//Magnitude threshold: corresponds to the magnitude of changes that are allowed.
    public double alphaOption = 0.9999;
    protected double Mint;
    protected double Maxt;
    protected double mt;
    protected long nt;
    protected double mean;
    protected double p;
    private int m_lastLevel;
    
    //added by me
    public int numDrift;//number of detected drift by ddm-oci
    public int[] driftLocation;// the time steps where the drift is detected
    public static int MAX_DRIFTNUM = 200;
    public Instances storedInstances; 
    public boolean store;// whether to store the current training example

    public PHT(Instances data) {
        resetLearning();
        numDrift = 0;
        store = false;
        storedInstances = new Instances(data,0);
        driftLocation = new int[MAX_DRIFTNUM];//assume the total number of detected drifts won't exceed MAX_DRIFTNUM. 
        for(int i = 0; i < MAX_DRIFTNUM; i ++)
        	driftLocation[i] = -1;
    }

    public void resetLearning() {
        this.mt = 0;
        this.Mint = Double.MAX_VALUE;
        this.Maxt = 0.0;
        this.nt = 1;
        this.mean = 0;
        this.p = 1;
        this.m_lastLevel = DDM_INCONTROL_LEVEL;
    }

    /**Test the increase of classification error. Need to monitor the minimum of 'mt'
     * x: prequential classification error*/
    public int input(double x) {
        if (this.m_lastLevel == 2) {
            resetLearning();
        }
        p += (x - p) / nt;
        nt++;
        this.mean = ((nt - 1) * this.mean + p) / nt;
        this.mt += p - this.mean - this.deltaOption;
        if (this.mt <= this.Mint) { // Mint is the minimum of mt
            this.Mint = this.mt;
        }
        double PHt = mt - Mint;

        if (nt < this.minNumInstancesOption) {
            return 0;
        }
        if (PHt > this.driftLevelOption) {
          this.m_lastLevel = 2;
        } else if (PHt > this.warningLevelOption*this.driftLevelOption) {
          this.m_lastLevel = 1;
        }
        
        return m_lastLevel;
    }
    
    /**Test the decrease of PAUC. Need to monitor the maximum of 'mt'
     * x: prequential PAUC*/
    public int input_PAUC(double x) {
    	if (this.m_lastLevel == 2) {
    		resetLearning();
    	} 	
    	this.mean = this.mean + (x - this.mean) / (double) nt;
    	nt++;
    	this.mt = this.alphaOption*this.mt + (x - this.mean + this.deltaOption);//Note: use "+delta" for monitoring the drop of PAUC, use "-delta" for monitoring the increase of the error
    	if (this.mt >= this.Maxt) { // Maxt is the maximum of mt
    		this.Maxt = this.mt;
    	}
    	double PHt = Maxt - mt;
    	//System.out.println(PHt);

    	if (nt < this.minNumInstancesOption) {
    		return 0;
    	}
    	if (PHt > this.driftLevel) {
    		this.m_lastLevel = 2;
    	} else if (PHt > this.warningLevel) {
    		this.m_lastLevel = 1;
    	}
    	else
    		this.m_lastLevel = 0;
    	return m_lastLevel;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
    }

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor,
            ObjectRepository repository) {
    }
    
    @Override
    public DriftDetectionMethod copy() {
        return (DriftDetectionMethod) super.copy();
    }

    @Override
    public int computeNextVal(boolean prediction) {
      // TODO Auto-generated method stub
      return 0;
    }
}
