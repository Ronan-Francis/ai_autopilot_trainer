package ie.atu.sw.autopilot;

import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class Trainer {
	private final BasicNetwork network;
	private final MLDataSet trainingSet;
	private final int epochs;

	public Trainer(BasicNetwork network, MLDataSet trainingSet, int epochs) {
		this.network = network;
		this.trainingSet = trainingSet;
		this.epochs = epochs;
	}

	public void train() {
	    MLTrain trainer = new ResilientPropagation(network, trainingSet);
	    for (int epoch = 0; epoch < epochs; epoch++) {
	        long startTime = System.currentTimeMillis(); // Start timing
	        trainer.iteration();
	        long epochTime = System.currentTimeMillis() - startTime; // Time taken for this epoch
	        
	        double error = trainer.getError();
	        System.out.println("Epoch " + (epoch + 1) 
	                + " | Error: " + error 
	                + " | Time: " + epochTime + " ms");

	        // Optionally, add more detailed logs every few epochs:
	        /*
	         	        if ((epoch + 1) % 10 == 0) {
	            System.out.println("Detailed info at epoch " + (epoch + 1) + ":");
	            for (int i = 0; i < network.getStructure().getLayers().size(); i++) {
	                System.out.println("Layer " + i + " weights: " + 
	                        java.util.Arrays.toString(network.getFlat().getWeights()));
	            }
	        }*/

	    }
	    trainer.finishTraining();
	    System.out.println("Training complete. Final Error: " + trainer.getError());
	}
}
