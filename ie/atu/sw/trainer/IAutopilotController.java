package ie.atu.sw.trainer;

import java.util.List;

public interface IAutopilotController {
	/**
	 * * Given a snapshot of the game state (for example, a flat array of cell
	 * values), * return an integer representing the movement: * -1 for moving up, 0
	 * for no movement, and +1 for moving down.
	 */
	int getMovement(double[] state);

	void trainNetwork(List<TrainingSample> trainingData, int epochs);
}
