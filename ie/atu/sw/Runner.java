package ie.atu.sw;

import javax.swing.SwingUtilities;

import ie.atu.sw.game.GameWindow;
import ie.atu.sw.network.GameLogger;

import static java.lang.System.*;

public class Runner {
	public static void main(String[] args) throws Exception {
		
		/*
		 * Always run a GUI in a separate thread from the main thread.
		 */
		SwingUtilities.invokeAndWait(() -> { //Sounds like the Command Pattern at work!
			try {
				new GameWindow();
			} catch (Exception e) {
				out.println("[ERROR] Yikes...problem starting up " + e.getMessage());
			}
		});
		
		// Register shutdown hook to ensure logs are flushed when application exits
		Runtime.getRuntime().addShutdownHook(new Thread(() -> {
			try {
				// Flush any remaining logs and close log files
				GameLogger.shutdown();
				out.println("Logging system shutdown complete.");
			} catch (Exception e) {
				out.println("[ERROR] Problem shutting down logging system: " + e.getMessage());
			}
		}));
	}
}