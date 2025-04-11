package ie.atu.sw.game;

import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;

import ie.atu.sw.network.AutopilotSystem;
import ie.atu.sw.network.GameLogger;

public class GameWindow implements KeyListener, ActionListener {
	private GameView view;
	private JFrame frame;
	private boolean autopilot;

	public GameWindow() throws Exception {
		// Keep the prompt but with a more descriptive message
		int choice = JOptionPane.showConfirmDialog(null, 
				"Do you wish to enable the Neural Network Autopilot?", 
				"Autopilot Mode", JOptionPane.YES_NO_OPTION);
		autopilot = (choice == JOptionPane.YES_OPTION);
		view = new GameView(autopilot);
		init();
		loadSprites();

		// Always show controls help
		showControlsHelp();
	}

	public void init() throws Exception {
		frame = new JFrame("ATU - B.Sc. in Software Development");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.addKeyListener(this);
		frame.getContentPane().setLayout(new FlowLayout());
		frame.add(view);

		JMenuBar menuBar = new JMenuBar();

		// Game menu
		JMenu gameMenu = new JMenu("Game");
		JMenuItem resetItem = new JMenuItem("Reset");
		resetItem.addActionListener(this);
		resetItem.setActionCommand("reset");
		gameMenu.add(resetItem);

		JMenuItem exitItem = new JMenuItem("Exit");
		exitItem.addActionListener(this);
		exitItem.setActionCommand("exit");
		gameMenu.add(exitItem);

		// Settings menu
		JMenu settingsMenu = new JMenu("Settings");
		JMenuItem configItem = new JMenuItem("Configuration");
		configItem.addActionListener(this);
		configItem.setActionCommand("config");
		settingsMenu.add(configItem);

		JMenuItem modeItem = new JMenuItem("Cycle Autopilot Mode");
		modeItem.addActionListener(this);
		modeItem.setActionCommand("cycle-mode");
		settingsMenu.add(modeItem);

		JMenuItem debugItem = new JMenuItem("Toggle Debug Info");
		debugItem.addActionListener(this);
		debugItem.setActionCommand("debug");
		settingsMenu.add(debugItem);

		if (autopilot) {
			JMenuItem autopilotMonitorItem = new JMenuItem("Autopilot Monitor");
			autopilotMonitorItem.addActionListener(this);
			autopilotMonitorItem.setActionCommand("autopilot-monitor");
			settingsMenu.add(autopilotMonitorItem);

			JMenuItem teacherPanelItem = new JMenuItem("Teacher Panel");
			teacherPanelItem.addActionListener(this);
			teacherPanelItem.setActionCommand("teacher-panel");
			settingsMenu.add(teacherPanelItem);
			
			JMenuItem flightAnalyzerItem = new JMenuItem("Flight Analyzer");
			flightAnalyzerItem.addActionListener(this);
			flightAnalyzerItem.setActionCommand("flight-analyzer");
			settingsMenu.add(flightAnalyzerItem);
		}

		// Help menu
		JMenu helpMenu = new JMenu("Help");
		JMenuItem controlsItem = new JMenuItem("Controls");
		controlsItem.addActionListener(this);
		controlsItem.setActionCommand("controls");
		helpMenu.add(controlsItem);

		JMenuItem aboutItem = new JMenuItem("About");
		aboutItem.addActionListener(this);
		aboutItem.setActionCommand("about");
		helpMenu.add(aboutItem);

		menuBar.add(gameMenu);
		menuBar.add(settingsMenu);
		menuBar.add(helpMenu);

		frame.setJMenuBar(menuBar);
		frame.setSize(1000, 1000);
		frame.setLocation(100, 100);
		frame.pack();
		frame.setVisible(true);
	}

	public void loadSprites() throws Exception {
		var player = new Sprite("Player", 2, "images/0.png", "images/1.png");
		view.setSprite(player);

		var explosion = new Sprite("Explosion", 7, "images/2.png", "images/3.png", "images/4.png", "images/5.png",
				"images/6.png", "images/7.png", "images/8.png");
		view.setDyingSprite(explosion);
	}

	private void showControlsHelp() {
		String helpMessage = """
				KEYBOARD CONTROLS:
				------------------
				UP Arrow Key:   Moves plane up
				DOWN Arrow Key: Moves plane down
				S:              Resets and restarts the game
				T:              Manually triggers neural network training
				C:              Clears all collected training data
				D:              Toggles debug information display
				G:              Toggles confidence graph display
				V:              Toggles visualizations display
				M:              Cycles through autopilot modes
				F1:             Shows this help screen
				F2:             Toggles Autopilot Monitor window
				F3:             Toggles Teacher Panel window
				F4:             Toggles Flight Analyzer window
				""";

		JOptionPane.showMessageDialog(null, helpMessage, "Game Controls", JOptionPane.INFORMATION_MESSAGE);
	}

	private void showAboutDialog() {
		String aboutMessage = """
				ATU Plane Autopilot Simulator
				-----------------------------
				A neural network training simulation for the
				ATU B.Sc. in Software Development AI module.

				This application demonstrates how neural networks and
				fuzzy logic controllers can be used to create an
				autonomous flying system.
				""";

		JOptionPane.showMessageDialog(null, aboutMessage, "About", JOptionPane.INFORMATION_MESSAGE);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		String command = e.getActionCommand();
		switch (command) {
		case "reset":
			view.reset();
			break;
		case "exit":
			// Perform clean shutdown
			if (autopilot) {
				AutopilotManager.getInstance().shutdown();
			}
			GameLogger.shutdown();
			System.exit(0);
			break;
		case "config":
			// Since ConfigUI isn't provided, we'll show a message instead
			JOptionPane.showMessageDialog(frame, "Configuration panel would appear here", "Configuration",
					JOptionPane.INFORMATION_MESSAGE);
			break;
		case "controls":
			showControlsHelp();
			break;
		case "about":
			showAboutDialog();
			break;
		case "debug":
			view.toggleDebugInfo();
			break;
		case "autopilot-monitor":
			AutopilotManager.getInstance().toggleMonitorVisibility();
			break;
		case "teacher-panel":
			AutopilotManager.getInstance().toggleTeacherPanel();
			break;
		case "flight-analyzer":
			AutopilotManager.getInstance().toggleFlightAnalyzer();
			break;
		case "cycle-mode":
			if (autopilot) {
				AutopilotSystem.Mode newMode = AutopilotManager.getInstance().cycleMode();
				JOptionPane.showMessageDialog(frame, "Autopilot mode set to: " + newMode,
						"Autopilot Mode", JOptionPane.INFORMATION_MESSAGE);
			}
			break;
		case "retrain":
			if (autopilot) {
				boolean initiated = AutopilotManager.getInstance().retrainModel();
				if (initiated) {
					JOptionPane.showMessageDialog(frame, "Retraining initiated. This will run in the background.",
							"Model Retraining", JOptionPane.INFORMATION_MESSAGE);
				} else {
					JOptionPane.showMessageDialog(frame, "Unable to initiate retraining. Check logs for details.",
							"Model Retraining", JOptionPane.WARNING_MESSAGE);
				}
			} else {
				JOptionPane.showMessageDialog(frame, "Retraining is only available in autopilot mode.",
						"Model Retraining", JOptionPane.WARNING_MESSAGE);
			}
			break;
		}
	}

	public void keyPressed(KeyEvent e) {
		if (e.getKeyCode() == KeyEvent.VK_S) {
			view.reset();
			return;
		}
		if (e.getKeyCode() == KeyEvent.VK_T) {
			if (autopilot) {
				GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
						"Manual training triggered...");
				AutopilotManager.getInstance().retrainModel();
			}
			return;
		}
		if (e.getKeyCode() == KeyEvent.VK_C) {
			GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
					"Clearing training data...");
			AutopilotManager.getInstance().clearTrainingData();
			return;
		}
		if (e.getKeyCode() == KeyEvent.VK_D) {
			view.toggleDebugInfo();
			return;
		}
		if (e.getKeyCode() == KeyEvent.VK_G) {
			view.toggleConfidenceGraph();
			return;
		}
		if (e.getKeyCode() == KeyEvent.VK_V) {
			view.toggleVisualizations();
			return;
		}
		if (e.getKeyCode() == KeyEvent.VK_M) {
			if (autopilot) {
				AutopilotSystem.Mode newMode = AutopilotManager.getInstance().cycleMode();
				// Display the new mode in the status label
				String modeMessage = "Autopilot mode changed to: " + newMode;
				GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK, modeMessage);
				JOptionPane.showMessageDialog(frame, modeMessage, "Autopilot Mode Changed", JOptionPane.INFORMATION_MESSAGE);
			}
			return;
		}
		if (e.getKeyCode() == KeyEvent.VK_F1) {
			showControlsHelp();
			return;
		}
		// Additional keyboard shortcuts for new UI panels
		if (e.getKeyCode() == KeyEvent.VK_F2) {
			AutopilotManager.getInstance().toggleMonitorVisibility();
			return;
		}
		if (e.getKeyCode() == KeyEvent.VK_F3) {
			AutopilotManager.getInstance().toggleTeacherPanel();
			return;
		}
		if (e.getKeyCode() == KeyEvent.VK_F4) {
			AutopilotManager.getInstance().toggleFlightAnalyzer();
			return;
		}
		int step = switch (e.getKeyCode()) {
		case KeyEvent.VK_UP -> -1;
		case KeyEvent.VK_DOWN -> 1;
		default -> 0;
		};
		view.move(step);
	}

	public void keyReleased(KeyEvent e) {
	}

	public void keyTyped(KeyEvent e) {
	}
}