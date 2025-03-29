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

import ie.atu.sw.network.GameLogger;

public class GameWindow implements KeyListener, ActionListener {
    private GameView view;
    private JFrame frame;
    
    public GameWindow() throws Exception {
        // Prompt the user to choose autopilot mode
        int choice = JOptionPane.showConfirmDialog(
            null, 
            "Do you want to use autopilot?", 
            "Autopilot Mode", 
            JOptionPane.YES_NO_OPTION
        );
        boolean autopilot = (choice == JOptionPane.YES_OPTION);
        view = new GameView(autopilot); // Use the user's choice
        init();
        loadSprites();
        
        // Show available controls
        if(!autopilot) showControlsHelp();
    }
    
    /*
     * Build and display the GUI. 
     */
    public void init() throws Exception {
        frame = new JFrame("ATU - B.Sc. in Software Development");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.addKeyListener(this);
        frame.getContentPane().setLayout(new FlowLayout());
        frame.add(view);
        
        // Add menu bar
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
        
        JMenuItem hybridItem = new JMenuItem("Toggle Hybrid Mode");
        hybridItem.addActionListener(this);
        hybridItem.setActionCommand("hybrid");
        settingsMenu.add(hybridItem);
        
        JMenuItem debugItem = new JMenuItem("Toggle Debug Info");
        debugItem.addActionListener(this);
        debugItem.setActionCommand("debug");
        settingsMenu.add(debugItem);
        
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
        
        // Add menus to menu bar
        menuBar.add(gameMenu);
        menuBar.add(settingsMenu);
        menuBar.add(helpMenu);
        
        // Set menu bar
        frame.setJMenuBar(menuBar);
        
        frame.setSize(1000,1000);
        frame.setLocation(100,100);
        frame.pack();
        frame.setVisible(true);
    }
    
    /*
     * Load the sprite graphics from the image directory
     */
    public void loadSprites() throws Exception {
        var player = new Sprite("Player", 2,  "images/0.png", "images/1.png");
        view.setSprite(player);
        
        var explosion = new Sprite("Explosion", 7,  "images/2.png", 
                "images/3.png", "images/4.png", "images/5.png", 
                "images/6.png", "images/7.png", "images/8.png");
        view.setDyingSprite(explosion);
    }
    
    /**
     * Shows a dialog with available keyboard controls.
     */
    private void showControlsHelp() {
        String helpMessage = """
            KEYBOARD CONTROLS:
            ------------------
            UP Arrow Key:   Moves plane up
            DOWN Arrow Key: Moves plane down
            S:              Resets and restarts the game
            T:              Manually triggers neural network training
            H:              Toggles hybrid control (neural network + fuzzy)
            C:              Clears all collected training data
            D:              Toggles debug information display
            G:              Toggles confidence graph display
            V:              Toggles visualizations display
            F1:             Shows this help screen
            """;
        
        JOptionPane.showMessageDialog(null, helpMessage, "Game Controls", JOptionPane.INFORMATION_MESSAGE);
    }
    
    /**
     * Shows information about the game.
     */
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

    /**
     * Handles menu item actions.
     */
    @Override
    public void actionPerformed(ActionEvent e) {
        String command = e.getActionCommand();
        
        switch (command) {
            case "reset":
                view.reset();
                break;
            case "exit":
                System.exit(0);
                break;
            case "config":
                ConfigUI configUI = new ConfigUI(frame);
                configUI.setVisible(true);
                break;
            case "hybrid":
                view.toggleHybridControl();
                break;
            case "debug":
                view.toggleDebugInfo();
                break;
            case "controls":
                showControlsHelp();
                break;
            case "about":
                showAboutDialog();
                break;
        }
    }

    /*
     * KEYBOARD OPTIONS
     * ----------------
     * UP Arrow Key:   Moves plane up
     * DOWN Arrow Key: Moves plane down
     * S:              Resets and restarts the game
     * T:              Manually triggers training
     * H:              Toggles hybrid control (neural network + fuzzy)
     * C:              Clears all collected training data
     * D:              Toggles debug information display
     * G:              Toggles confidence graph display
     * F1:             Shows help screen
     */
    public void keyPressed(KeyEvent e) {
        if (e.getKeyCode() == KeyEvent.VK_S) {  //Press "S" to restart
            view.reset();                        //Reset the view and bail out            
            return;
        }
        
        if (e.getKeyCode() == KeyEvent.VK_T) {  //Press "T" to train
            if (view.getNeuralNetworkController() != null) {
                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                        "Manual training triggered...");
                new Thread(() -> {
                    view.getNeuralNetworkController().trainNetwork();
                }).start();
            }
            return;
        }
        
        if (e.getKeyCode() == KeyEvent.VK_H) {  //Press "H" to toggle hybrid control
            view.toggleHybridControl();
            return;
        }
        
        if (e.getKeyCode() == KeyEvent.VK_C) {  //Press "C" to clear training data
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                    "Clearing training data...");
            view.clearTrainingData();
            return;
        }
        if (e.getKeyCode() == KeyEvent.VK_V) {  // Press "V" to toggle visualizations
            view.toggleVisualizations();
            return;
        }
        
        if (e.getKeyCode() == KeyEvent.VK_D) {  //Press "D" to toggle debug info
            view.toggleDebugInfo();
            return;
        }
        
        if (e.getKeyCode() == KeyEvent.VK_G) {  //Press "G" to toggle confidence graph
            view.toggleConfidenceGraph();
            return;
        }
        
        if (e.getKeyCode() == KeyEvent.VK_F1) { //Press F1 for help
            showControlsHelp();
            return;
        }
        
        int step = switch(e.getKeyCode()) {
            case KeyEvent.VK_UP     -> -1;      //Press "UP Arrow"   
            case KeyEvent.VK_DOWN   ->  1;      //Press "DOWN Arrow"     
            default                 ->  0;      //No change. Fly straight
        };
        view.move(step);                        //Move one step
    }
    
    public void keyReleased(KeyEvent e) {} 		//Ignore
    public void keyTyped(KeyEvent e) {} 		//Ignore
}