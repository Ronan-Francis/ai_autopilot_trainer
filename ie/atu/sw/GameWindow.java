package ie.atu.sw;

import java.awt.FlowLayout;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JFrame;

public class GameWindow implements KeyListener{
	private GameView view;
	
	public GameWindow() throws Exception {
		view = new GameView(true); //Use true to get the plane to fly in autopilot mode...
		init();
		loadSprites();
	}

	
	/*
	 * Build and display the GUI. 
	 */
	public void init() throws Exception {
	 	var f = new JFrame("ATU - B.Sc. in Software Development");
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.addKeyListener(this);
        f.getContentPane().setLayout(new FlowLayout());
        f.add(view);
        f.setSize(1000,1000);
        f.setLocation(100,100);
        f.pack();
        f.setVisible(true);
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
	
	
	/*
	 * KEYBOARD OPTIONS
	 * ----------------
	 * UP Arrow Key: 	Moves plane up
	 * DOWN Arrow Key: 	Moves plane down
	 * S:				Resets and restarts the game
	 * 
	 * Maybe consider adding options for "start sampling" and "end
	 * sampling"
	 * 
	 */
	public void keyPressed(KeyEvent e) {
		if (e.getKeyCode() == KeyEvent.VK_S) {	//Press "S" to restart
			view.reset(); 						//Reset the view and bail out			
			return;
		}
		
		int step = switch(e.getKeyCode()) {
			case KeyEvent.VK_UP 	-> -1;		//Press "UP Arrow" 	
			case KeyEvent.VK_DOWN 	->  1;		//Press "DOWN Arrow" 	
			default 				->  0;		//No change. Fly straight
		};
		view.move(step);						//Move one step
	}

    public void keyReleased(KeyEvent e) {} 		//Ignore
	public void keyTyped(KeyEvent e) {} 		//Ignore
}