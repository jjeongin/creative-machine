package ml;

import processing.core.*;
// import org.tensorflow.*;

/**
 * This is a template class and can be used to start a new processing Library.
 * Make sure you rename this class as well as the name of the example package 'template' 
 * to your own Library naming convention.
 * 
 * (the tag example followed by the name of an example included in folder 'examples' will
 * automatically include the example in the javadoc.)
 *
 * @example HelloML4Processing
 */

public class MLTest {
	
	// myParent is a reference to the parent sketch
	PApplet myParent;

	int myVariable = 0;
	
	public final static String VERSION = "##library.prettyVersion##";
	

	/**
	 * a Constructor, usually called in the setup() method in your sketch to
	 * initialize and start the Library.
	 * 
	 * @example Hello
	 * @param theParent the parent PApplet
	 */
	public MLTest(PApplet theParent) {
		myParent = theParent;
		welcome();
	}
	
	
	private void welcome() {
		System.out.println("ML4Processing 0.0.1 by Jeongin Lee");
	}
	
	
	public String getTensorFlowVersion() {
		// return TensorFlow.version();
		return "";
	}
	/**
	 * return the version of the Library.
	 * 
	 * @return String
	 */
	public static String version() {
		return VERSION;
	}

	/**
	 *
	 * @param theA the width of test
	 * @param theB the height of test
	 */
	public void setVariable(int theA, int theB) {
		myVariable = theA + theB;
	}

	/**
	 *
	 * @return int
	 */
	public int getVariable() {
		return myVariable;
	}
}

