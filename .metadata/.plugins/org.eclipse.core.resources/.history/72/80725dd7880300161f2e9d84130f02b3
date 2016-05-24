package HansNet;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Array;
import java.util.*;
import java.io.*;

public class Neural_Net {
	public static final double e = 2.71828;
	private double[][] yHat;
	//public static double[] firstCol;
	//public static double[] secondCol;
	//public static double[] thirdCol;
	private double[][] input;
	private double[][] output;
	private double[][]  dJdW1;
	private double[][]  dJdW2;
	private  int input_Size;
	private  int hidden_Layer;
	private  int output_Size;
	private double[][] W1 ;
	private double[][] W2 ;
	private double[][] z3 ;
	private double[][] a2 ; 
	private double[][] z2 ;
	
	
	
	
	public static void main(String args[]) throws FileNotFoundException{
		//I should be able to make multiple neural nets
		//if you see anything bad just leave a comment on a method I should implement 
		//or a variable I should change
		double inputData[][] = new double[][]{{.1,.1},{.2,.2},{10,10},{20,20}};
		 double outputData[][] = new double[][]{{10},{10},{99},{105}};
		//readInput()
			Neural_Net  firstnet =  new Neural_Net(inputData,outputData); 
			firstnet.lernen();
		
	}
	
	//this structure might change when I refactor
public Neural_Net(double[][] X ,double[][] y){
	//this vs not this ?
	 this.input = X;
	 this.output = y;
	 this.input_Size = X[0].length;
	 this.hidden_Layer = X.length;
	 this.output_Size = y[0].length; 
	 
	 this.W1 = randmat(input_Size,hidden_Layer);
	this.W2 = randmat(hidden_Layer,output_Size);
		//these are Hyperparameters
		}
	
	//combines all the methods to do full gradient descent
	public void lernen(){
		scaling(input);
		scaling(output);
		//forms new weights
		int x =0;
		while(x<100){
		forward();
		costFunction();
		costFunctionPrime();
		//subtract djdw from 
		
		W1 = matrixSub(W1,multScal(dJdW1,1));
		W2 = matrixSub(W2,multScal(dJdW2,1));
		x++;
		}
		forward();
		costFunction();
		
		
	  }
	
public double[][] forward(){
		//propagates inputs through network
		//z_2 = xW
	z2 = matrixMult(input,W1);
	 //a2 = f(z_3)
	 a2 = matrixSigmoid(z2);
	 //z3 = a_2 * W_2
	 z3 = matrixMult(a2,W2);
	 //this is somehow wrong
	 //yHat is supposed to be 3 by 1 matrix 
	  this.yHat = matrixSigmoid(z3);
	 
		//this is our prediction	 
		return yHat;
		
	}

//this is the activation function
 public static double sigmoid(double z){
		
		return 1/(1+Math.pow(2.718, -z));
		
	}
	
	public static double[][] matrixSigmoid(double[][] unact){
		double[][] modified = new double[unact.length][unact.length];
		for(int i= 0;i<unact.length;i++){
			for(int j=0;j<unact[0].length;j++){
				modified[i][j] = sigmoid(unact[i][j]);
			}
		}
		return modified;
		
		
	}
	
	//generates random matrix that will be replaced when we start learning
	public static double[][]  randmat(int arg,int arg2){
		double[][] randn = new double[arg][arg2];
	    for(int i=0;i<arg;i++){
	    	for(int j=0;j<arg2;j++){
	    	randn[i][j] = Math.random();
	    }
	    }
	    return randn;
	}
	//this scales  the vectors so their can be proper comparison
	public static void scaling(double[][] prescal){
		double maxvalue = 0;
		
		for(int i=0;i<prescal.length;i++){
			for(int j = 0;j<prescal[0].length;j++){
			if(prescal[i][j]>maxvalue){
				maxvalue = prescal[i][j];
			}
		}
			
		}
		for(int i=0;i<prescal.length;i++){
			for(int j = 0;j<prescal[0].length;j++){
			prescal[i][j] = prescal[i][j]/maxvalue;
			}
		}
		}
		//this is the derivative of the sigmoid activation function used in backpropogation
	public static double sigmoidPrime(double pre){
	   return Math.pow(e, - pre)/(Math.pow(1 + Math.pow(e,-pre),2));
	}
	//applies sigmoid function to a whole matrix
	 public static double[][] matrixSigmoidPrime(double[][] pre){
		 double[][] returned = new double[pre.length][pre[0].length];
		 for(int i = 0;i<pre.length;i++){
			 for(int j = 0;j<pre[0].length;j++){
				 returned[i][j] = sigmoidPrime(pre[i][j]);
				 
			 }
		 }
		 return returned;
	 }
	//this uses the error to adjust the weight matrix using gradient descent
	public void costFunctionPrime(){
	double[][]  delta3 = handamard(multScal((matrixSub(output,yHat)),-1),matrixSigmoidPrime(z3));
 dJdW2 = matrixMult(transpose(a2), delta3);
	
	double[][] delta2 = handamard(matrixMult(delta3 , transpose(W2)),matrixSigmoidPrime(z2));
	dJdW1 =  matrixMult(transpose(input), delta2);
	}
	//this cost function measures the accuracy of our prediction 
	//this could be either static or non static depending on if you want a general cost function or
	public double  costFunction(){
		
		double cost = 0;
	for(int i=0;i<yHat.length;i++){
			cost = .5* (output[i][0] - yHat[i][0]);
			
		}
		
		System.out.println(cost);
		return cost;
	}
	
	//subtracts the second matrix from the first matrix
public static  double[][] matrixSub(double[][] first, double[][] second){
		double[][] solution = new double[first.length][first[0].length];
		for( int i = 0;i<first.length;i++){
			for( int j=0;j<first[0].length;j++){
				solution[i][j] = first[i][j] - second[i][j];
			}
		}
		return solution;
	}
	
	//transposes 
	public static double[][] transpose(double[][] untrans){
		double[][] temp = new double[untrans[0].length][untrans.length];
		for(int i = 0;i< untrans.length;i++){
			for(int j= 0;j<untrans[0].length;j++){
				temp[j][i] = untrans[i][j];
		}
		}
		untrans = temp;
		return untrans;
		//is this good form 
	}
	//multiplies a matrix by a scalar
	public  static double[][] multScal(double[][] inin, double scalar){
		for(int i = 0;i<inin.length;i++){
			for(int j = 0;j<inin[0].length;j++){
				inin[i][j] = inin[i][j]*scalar;
			}
		}
		return inin;
	}
	//multiplies a matrix by a matrix
	public static double[][] matrixMult(double[][] arg,double[][] arg2){
	double sum = 0;
	
		double[][] multiplied_matrix = new double[arg.length][arg2[0].length];
		//iterating over all rows
		for(int i=0;i<arg.length;i++){
			//iterating over all columns in second matrix
			for(int j= 0;j<arg2[0].length;j++){
				//iterating over columns in arg1 
				for(int k=0;k<arg[0].length;k++){
				
					multiplied_matrix[i][j] += arg[i][k]*arg2[k][j];
				}
				
			}
			
		}
		 
		
		return multiplied_matrix;
	}
	//performs handamard matrix multiplication this is component by component multiplication
	public static double[][] handamard(double[][] first, double[][] second){
		if(first.length != second.length){
			System.out.println("This will not work");
			//break;
		}
		double[][] returned = new double[first.length][first[0].length];
		for(int i = 0;i<first.length;i++){
			for(int j = 0;j<first[0].length;j++){
				returned[i][j] = first[i][j] * second[i][j];
			}
		}
		return returned;
	}
	
	//supposed to get input Data  doesnt actually do it ignore below
	
	////////////////////////////////////////////////////////////////////////??????????????????????????????????
	
	public static int measure() throws FileNotFoundException{
		 File newinputFile = new File("C:/Users/hansg17/workspace/newJAVart/src/newJAVart/Neural_NetInput.txt");
			Scanner sc = new Scanner(new FileReader(newinputFile));
		 
		
		int length = 0;
		while(sc.hasNextLine()){
		length++;
		sc.nextLine();
		}
		System.out.println(length);
		sc.close();
		return length;
		
	}
	
	
	//tries to read txt data for input into neural_net
	public static void readInput() throws FileNotFoundException{
		//change File to whatever file you want to your input file on your computer
		File newinputFile = new File("C:/Users/hansg17/workspace/newJAVart/src/newJAVart/Neural_NetInput.txt");
		Scanner scit = new Scanner(new FileReader(newinputFile));
		
		
		int rows = measure();
		
		//String traveler;
		int trav= 0;
		 firstCol = new double[rows];
		 secondCol = new double[rows];
	     thirdCol = new double[rows];
		
		//while(scit.hasNext()){
			System.out.println("yes");
			//traveler = scit.next();
			//trav =Double.parseDouble(traveler);
			System.out.println(scit.hasNextDouble());
			while(scit.hasNext()){
				//I know has nect double would work but it is not
				
				firstCol[trav] =  Double.parseDouble(scit.next());
				secondCol[trav] = Double.parseDouble(scit.next());
				thirdCol[trav] =  Double.parseDouble(scit.next());
				
				trav = trav+1;
				System.out.println(trav);
			}
		   
			//scit.nextLine();
			System.out.println("go");
			//System.out.println(scit.hasNextLine());
			//trav++;
			
			
			
		//}
		scit.close();
	}
	
	}
	


//}