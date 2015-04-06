import java.util.*;
import java.io.*;
/**
* BaselinePredictors class predicts users' movie rating
* given training dataset firstly
*/
public class BaselinePredictors{
	
	static final double EPS = 1e-5;
	static final double DECR_RATE = 0.33; // learning rate decrement
	static final double LAMBDA_U = 0.03; // regularization coefs
	static final double LAMBDA_M = 0.1; // regularization coefs
	static final int MAX_FEATURES = 2;
	
	private static int k; // max rating
	private static int U; // number of users
	private static int M; // number of movies
	private static int D; // training dataset size
	private static int T; // test dataset size
	
	private static double mu = 0; // raw average
	private static double learnRate = 0.1; // learning rate
	private static double rmse = 1; 
	private static double oldRmse = 0; 
	private static double threshold = 0.01; 
	
	private static double[] biasU; 
	private static double[] biasM; 
	private static double[][] userFeatures; 
	private static double[][] movieFeatures; 
	private static Like[] likesM;
	
	private static Scanner scan = new Scanner(System.in);
	
	public static void main(String[] args){
		readTrainInfo();
		initialize();
		train();
		predict();
	}
	
	private static void readTrainInfo(){
		k = scan.nextInt();
		U = scan.nextInt();
		M = scan.nextInt();
		D = scan.nextInt();
		T = scan.nextInt();
		likesM = new Like[D];
		for (int i = 0; i < D; i++){
			likesM[i] = new Like();
			likesM[i].userId = scan.nextInt();
			likesM[i].movieId = scan.nextInt();
			likesM[i].rating = scan.nextInt();
		}
	}
	
	private static void initialize(){
		biasM = new double[M];
		biasU = new double[U];
		Arrays.fill(biasM,0);
		Arrays.fill(biasU,0);
		
		userFeatures = new double[MAX_FEATURES][U];
		movieFeatures = new double[MAX_FEATURES][M];
		for (int i = 0; i < MAX_FEATURES; i++){
			Arrays.fill(userFeatures[i],0.1);
			Arrays.fill(movieFeatures[i],0.1);
		}
	}
	
	private static double dot(int u, int m){ // dot product
		double res = 0;
		for (int i = 0; i < MAX_FEATURES; i++){
			res += userFeatures[i][u] * movieFeatures[i][m];
		}
		return res;
	}
	
	private static void train(){
		while (Math.abs(oldRmse - rmse) > EPS){
			oldRmse = rmse;
			rmse = 0;
			for (int i = 0; i < D; i++){
				int curUser = likesM[i].userId;
				int curMovie = likesM[i].movieId;
				double curRating = mu + biasU[curUser] + biasM[curMovie] + dot(curUser,curMovie);

				double err = likesM[i].rating - curRating;
				rmse += err * err;
				
				mu += learnRate * err;
				biasU[curUser] += learnRate * (err - LAMBDA_U * biasU[curUser]);
				biasM[curMovie] += learnRate * (err - LAMBDA_M * biasM[curMovie]);
				
				for (int j = 0; j < MAX_FEATURES; j++){
					double usFeatCurr = userFeatures[j][curUser];
					double movFeatCurr = movieFeatures[j][curMovie];
					userFeatures[j][curUser] += learnRate * (err * movFeatCurr - LAMBDA_U * usFeatCurr);
					movieFeatures[j][curMovie] += learnRate * (err * usFeatCurr - LAMBDA_M * movFeatCurr);
				}
			}
			rmse = Math.sqrt(rmse/D);
					
//			if RMSE changes slowly, decrease learning rate
			if (rmse > oldRmse - threshold){
				learnRate = learnRate * DECR_RATE;
				threshold = threshold / 2;
			}
		}
	}
	
	private static void predict(){
		try{
			BufferedWriter bw = new BufferedWriter(new FileWriter("output.txt"));
			for (int i = 0; i < T; i++){
				int u = scan.nextInt();
				int m = scan.nextInt();
				double r = mu + biasU[u] + biasM[m] + dot(u, m);
				if (r > k) r = k;
				if (r < 1) r = 1;
				bw.write(r+"\n");
			}
			bw.close();
		}catch(IOException e){
			e.printStackTrace();
		}
	}
	
	private static class Like{
		public int userId;
		public int movieId;
		public int rating;
	}
}
