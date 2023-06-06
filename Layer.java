public class Layer {
    public static double[][] splitInputs(double[][] in){
        //Based on the process and code from:
        //https://javaconceptoftheday.com/how-to-perform-matrix-operations-in-java
        double[][] res = new double[in[0].length][in.length]; // Only works if all rows have equal length
        for (int i = 0; i < in.length; i++) {
            for (int j = 0; j < in[i].length; j++) {
                res[j][i] = in[i][j];
            }
        }
        return res;
    }
}