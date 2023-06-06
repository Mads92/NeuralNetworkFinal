public class Neuron {

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }
    public static double inputFunctionSum(double[] inputs) {
        double res = 0;
        for (int i = 0; i < inputs.length; i++) {
            res = res + inputs[i];
        }
        return res;
    }
}