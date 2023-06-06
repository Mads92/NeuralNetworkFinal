import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;

public class Main {
    static double generateRandom(double min, double max) {
        double random = ThreadLocalRandom.current().nextDouble(min, max);
        return random;
    }

    static double[] generateNeuron(int numberOfWeights,double min, double max){
        double[] res = new double[numberOfWeights];
        for (int i = 0; i < numberOfWeights; i++) {
            res[i] = generateRandom(min,max);
        }
        return res;
    }

    static double[][] generateLayer(int thisSize, int nextSize, double min, double max){
        double[][] res = new double[thisSize][nextSize];

        for (int i = 0; i < thisSize; i++) {
            res[i] = generateNeuron(nextSize,min,max);
        }
        return res;
    }
    public static void main(String[] args) throws IOException {
        String path1 = "C:\\Users\\madsk\\Documents\\trainingsets\\trainingGoodfix.csv";
        String path2 = "C:\\Users\\madsk\\Documents\\trainingsets\\validation.csv";

        double[][] training = NetworkArrays.readInputFromFile(path1);
        double[][] control = NetworkArrays.readInputFromFile(path2);

        int inputSize = training[0].length-1;
        int hiddensize = 13;
        int outsize = 1;

        double min = -0.3;
        double max= 0.3;

        double lower = 0.2;
        double upper = 0.9;
        double learningrate = 0.0001;
        double[][] inputLayer = generateLayer(inputSize,hiddensize,min,max);
        double[][] hiddenlayer = generateLayer(hiddensize,outsize,min,max);
        double[][] outputLayer = generateLayer(outsize,0,min,max);

        NetworkArrays net = new NetworkArrays(inputLayer,hiddenlayer,outputLayer,learningrate,upper,lower);
        int actualPositive = 0;
        int actualNegative = 0;
        for (int i = 0; i < training.length; i++) {
            if (training[i][training[i].length-1] == 1.0){
                actualPositive++;
            }
            if (training[i][training[i].length-1] == 0.0){
                actualNegative++;
            }
        }
        int validationPositive = 0;
        int validationNegative = 0;
        for (int i = 0; i < control.length; i++) {
            if (control[i][control[i].length-1] == 1.0){
                validationPositive++;
            }
            if (control[i][control[i].length-1] == 0.0){
                validationNegative++;
            }
        }
        System.out.println("Positive" + actualPositive);
        System.out.println("Negative" + actualNegative);
        System.out.println("Validation Positive" + validationPositive);
        System.out.println("Validation Negative" + validationNegative);
        net.train(training,control);

        upper = 0.4;
        lower = 0.4;

        double trainingTruePostive = 0;
        double trainingFalsePositive = 0;
        double trainingTrueNegative = 0;
        double trainingFalseNegative = 0;

        for (int i = 0; i < training.length; i++) {
            double out =net.forwardProp(training[i]);
            double correctedOutput = net.correctedOut(upper,lower,out);
            double target = training[i][training[i].length-1];
            if(correctedOutput == 1.0 && target == 1.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " true positive");
                trainingTruePostive++;
            }

            if(correctedOutput == 1.0 && target== 0.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " false positive");
                trainingFalsePositive++;
            }
            if (correctedOutput == 0.0 && target == 1.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " false negative");
                trainingFalseNegative++;
            }

            if (correctedOutput == 0.0 && target == 0.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " true negative");
                trainingTrueNegative++;
            }

            System.out.println("output: " + out + " " + "Target " + training[i][training[i].length-1]);
        }
        System.out.println("Training has " + actualPositive + " postive and " + actualNegative + " negative");
        System.out.println("training true positive " + trainingTruePostive );
        System.out.println("training false positive " + trainingFalsePositive);
        System.out.println("training true negative " +trainingTrueNegative);
        System.out.println("training false negative " +trainingFalseNegative);

        int actualPositiveControl = 0;
        int actualNegativeControl = 0;

        for (int i = 0; i < control.length; i++) {
            if (control[i][control[i].length-1] == 1.0){
                actualPositiveControl++;
            }
            if (control[i][control[i].length-1] == 0.0){
                actualNegativeControl++;
            }
        }
        int truePositive = 0;
        int trueNegative = 0;
        int falseNegative = 0;
        int falsePositive = 0;
        System.out.println("Control time");
        for (int i = 0; i < control.length; i++) {
            double out =net.forwardProp(control[i]);
            double correctedOutput = net.correctedOut(upper,lower,out);
            double target = control[i][control[i].length-1];

            if(correctedOutput == 1.0 && target == 1.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " true positive");
                truePositive++;
            }

            if(correctedOutput == 1.0 && target== 0.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " false positive");
                falsePositive++;
            }

            if (correctedOutput == 0.0 && target == 1.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " false negative");
                falseNegative++;
            }

            if (correctedOutput == 0.0 && target == 0.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " true negative");
                trueNegative++;
            }
        }
        double missedPositive = actualPositiveControl-truePositive;
        double missedNegative = actualNegativeControl - trueNegative;
        System.out.println("Control is of length " + control.length);
        System.out.println("true positive: " + truePositive + " out of " + actualPositiveControl + " " + truePositive/actualPositiveControl);
        System.out.println("True negative: " + trueNegative + "out of " + actualNegativeControl);
        System.out.println("False negative: " + falseNegative);
        System.out.println("False positive: " + falsePositive);
        System.out.println("missed positive " +missedPositive);
        System.out.println("missed negative " + missedNegative);
    }
}