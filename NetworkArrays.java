import org.junit.jupiter.params.shadow.com.univocity.parsers.annotations.Convert;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class NetworkArrays {

    double learningRate;
    double[][] inputNeurons;
    double[][] hiddenNeurons;
    double[][] outputNeurons;

    double[][] inputLayer;
    double[][] hiddenLayer;
    double[][] outputLayer;

    public double[] getActivationValues() {
        return activationValues;
    }

    public void setActivationValues(double[] activationValues) {
        this.activationValues = activationValues;
    }

    double[] inputActivations;
    double[] activationValues;

    double[] deltaIList;
    double[] deltaJList;

    public double[] getSumsIntoHiddenLayer() {
        return sumsIntoHiddenLayer;
    }

    public void setSumsIntoHiddenLayer(double[] sumsIntoHiddenLayer) {
        this.sumsIntoHiddenLayer = sumsIntoHiddenLayer;
    }

    double[] sumsIntoHiddenLayer;

    public double[] getSumsIntoOutputLayer() {
        return sumsIntoOutputLayer;
    }

    public void setSumsIntoOutputLayer(double[] sumsIntoOutputLayer) {
        this.sumsIntoOutputLayer = sumsIntoOutputLayer;
    }

    double[] sumsIntoOutputLayer;

    public double[] getDeltaIList() {
        return deltaIList;
    }

    public void setDeltaIList(double[] deltaI) {
        this.deltaIList = deltaI;
    }

    public double[] getDeltaJList() {
        return deltaJList;
    }

    public void setDeltaJList(double[] deltaJL) {
        this.deltaJList = deltaJL;
    }

    public void setInputNeurons(double[][] inputNeurons) {
        this.inputNeurons = inputNeurons;
    }

    public void setHiddenNeurons(double[][] hiddenNeurons) {
        this.hiddenNeurons = hiddenNeurons;
    }

    double upper;
    double lower;

    public NetworkArrays(double[][] inputNeurons , double[][] hiddenNeurons, double[][] outputNeurons, double learningRate,double upperThresh,double lowerThresh){
       this.inputNeurons = inputNeurons;
       this.hiddenNeurons = hiddenNeurons;
       this.outputNeurons = outputNeurons;
       this.learningRate = learningRate;
       this.upper = upperThresh;
       this.lower = lowerThresh;

       this.inputLayer = new double[inputNeurons.length][];
       for (int i = 0; i < inputNeurons.length; i++) {
           inputLayer[i] = inputNeurons[i];
       }

       this.hiddenLayer = new double[hiddenNeurons.length][];
       for (int i = 0; i < hiddenNeurons.length; i++) {
           hiddenLayer[i] = hiddenNeurons[i];
       }

       this.outputLayer = new double[outputNeurons.length][];
       for (int i = 0; i < outputNeurons.length; i++) {
           outputLayer[i] = outputNeurons[i];

           this.deltaIList = new double[outputLayer.length];
           this.deltaJList = new double[hiddenLayer.length];
       }
   }


    static double[][] readInputFromFile(String path) throws FileNotFoundException {
        //Method for reading inputs from a csv file
        //Parts of code, and general procedure, taken from
        //https://www.baeldung.com/java-csv-file-array
        File inputFile = new File(path);
        Scanner sc = new Scanner(inputFile);
        sc.useDelimiter(";");
        ArrayList<String[]> loaded = new ArrayList<>();
        while(sc.hasNext()){
                String[] temp;
                temp = (sc.nextLine().split(","));
                loaded.add(temp);
        }

        double[][] finalOut = new double[loaded.size()][loaded.get(0).length]; //Går den?
        System.out.println("Loaded size: " + loaded.size());
        for (int i = 0; i < loaded.size(); i++) {
            for (int j = 0; j < loaded.get(i).length; j++) {
                finalOut[i][j] = Double.valueOf(loaded.get(i)[j]);
            }
        }
        sc.close();
        return finalOut;
    }
   double[][] firstLayer(double[] in){
       double[] correctedIn = Arrays.copyOf(in,in.length-1);
       double[][] res = new double[inputLayer.length][];
       double[] acts = new double[inputLayer.length];
       if(correctedIn.length != inputLayer.length){
           System.out.println(correctedIn.length + " " + inputLayer.length);
       }
       for (int i = 0; i < inputLayer.length; i++) {
           double[] neuronOutputs = new double[inputLayer[i].length];
           for (int j = 0; j < inputLayer[i].length; j++) {
               double neuronres = 0;
               neuronres = (inputLayer[i][j]) * correctedIn[i];
               neuronOutputs[j] = neuronres;
           }
           res[i] = neuronOutputs;
       }
       return res;
   }
    public double[] calculateNeuronSumsHidden(double[][] in){
        // Calculate the sums of inputs into each hidden layer neuron
        double[] preweight = new double[hiddenLayer.length];
        //double[][] split = Layer.splitInputs(in);
        double calc=0;
        for (int i = 0; i < in.length; i++) {
            for (int j = 0; j <hiddenLayer[i].length; j++) { //USIKKER
                calc = Neuron.inputFunctionSum(in[i]);
            }
            preweight[i] = calc;
        }
        setSumsIntoHiddenLayer(preweight);
        return preweight;
    }

   double[][] calculateFromHiddenLayer(double[][] in){
       double[][] res = new double[hiddenLayer.length][];
       double[][] transformed = Layer.splitInputs(in);
       double[] sums = calculateNeuronSumsHidden(transformed);
       setSumsIntoHiddenLayer(sums);
       double[] activationValues = new double[hiddenLayer.length];

       for (int i = 0; i < hiddenLayer.length; i++) {
           double[] temp = new double[hiddenLayer[i].length];
           double activation = Neuron.sigmoid(sums[i]);
           activationValues[i] = activation;
           for (int j = 0; j < hiddenLayer[i].length; j++) {
               temp[j] = activation * hiddenLayer[i][j]; // vVED IKKE
           }
           res[i] = temp;
       }
       setActivationValues(activationValues);
       return res;
   }

    double[] computeOutputSigmoid(double[][] in){
       double[] res = new double[outputLayer.length];
       double[] sums = new double[outputLayer.length];
        double[][] transformed = Layer.splitInputs(in);

        for (int i = 0; i < outputLayer.length; i++) {
            sums[i] = Neuron.inputFunctionSum(transformed[i]);
        }
        for (int i = 0; i < outputLayer.length; i++) {
            res[i] = Neuron.sigmoid(sums[i]);
        }
        setSumsIntoOutputLayer(sums);
        return res;
    }

   double forwardProp( double[] in){
        //Compute the forward propagation from the inputs. Is currently set up specifically for a single output neuron.
        double[][] first = firstLayer(in);
        double[][] calFromHidden = calculateFromHiddenLayer(first);
        double[] res = computeOutputSigmoid(calFromHidden);
        double out = (res[0]);
        return out;
   }

    double outputError(double target, double observed){
        return target-observed;
    }

    double[] deltaI(double outputError, double[] weightedInput){
        // Currently set up for a single output neuron
        double[] res = new double[outputLayer.length];
        for (int i = 0; i < outputLayer.length; i++) {
            res[i] = outputError* (Neuron.sigmoidDerivative(weightedInput[i]));
        }
        setDeltaIList(res);
        return res;
    }

    void updateWeightsToOutput(double[] activations, double[] delt){
        double[][] updatedWeights = new double[hiddenLayer.length][];
        double[] sum = getSumsIntoOutputLayer();

        for (int i = 0; i < hiddenLayer.length; i++) {
            double[] temp = new double[hiddenLayer[i].length];
            for (int j = 0; j < hiddenLayer[i].length; j++) {
                this.hiddenLayer[i][j] = hiddenLayer[i][j] + (learningRate * activations[i] *delt[j]); //spicy, CHECK sum[i] eller [j]?
            }
        }
        setHiddenNeurons(updatedWeights);
    }

    double[] deltaJ(double[] deltaI, double out){
        //computes the error at each node in the hidden layer
        double[] deltas = new double[hiddenLayer.length];
        double[] hiddenSums =getSumsIntoHiddenLayer();
        for (int i = 0; i < hiddenLayer.length; i++) {
            double derived = Neuron.sigmoidDerivative(hiddenSums[i]);
            double sum = 0;
            double temp=0;
            for (int j = 0; j < deltaI.length; j++) {
                temp = hiddenLayer[i][j] * deltaI[j]; // usikker
                sum += temp;
            }
            deltas[i] = sum * derived;
        }
        setDeltaJList(deltas);
        return deltas;
    }

    double[] fullHiddenWeights(){
        //Calculates the sum of the weights from the hidden layer to the output layer.
       ArrayList<Double> temp = new ArrayList<>();
        double[] res;
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < hiddenLayer[i].length; j++) {
                temp.add(hiddenLayer[i][j]);
            }
        }
        res = new double[temp.size()];
        for (int i = 0; i < temp.size(); i++) {
            res[i] = temp.get(i);
        }
        temp.clear();
        return res;
    }

    double[] fullInputWeights(){
        //Returns the weights from the input layer to the hidden layer in one single array
        ArrayList<Double> temp = new ArrayList<>(); // OOF
        double[] res;
        for (int i = 0; i < inputLayer.length; i++) {
            for (int j = 0; j < inputLayer[i].length; j++) {
                temp.add(inputLayer[i][j]);
            }
        }
        res = new double[temp.size()];
        for (int i = 0; i < temp.size(); i++) {
            res[i] = temp.get(i);
        }
        temp.clear();
        return res;
    }

    void updateWeightsToHiddenLayer(double[] in,double[] deltaJ){
        //Updates the weights from the input layer to the hidden layer
        double[][] res = new double[inputLayer.length][];

        for (int i = 0; i < inputLayer.length; i++) {
            double[] temp = new double[inputLayer[i].length];
            for (int j = 0; j < inputLayer[i].length; j++) {
                temp[i] = inputLayer[i][j] + (learningRate *in[i] * deltaJ[j]);
            }
            res[i] = temp;
        }
        inputLayer = res;
    }

    void updateWeightsToHiddenLayerDot(double[] in,double[] deltaJ){
        //updates the weights from the hidden layer to the outputlayer. Set up for a single output neuron.
        double[] inputWeights = fullInputWeights();
        double[][] res = new double[inputLayer.length][inputLayer[0].length];
        for (int i = 0; i < inputLayer.length; i++) {
            double[] temp = new double[inputLayer[i].length];
            for (int j = 0; j < inputLayer[i].length; j++) {
                temp[j] = inputLayer[i][j] + (learningRate*in[i]*deltaJ[j]);
                res[i][j] = temp[j];
            }
        }
        inputLayer = res;
    }

    void backprop(double observed, double target,double[] in){
        //compute the error for thisexample
        double err = (outputError(target,observed));
        double[] deltaI = deltaI(err,getSumsIntoOutputLayer());

        //update the weights leading from the hidden layer to the outputLayer
        double[] hidden = getActivationValues();
        updateWeightsToOutput(hidden,deltaI);

        //compute the error at each hidden node
        double[] deltaJ =deltaJ(deltaI,observed);
        setDeltaJList(deltaJ);

        //update the weights leading into the hidden layer
         updateWeightsToHiddenLayerDot(in,deltaJ);
    }

    double correctedOut(double upper,double lower, double observed){
        //Rounds the observed output. Used to classify wether an output is within the thresholds.
        double res;
        if(observed < lower){
            res = 0;
        }
        else if(observed >= upper){
            res = 1;
        }
        else {
            res = observed;
        }
        return res;
    }

    void train(double[][] inputs, double[][] validation) throws IOException {
        //The training algortihm.
        //Writing to CSV files follows the code from the following link:
        //https://www.section.io/engineering-education/working-with-csv-files-in-java
        //objects for the error calculations.
        FileWriter fileWriterError = new FileWriter("C:\\Users\\madsk\\Documents\\Python\\errors.csv");
        BufferedWriter buffwrite = new BufferedWriter(fileWriterError);

        //writer for the trained weights.
        FileWriter fileWriterWeights = new FileWriter("C:\\Users\\madsk\\Documents\\trainedweights");
        BufferedWriter writeWeights = new BufferedWriter(fileWriterWeights);
        double originalLearningRate = learningRate;
        double correctedLearningRate = originalLearningRate * 0.1;
        boolean identical = false;
        int count = 0; // epochs
        double previousErrorsum = Integer.MAX_VALUE;
        double error;
        ArrayList<Double> errorList = new ArrayList<>();
        double[] errors;
        double[] outs;
        double[] results = new double[inputs.length];
        double validationError = Double.MAX_VALUE;
        int validationCheck = 0;
        int correctCounter = 0;
        while ((!identical)) {
            int incorrectCount = 0;
            errors = new double[inputs.length];
            outs = new double[inputs.length];
            double errorsum = 0;
            double errorSquare=0;
            ArrayList<Double> andErrors = new ArrayList<>();

            //Shuffle the inputs. Training appears more consistent when the inputs get shuffled
            double[][] shuffledIn;
            shuffledIn = Network.shuffleInputs(inputs);
            //extract targets
            double[] extractedTargets = new double[inputs.length]; // array for the target values
            for (int i = 0; i < shuffledIn.length; i++) {
                //adds the last value in an input array to the target array. This is assumming the last value in an input is the target.
                extractedTargets[i] = shuffledIn[i][shuffledIn[i].length-1];
            }

            for (int i = 0; i < shuffledIn.length; i++) {
                //the training algorithm starts
                double target = extractedTargets[i];
                double observedOutput = forwardProp(shuffledIn[i]);
                double sigmoidOutput = Neuron.sigmoid(observedOutput);
                double correctedOut = correctedOut(upper,lower,observedOutput);

                outs[i] = correctedOut;
                error = outputError(target, observedOutput);
                errorsum = errorsum + error; //Math.abs(error);
                errors[i] = (error);
                //errorList.add(error); // Used for plotting

                if (correctedOut == target) {
                    // if the corected output of the forward prop is equal to the target, correct counter increments.
                    // If this counter reaches the length of inputs (meaning it guesses correct for every input), the training terminates
                    correctCounter++;
                } else{
                    //reset the counter if the guess is incorrect.
                    correctCounter = 0;
                    incorrectCount++;
                }
                backprop(observedOutput,target,shuffledIn[i]);
            }
            double validationErrorSum = 0;
            for (int i = 0; i < validation.length; i++) {
                double valOut = forwardProp(validation[i]);
                double valErr= outputError(validation[i][validation[i].length-1],valOut);
                validationErrorSum = validationErrorSum+(valErr);
            }
            double currentvalidationError= validationErrorSum*validationErrorSum;
            if (currentvalidationError > validationError){
                validationCheck++;
            } else {validationCheck=0;}

            validationError = currentvalidationError;
            if (validationCheck >= inputs.length && count > 10000){ //10000 chosen completely arbitrarily
                System.out.println("Ended from validationcheck");
                identical = true;
            }
            errorSquare= errorsum*errorsum;

            //saves the error to a file. Outcommented for faster training.
/*            buffwrite.write(Double.toString(errorSquare));
            buffwrite.newLine();
            buffwrite.flush();*/
            if (errorSquare < 0.001 && validationErrorSum < 0.001){ //oprindeligt 0.0001
                System.out.println("Ended from errorsquare WAAA");
                identical = true;
            }

            //Below are some additional conditions which were deemed unnecessary. They are left for
            //archival purposes
/*            if(validationError + errorSquare < (validation.length + inputs.length)){
                System.out.println("Ended from validationerror < length");
                identical = true;
            }
            if(validationError < validation.length*0.5){
                System.out.println("Pure validationerror win lessgooo");
                identical=true;
            }*/
            if (count%10000 == 0){
                //Prints several variables every 10.000 epochs. Used to keep track of where the training is headed.
                double epochErrorsum = 0;
                double errorSquareCount=0;
                double currenterror = 0;
                for (int i = 0; i < errors.length; i++) {
                    currenterror = errors[i];
                    epochErrorsum = epochErrorsum+ currenterror;
                }
                errorSquareCount =   epochErrorsum*epochErrorsum; //Square the error

                if (previousErrorsum < errorSquare){ //errorsquarecount før
                    System.out.println("errorsquare has increased");
                    //identical = true; it no work
                }
                previousErrorsum= errorSquare; //errorsquarecount før

                if (count % 100000 == 0) {
                    //print periodically to see how many rows the network guessed incorrectly.
                    System.out.println("incorrect count: " + incorrectCount);
                }
                System.out.println("Epoch " + count +  " error: " + errorSquare + " " + "validation error: " + validationError);
            }
            if (Arrays.equals(extractedTargets,outs) || correctCounter >= inputs.length || (identical)) { //Før target blev omskrevet
                //if the outputs are equal to the list of targets at the end of an epoch, the training ends.
                if(correctCounter >= inputs.length){
                    System.out.println("Guessed from correctcounter");
                }
                //if the array of rounded outputs is equal to the array of targets, training ends.
                if(Arrays.equals(extractedTargets,outs)){
                    System.out.println("Extracted and out identical");
                }
                System.out.println("ended on iteration " + (count + 1));
                int correct = 0;
                int falsePositive = 0;
                int falseNegative =0;
                for (int i = 0; i < inputs.length; i++) {
                    //mainly here to confirm that the network will predict correctly for evey onput.
                    //System.out.println("for inputs " + inputs[i][0] + "," + inputs[i][1] + ",");
                    double res = forwardProp(inputs[i]);
                    double target = inputs[i][inputs[i].length-1];
                    double correctedOutput = correctedOut(upper,lower,res);

                    System.out.println(res + " , " + correctedOutput + " target: " + target + " error: " + outputError(inputs[i][inputs[i].length-1],res));
                    results[i] = res;
                }

                //write weights to a file.
                for (int i = 0; i < inputLayer.length; i++) {
                    for (int j = 0; j < inputLayer[i].length; j++) {
                        writeWeights.write(Double.toString(inputLayer[i][j]) + ",");
                    }
                    writeWeights.newLine();
                    writeWeights.flush();
                }
                for (int i = 0; i < hiddenLayer.length; i++) {
                    for (int j = 0; j < hiddenLayer[i].length; j++) {
                        writeWeights.write(Double.toString(hiddenLayer[i][j]) + ",");
                    }
                    writeWeights.newLine();
                    writeWeights.flush();
                }
                writeWeights.close();
                buffwrite.close();
                break;
            }
            count++;
        }
    }
    void printNet(){
        // Prints the weights for the network. Primarily used for testing purposes, and to ensure that the weights don't act weird.

        for (int i = 0; i < inputLayer.length; i++) {
            System.out.println("Weights for input " + i);
            for (int j = 0; j < inputLayer[i].length; j++) {
                System.out.println(inputLayer[i][j] + "");
            }
        }

        for (int i = 0; i < hiddenLayer.length;i++){
            System.out.println("Weights for hidden " + i);
            for (int j = 0; j < hiddenLayer[i].length; j++) {
                System.out.println(hiddenLayer[i][j] + " ");
            }
        }

        //Prints the deltas.
        System.out.println("Delta j");
        for (int i = 0; i < deltaJList.length; i++) {
            System.out.println(deltaJList[i]);
        }
        System.out.println("Delta I");
        for (int i = 0; i < deltaIList.length; i++) {
            System.out.println(deltaIList[i]);
        }
    }
}
