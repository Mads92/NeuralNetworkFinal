import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.ParseException;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.*;

class NetworkArraysTest {
    double generateRandom(double min, double max) {
        double random = ThreadLocalRandom.current().nextDouble(min, max);
        double random2 = ThreadLocalRandom.current().nextDouble();
        if (random == 0){
            System.out.println("WHY");
        }
        return random;
    }

    double[] generateNeuron(int numberOfWeights,double min, double max){
        double[] res = new double[numberOfWeights];
        for (int i = 0; i < numberOfWeights; i++) {
            res[i] = generateRandom(min,max);
        }
        return res;
    }

    double[][] generateLayer(int thisSize, int nextSize, double min, double max){
        double[][] res = new double[thisSize][nextSize];

        for (int i = 0; i < thisSize; i++) {
            res[i] = generateNeuron(nextSize,min,max);
        }
        return res;
    }

    @Test
    void testLayerCreator(){
        int thisSize = 3;
        int nextSize = 1;
        double min = 0;
        double max = 1;

        double[][] test =generateLayer(thisSize,nextSize,min,max);

        assertEquals(thisSize,test.length);
        assertEquals(nextSize,test[0].length);
        assertEquals(nextSize,test[1].length);
    }

    NetworkArrays createNetRandom(double min, double max, double upper, double lower, double learn) {

        double[] n1 = {generateRandom(min, max), generateRandom(min, max)};
        double[] n2 = {generateRandom(min, max), generateRandom(min, max)};
        double[] n3 = {generateRandom(min, max)};
        double[] n4 = {generateRandom(min, max)};
        double[] n5 = {generateRandom(min, max)};

        double[][] inLayer = {n1, n2};
        double[][] hiddenLayer = {n3, n4};
        double[][] outLayer = {n5};


        NetworkArrays net = new NetworkArrays(inLayer, hiddenLayer, outLayer, learn, upper, lower);
/*        System.out.println("After creation:");
        for (int i = 0; i < net.inputLayer.length; i++) {
            System.out.println("Weights for input " + i);
            for (int j = 0; j < net.inputLayer[i].length; j++) {
                System.out.println(net.inputLayer[i][j] + "");
            }
        }*/

/*        for (int i = 0; i < net.hiddenLayer.length;i++){
            System.out.println("Weights for hidden " + i);
            for (int j = 0; j < net.hiddenLayer[i].length; j++) {
                System.out.println(net.hiddenLayer[i][j] + " ");
            }
        }*/
        return net;
    }
    NetworkArrays createNetRandom3(double min, double max, double upper, double lower, double learn) {

        double[] n1 = {generateRandom(min, max), generateRandom(min, max),generateRandom(min, max)};
        double[] n2 = {generateRandom(min, max), generateRandom(min, max),generateRandom(min, max)};
        double[] n3 = {generateRandom(min, max)};
        double[] n4 = {generateRandom(min, max)};
        double[] n5 = {generateRandom(min, max)};
        double[] n6 = {generateRandom(min, max)};

        double[][] inLayer = {n1, n2};
        double[][] hiddenLayer = {n3, n4,n5};
        double[][] outLayer = {n6};


        NetworkArrays net = new NetworkArrays(inLayer, hiddenLayer, outLayer, learn, upper, lower);
/*        System.out.println("After creation:");
        for (int i = 0; i < net.inputLayer.length; i++) {
            System.out.println("Weights for input " + i);
            for (int j = 0; j < net.inputLayer[i].length; j++) {
                System.out.println(net.inputLayer[i][j] + "");
            }
        }

        for (int i = 0; i < net.hiddenLayer.length;i++){
            System.out.println("Weights for hidden " + i);
            for (int j = 0; j < net.hiddenLayer[i].length; j++) {
                System.out.println(net.hiddenLayer[i][j] + " ");
            }
        }*/
        return net;
    }


    @Test
    void loadFromFile() throws IOException, ParseException {
        String path = "C:\\Users\\madsk\\Documents\\inputs.csv";
        double[][] test = NetworkArrays.readInputFromFile(path);
        double[] manual1 = {0,0,0};
        double[] manual2 = {0,1,1};
        double[] manual3 = {1,0,0};
        double[] manual4 = {1,1,1};
        assertArrayEquals(manual1,test[0]);

    }
    @Test
    void forwardProp() {
        double sig = Neuron.sigmoid(1);
        double sigSum = sig + sig;

        double outSig = Neuron.sigmoid(sigSum);
        double[] in1 = {1, 1};
        double[] in2 = {1, 1};

        double[] hi1 = {1};
        double[] hi2 = {1};

        double[] out = {0};

        double[][] inputNeurons = {in1, in2};
        double[][] hiddenNeurons = {hi1, hi2};

        double[][] outputNeurons = {out};
        double learningRate = 1;


        double sig2 = Neuron.sigmoid(2);
        double ex3 = Neuron.sigmoid((sig2 + sig2));
        double[] testin = {1, 1};
        NetworkArrays n = new NetworkArrays(inputNeurons, hiddenNeurons, outputNeurons, learningRate, 0.9, 0.1);
        double observed = n.forwardProp(testin);
        assertEquals(ex3, observed);

        assertEquals(2, n.sumsIntoHiddenLayer[0]);
        assertEquals(2, n.sumsIntoHiddenLayer[1]);
        assertEquals(sig2, n.activationValues[0]);
        assertEquals(sig2, n.activationValues[1]);

        assertEquals(sig2 + sig2, n.sumsIntoOutputLayer[0]);
    }

    @Test
    void deltaJTest() {
        double[] in1 = {1, 1};
        double[] in2 = {1, 1};

        double[] hi1 = {1};
        double[] hi2 = {1};

        double[] out = {0};

        double[][] inputNeurons = {in1, in2};
        double[][] hiddenNeurons = {hi1, hi2};

        double[][] outputNeurons = {out};
        double learningRate = 1;


        double sig2 = Neuron.sigmoid(2);
        double sig2Derivative = Neuron.sigmoidDerivative(2);
        double ex3 = Neuron.sigmoid((sig2 + sig2));
        double[] testin = {1, 1, 1};
        NetworkArrays n = new NetworkArrays(inputNeurons, hiddenNeurons, outputNeurons, learningRate, 0.9, 0.1);
        double observed = n.forwardProp(testin);
        for (int i = 0; i < n.getDeltaJList().length; i++) {
            System.out.println(n.getDeltaJList()[i]);
        }
        // double calcedDelta = sig2Derivative *
        n.backprop(observed, 1, testin);
        for (int i = 0; i < n.getDeltaJList().length; i++) {
            System.out.println(n.getDeltaJList()[i]);
        }
        double err = n.outputError(1, observed);
        double calculatedDelta = err * Neuron.sigmoidDerivative(n.sumsIntoOutputLayer[0]);
        //double[] delt = n.deltaJ(n.getDeltaIList());
        //assertEquals(calculatedDelta,delt[0]);
        //assertEquals(calculatedDelta,n.getDeltaJList()[0]);
    }

    @Test
    void testUpdateToOutput() {
        double[] in1 = {1, 1};
        double[] in2 = {1, 1};

        double[] hi1 = {1};
        double[] hi2 = {1};

        double[] out = {0};

        double[][] inputNeurons = {in1, in2};
        double[][] hiddenNeurons = {hi1, hi2};

        double[][] outputNeurons = {out};
        double learningRate = 1;


        double sig2 = Neuron.sigmoid(2);
        double sig2Derivative = Neuron.sigmoidDerivative(2);
        double ex3 = Neuron.sigmoid((sig2 + sig2));
        double[] activations = {sig2, sig2};
        double[] testin = {1, 1, 1};
        NetworkArrays n = new NetworkArrays(inputNeurons, hiddenNeurons, outputNeurons, learningRate, 0.9, 0.1);
        double observed = n.forwardProp(testin);
        n.deltaIList[0] = 0.5;
        n.updateWeightsToOutput(activations, n.getDeltaIList());
        double w1 = 1 + sig2 * 0.5;
        assertEquals(w1, n.hiddenLayer[0][0]);
        assertEquals(w1, n.hiddenLayer[1][0]);
    }

    @Test
    void updateToHidden() {
        double[] in1 = {1, 1};
        double[] in2 = {1, 1};

        double[] hi1 = {1};
        double[] hi2 = {1};

        double[] out = {0};

        double[][] inputNeurons = {in1, in2};
        double[][] hiddenNeurons = {hi1, hi2};

        double[][] outputNeurons = {out};
        double learningRate = 1;


        double sig2 = Neuron.sigmoid(2);

        double[] testin = {1, 1, 1};
        NetworkArrays n = new NetworkArrays(inputNeurons, hiddenNeurons, outputNeurons, learningRate, 0.9, 0.1);
        double observed = n.forwardProp(testin);
        double[] temp = {0.5, 0.5};
        n.setDeltaJList(temp);
        ; // deltaJlist og deltaIlist opfører sig forskelligt=

        n.updateWeightsToHiddenLayer(testin, n.deltaJList);
        double w1 = 1.5;
        assertEquals(w1, n.inputLayer[0][0]);
        assertEquals(w1, n.inputLayer[1][0]);

    }

    @Test
    void backPropManualCheck() {
        double[] in1 = {1, 1};
        double[] in2 = {1, 1};

        double[] hi1 = {1};
        double[] hi2 = {1};

        double[] out = {0};

        double[][] inputNeurons = {in1, in2};
        double[][] hiddenNeurons = {hi1, hi2};

        double[][] outputNeurons = {out};
        double learningRate = 1;


        double sig2 = Neuron.sigmoid(2);
        double sig2Derivative = Neuron.sigmoidDerivative(2);
        double ex3 = Neuron.sigmoid((sig2 + sig2));
        double[] testin = {1, 1, 1};
        NetworkArrays n = new NetworkArrays(inputNeurons, hiddenNeurons, outputNeurons, learningRate, 0.9, 0.1);
        double observed = n.forwardProp(testin);

        n.backprop(observed, 1, testin);

        double err = 1 - observed;

        assertEquals(err, n.outputError(1, observed));
        System.out.println("Output error passed");
        System.out.println("test sums into outputlayer");
        assertEquals(1, n.sumsIntoOutputLayer.length);

        System.out.println("Test Delta I");
        assertEquals(1, n.deltaIList.length);
        System.out.println("Delta I length confirmed");

        System.out.println("Sums into outputlayer test");
        assertEquals(1, n.sumsIntoOutputLayer.length);
        assertEquals(sig2 + sig2, n.sumsIntoOutputLayer[0]);

        double computedDelta = err * Neuron.sigmoidDerivative(n.sumsIntoOutputLayer[0]);

        System.out.println("Testing delta I length");
        assertEquals(1, n.deltaIList.length);

        System.out.println("Testing Delta I");
        double[] delt = n.deltaI(err, n.sumsIntoOutputLayer); // hvis den her ikke er her virker det ikke???
        assertEquals(delt[0], n.deltaIList[0]);
        assertEquals(computedDelta, n.deltaIList[0]);

        System.out.println("Test delta j");
        assertEquals(n.deltaJList, n.getDeltaJList());
        assertEquals(2, n.deltaJList.length);

        double firstDeltaJComputed = sig2Derivative * (n.hiddenLayer[0][0] + n.deltaIList[0]); // Delt og
        double firstDeltaJComputed2 = sig2Derivative * (n.hiddenLayer[0][0] + delt[0]); // Delt og
        assertEquals(firstDeltaJComputed, firstDeltaJComputed2);
        double[] temp = n.getDeltaJList();
        double[] deltaJ = n.deltaJ(delt,observed);
        assertEquals(n.deltaJList[0], deltaJ[0]);

        assertEquals(firstDeltaJComputed, firstDeltaJComputed2);
        assertEquals(n.getDeltaJList()[0], deltaJ[0]);
        assertEquals(firstDeltaJComputed, n.deltaJList[0]);
    }

    @Test
    void deltaITest() {
        double[] in1 = {1, 1};
        double[] in2 = {1, 1};

        double[] hi1 = {1};
        double[] hi2 = {1};

        double[] out = {0};

        double[][] inputNeurons = {in1, in2};
        double[][] hiddenNeurons = {hi1, hi2};

        double[][] outputNeurons = {out};
        double learningRate = 1;

        double[] insum = {Neuron.sigmoid(2) + Neuron.sigmoid(2)};
        double sig2 = Neuron.sigmoid(2);
        double ex3 = Neuron.sigmoid((sig2 + sig2));
        double[] testin = {1, 1, 1};
        NetworkArrays n = new NetworkArrays(inputNeurons, hiddenNeurons, outputNeurons, learningRate, 0.9, 0.1);


        System.out.println(n.deltaIList[0]);
        System.out.println(n.getDeltaIList()[0]);
        double observed = n.forwardProp(testin);
        assertEquals(ex3, observed);
        System.out.println(n.deltaIList[0]);
        System.out.println(n.getDeltaIList()[0]);

        n.backprop(observed, 1, testin);
        System.out.println(n.deltaIList[0]);
        System.out.println(n.getDeltaIList()[0]);
        double calcedDelta = n.outputError(1, observed) * ex3;

        //assertEquals(1,delta.length);
        System.out.println("delta I length confirmed");

        double[] delta = n.deltaI(n.outputError(1, observed), n.sumsIntoOutputLayer); // virker ikke når den her ikke bliver kaldt seperat.
        assertEquals(calcedDelta, delta[0]);
        assertEquals(calcedDelta, n.getDeltaIList()[0]);
    }

    @Test
    void testHiddenSums() {
        double[][] in = {{1, 2}, {3, 4}};

        double[][] h = {{1, 1}, {2, 2}};

        double[][] out = {{1}};
        NetworkArrays n = new NetworkArrays(in, h, out, 0.1, 1, 0);

        n.calculateNeuronSumsHidden(Layer.splitInputs(in));
        assertEquals(4, n.sumsIntoHiddenLayer[0]);
        assertEquals(6, n.sumsIntoHiddenLayer[1]);
    }

    @Test
    void back() {
        double[] testin = {1, 1, 1};
        double target = 1;
        double upper = 0.9;
        double lower = 0.2;

        double min = 0;
        double max = 1.0;
        NetworkArrays n = createNetRandom(min, max, upper, lower, 0.010);

        System.out.println("ecpected " + testin[testin.length - 1]);
        System.out.println(n.hiddenLayer[1][0]);
        double out = n.forwardProp(testin);
        System.out.println(out);
        n.backprop(target, 0, testin);
        System.out.println(n.hiddenLayer[1][0]);
    }

   /* @Test
    void trainTestAndIndividually() throws IOException {
        double upper = 0.9;
        double lower = 0.1;

        double min = 0;
        double max = 1;
        double[][] inputXOR = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 0}};

        double[][] inputAND = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}};
        double[][] inputANDSmall11 = {{1, 1, 1}}; // confirmed
        double[][] inputANDSmall10 = {{1, 0, 0}}; // confirmed
        double[][] inputANDSmall01 = {{0, 1, 0}}; // confirmed
        double[][] inputANDSmall00 = {{0, 0, 0}}; // confirmed

        double[][] inputOR = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
        NetworkArrays net = createNetRandom(min, max, upper, lower, 0.001);


        net.train(inputANDSmall00);
        //net.train(inputANDSmall10, upper, lower);
        //net.train(inputANDSmall01, upper, lower);
        //net.train(inputANDSmall00, upper, lower);
    }
*/

   /* @Test
    void trainOrIndividually() throws IOException {
        double upper = 0.9;
        double lower = 0.1;

        double min = 0.2;
        double max = 0.8;
        double[][] inputOR00 = {{0, 0, 0}}; // confirmed
        double[][] inputOR10 = {{1, 0, 1}}; // confirmed
        double[][] inputOR01 = {{0, 1, 1}};
        double[][] inputOR11 = {{1, 1, 1}};

        NetworkArrays net = createNetRandom(min, max, upper, lower, 0.001);

        net.train(inputOR10);
        NetworkArrays net1 = createNetRandom(min, max, upper, lower, 0.01);
        net1.train(inputOR11);
        NetworkArrays net2 = createNetRandom(min, max, upper, lower, 0.01);
        net2.train(inputOR01);
        NetworkArrays net3 = createNetRandom(min, max, upper, lower, 0.01);
        net3.train(inputOR00);
    }
*/

   /* @Test
    void shuffleTest(){
        double[][] inputXOR = {{0, 0, 0 }, {1, 0 , 1}, {0, 1 , 1}, {1, 1 ,0}};
        double[][] shuffled = Network.shuffleInputs(inputXOR);

        for (int i = 0; i < shuffled.length; i++) {
            System.out.println("For row " + i);
            for (int j = 0; j < shuffled[i].length; j++) {
                System.out.println(shuffled[i][j]);
            }
        }
    }
*/
   /* @Test
    void trainTestAnd() throws IOException {
        double upper = 0.9;
        double lower = 0.1;

        double min = 0;
        double max = 1;
        double[][] inputAND = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}};

        NetworkArrays net = createNetRandom(min, max, upper, lower, 0.3);

        net.train(inputAND);

    }
    */
   /*@Test
    void trainOr() throws IOException {
        double upper = 0.9;
        double lower = 0.1;

        double min = 0;
        double max = 1;
        double[][] inputOR = {{0, 0, 0 },{1, 0 , 1},{0, 1 , 1},{1, 1 ,1}}; // confirmed
        double[][] inputOR10 = {}; // confirmed
        double[][] inputOR01 = {};
        double[][] inputOR11 = {};

        NetworkArrays net = createNetRandom(min,max,upper,lower,0.3);

        net.train(inputOR);
    }
*/

    @Test
    void trainXOR() throws IOException {
        double upper = 0.9;
        double lower = 0.1;

        double min = 0;
        double max = 1;
        double[][] inputXOR = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 0}};

        double[][] inputAND = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}};

        NetworkArrays net = createNetRandom(min, max, upper, lower, 0.3);

        net.train(inputXOR,inputXOR);
    }

    /*
    @Test
    void trainAll() throws IOException {

        trainOr();
        trainXOR();
        trainAnd2();
    }
*/
    /*
    @Test
    void trainAnd2() throws IOException {
        double upper = 0.9;
        double lower = 0.1;

        double min = 0;
        double max = 1;

        double[][] inputAND = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}};

        NetworkArrays net = createNetRandom3(min, max, upper, lower, 0.3);
        net.train(inputAND);
    }
*/
    /*@Test
    void testOrWithLoadedFile() throws IOException {
        String path = "C:\\Users\\madsk\\Documents\\inputs.csv";
        double upper = 0.9;
        double lower = 0.1;

        double min = 0;
        double max = 1;
        double[][] inputs = NetworkArrays.readInputFromFile(path);
        NetworkArrays net = createNetRandom(min,max,upper,lower,0.3);

        net.train(inputs);
    }
*/
   /* @Test
    void getTestData() throws FileNotFoundException {
        String path = "C:\\Users\\madsk\\Documents\\trainset.csv";
        double[][] training = NetworkArrays.readInputFromFile(path);

        System.out.println(training.length);
        System.out.println(training[0].length);
        for (int i = 0; i < training.length; i++) {
            if (training[i].length != 19){
                System.out.println("oh no at " + i);
            }
        }

        for (int i = 0; i < training.length; i++) {
            System.out.println(i);
            String temp = "";
            for (int j = 0; j < training[i].length; j++) {
                temp = temp + training[i][j] + " ";
            }
            System.out.println(temp);
        }
    }
*/
   /* @Test
    void getTestData2() throws FileNotFoundException {
        String path = "C:\\Users\\madsk\\Documents\\trainset2fix2.csv";
        double[][] training = NetworkArrays.readInputFromFile(path);

        System.out.println(training.length);
        System.out.println(training[0].length);
        for (int i = 0; i < training.length; i++) {
            if (training[i].length != 19){
                System.out.println("oh no at " + i);
            }
        }

        for (int i = 0; i < training.length; i++) {
            System.out.println(i);
            String temp = "";
            for (int j = 0; j < training[i].length; j++) {
                temp = temp + training[i][j] + " ";
            }
            System.out.println(temp);
        }
    }
*/
   /* @Test
    void trainWithTestData() throws IOException {
        String path = "C:\\Users\\madsk\\Documents\\trainset.csv";
        double[][] training = NetworkArrays.readInputFromFile(path);

        int inputSize = 18;
        int hiddensize = 19;
        int outsize = 1;

        double min = -1;
        double max= 1;

        double lower = 0.1;
        double upper = 0.9;
        double learningrate = 0.001;
        double[][] inputLayer = generateLayer(inputSize,hiddensize,min,max);
        double[][] hiddenlayer = generateLayer(hiddensize,outsize,min,max);
        double[][] outputLayer = generateLayer(outsize,0,min,max); //

        NetworkArrays net = new NetworkArrays(inputLayer,hiddenlayer,outputLayer,learningrate,upper,lower);

        net.train(training);

    }
*/
   /* @Test
    void trainWithTestData2() throws IOException {
        String path1 = "C:\\Users\\madsk\\Documents\\trainingsets\\fullsetfix.csv";
        String path2 = "C:\\Users\\madsk\\Documents\\trainingsets\\controlgroup.csv";
        double[][] training = NetworkArrays.readInputFromFile(path1);
        double[][] control = NetworkArrays.readInputFromFile(path2);

        int inputSize = training[0].length-1;
        int hiddensize = 16;
        int outsize = 1;

        double min = -0.3;
        double max= 0.3;

        double lower = 0.2;
        double upper = 0.9;
        double learningrate = 0.0005;
        double[][] inputLayer = generateLayer(inputSize,hiddensize,min,max);
        double[][] hiddenlayer = generateLayer(hiddensize,outsize,min,max);
        double[][] outputLayer = generateLayer(outsize,0,min,max); //

        NetworkArrays net = new NetworkArrays(inputLayer,hiddenlayer,outputLayer,learningrate,upper,lower);

        System.out.println(inputLayer[0].length);
        net.train(training);
        System.out.println("Control is of length " + control.length);
        int correct =0;
        int falseNegative=0;
        int falsePositive=0;
        for (int i = 0; i < control.length; i++) {
            double out =net.forwardProp(control[i]);
            double correctedOutput = net.correctedOut(upper,lower,out);
            double target = control[i][control[i].length-1];
            if(correctedOutput == target){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " Correct");
                correct++;
            }
            if(correctedOutput == 1.0 && target== 0.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " false positive");
                falsePositive++;
            }
            if (correctedOutput == 0.0 && target == 1.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " false negative");
                falseNegative++;
            }


            System.out.println("output: " + out + " " + "Target " + control[i][control[i].length-1]);
        }
        System.out.println("Correct: " + correct);
        System.out.println("False negative: " + falseNegative);
        System.out.println("False positive: " + falsePositive);

    }
*/
    /*@Test
    void testWithTrainedWeights() throws FileNotFoundException {
        String path1 = "C:\\Users\\madsk\\Documents\\trainingsets\\fullsetfix.csv"; // not ideal since this is the trainingset

        double[][] control = NetworkArrays.readInputFromFile(path1);

        int inputSize = control[0].length-1;
        int hiddensize = 16;
        int outsize = 1;

        double min = -0.3;
        double max= 0.3;

        double lower = 0.2;
        double upper = 0.9;
        double learningrate = 0.005;
        double[] inneuron1 = {-0.1422824377295429,0.3042923335414057,-0.4649024949076659,0.2637798521341422,-6.3162893858184885,0.00456608177510193,1.0108801907039355,0.06513760602532045,-0.3821275269803889,0.18960441156664726,-0.08926139078150788,0.011626958263396634,-0.3044351655377236,0.29131186930794084,0.28328042076820015,0.10973243605770708};
        double[] inneuron2 = {0.22244883513997238,0.02250101126437756,-0.022583509881849845,-0.021725903176563446,0.07300538801920459,-0.19260983424730047,0.09356247758582482,0.04348412217393251,-0.022144125874606128,0.2741417988352147,0.2632462399612597,-0.1948477921837232,-0.19778837836410204,-0.2963666183332106,0.15546671645050744,-0.10219391492450437};
        double[] inneuron3 = {-0.03727798400058513,0.01210971169245978,-0.007723841147110026,-0.025899852717169162,-0.04842925624148066,-0.010838464587050218,-1.766191233219104,-3.658610154877344,-0.019885417770578624,0.04126162106054858,0.052651306913179846,-0.08664982935512677,4.842876789901067,-0.18054153865477748,0.14046902513529916,0.14771022369914746};
        double[] inneuron4 = {0.07095600514318884,-0.22509023239776121,-0.07782918125978167,-0.026469994741295744,-0.05430962716180662,0.032881404108934253,-0.23444761986090845,-0.10611499932865101,0.0696949924886594,-0.1443556610569718,0.16260833130976554,0.02740806842521526,0.19991851941642347,0.17024079265840647,0.07880378294565549,-0.04763358549086871};
        double[] inneuron5 = {-0.18080649042856933,-0.1761240570560405,0.28295813261007113,-0.12011669847958131,-0.23620824397785073,-0.296793491862081,-0.03884602773848189,-0.18624671680197274,-0.2513408139861521,-0.1036334320898924,0.1239245364817203,0.09092812032133069,-0.007073806509997782,-0.003133341844753057,0.10497092765816979,-0.019129879864778765};
        double[] inneuron6 = {0.1225979294750551,0.7429826085336072,-0.46988975338233413,-0.42820775997765936,7.165811564046045,0.2486272901127074,0.17894854760040652,-0.27859570788845706,-0.4139188531062126,-0.16333558252959066,-0.2770920191480049,0.2041530106745116,-0.08003250963729817,-0.2306135097277624,-0.17532053264480862,0.22827693516182307};
        double[] inneuron7 = {-0.21366307535154044,-0.13692678477711823,0.03789118052536238,-0.016030193912507994,-7.329097180113822,-0.13225654126897804,1.3703236921185586,0.4239036825648619,0.18484830123334037,-0.148342984410344,-0.24753127390833288,0.10256090965447803,0.49575617774107283,-0.0017461657251205604,-0.22638665068684122,0.13295604015782417};
        double[] inneuron8 = {0.276685361355641,-0.20892641689390834,0.16204737873318364,0.4486498512094918,0.6970539746821357,-0.19325706341165902,3.3579324803168697,-1.8408034578912107,0.49353446771380527,0.013334153126681866,-0.022140194773872676,0.12259996181208747,0.05860899344861282,0.23556125198605266,0.23655654012879954,0.16529100010008607};
        double[] inneuron9 = {0.2906211918505824,-0.14017206282460237,-0.026460261247320583,0.045565193288017246,1.2071985632243982,0.07912620193939805,2.230667028161521,-1.4476913420583002,0.1261086939423952,-0.1403271983532997,0.001521875341734114,-0.2556041806711255,-0.13862484103754913,-0.13139611350939684,-0.21379707563831812,-0.20642434793332992};
        double[] inneuron10 = {-0.20242377876669654,-0.2370848973503418,-0.20943470430965833,-0.08775455477243273,-1.361665357370612,0.08808408490238724,-0.13669421619015046,0.18531266898804644,0.1259503416159151,-0.08887269732024389,-0.002065746894985911,-0.22060270249828623,0.3642894284523748,0.10413764320641153,0.031673528616824986,-0.24010258308650115};
        double[] inneuron11 = {0.13328304712560635,-0.2844986645612079,-0.053884827092619075,-0.2545988164001816,0.05505509602927677,0.03548615639682534,-0.04626855682416664,-0.19440544266682552,0.13637326451430098,-0.17161984680256048,-0.13358540738889327,-0.1903435766372673,0.023054931684479973,0.1832766311886576,0.21688873935614333,-0.06234576062728758};
        double[] inneuron12 = {0.281125513591735,-0.14709168074399853,-0.12983987926670898,-0.29955062897161666,2.102257720918411,-0.16517508913151585,-1.0005832909228134,1.6606172478606773,-0.042865245581393305,0.21743338627323405,0.22964979816989772,0.22704290868262247,-0.1126539215010752,-0.18115075285776638,-0.28751011909341956,0.015588337517304337};
        double[] inneuron13 = {-0.22435709486037822,-0.5598799202622533,0.04614354539501799,0.48764195927285253,-7.988200971122179,0.27817810067640397,0.42430954358335193,0.09126199189678905,0.4020984927933976,0.10927470659408452,-0.11823752912463449,0.0032470710687928636,-0.034681075297908225,-0.005742505522012705,-0.0685213024337798,0.1495299923261012};
        double[] inneuron14 = {-0.002451105011936035,0.1430306043072127,-0.17805189417209585,0.2973274800404894,-3.5379880103030232,0.17879285943939371,0.45203325917212916,0.1624268066558028,0.3263275859643293,-0.20943786477381873,0.25386269727637656,0.2624383417229379,0.4838239404116222,-0.1756101518792704,0.17400286181845456,0.039267296867846316};
        double[] inneuron15 = {-0.06547548347683453,-0.42846035623028045,0.7157137343315206,0.2713756274995109,-12.178311469085319,-0.2466460326021001,0.5016670319990957,0.7531029704547799,0.48146458029009864,0.07101600644802734,0.007860446992776184,0.24853210981157853,0.4247459194832734,0.24404641395073892,0.026109513372783038,-0.019781774340614285};
        double[] inneuron16 = {-0.006303358939455124,0.22330643722198487,0.09547800630966852,-0.3241961225548289,2.4221264122364827,0.2556238222284298,-0.36594059399319845,-0.09250794355874464,-0.36036546682984816,0.2533976790508745,0.05294624892640137,-0.07548005897873153,-0.680951764916356,-0.2161957085565833,-0.25067023412463063,-0.01769709685767692};
        double[] inneuron17 = {0.20572381496628525,-0.5883977800900059,0.0795730697500411,0.18697949098351263,-5.017414849361386,0.16189162183645578,0.06264293747722867,2.721369082424463,-0.022363949014630836,0.28524737909372194,-0.2507940531954789,0.02352574468157214,0.19533648463559902,-0.11917923910818351,-0.18083124972791678,-0.1510523425140583};
        double[] inneuron18 = {0.25863086136741914,0.33699810949603265,-0.33538042610303465,-0.28065966660783215,3.6631990486227095,-0.09968064980626339,0.028668512073605223,-0.12423947064617408,0.1068693050507153,-0.1529416477393824,-0.1448064361615472,0.19606229781923679,0.030867836655748722,0.20015210850689857,-0.07321185233465143,0.08371202488548156};

        double[] hidden1 = {-0.006281940741231158};
        double[] hidden2 = {-2.000594166032418};
        double[] hidden3 = {2.0042919192644106};
        double[] hidden4 = {1.1544372000999537};
        double[] hidden5 = {-13.02770542218441};
        double[] hidden6 = {-0.25714289609997315};
        double[] hidden7 = {3.9826536996269124};
        double[] hidden8 = {4.252232439697982};
        double[] hidden9 = {1.8425038600615005};
        double[] hidden10 = {-0.020332483823704697};
        double[] hidden11 = {-0.17587014375826887};
        double[] hidden12 = {-0.19824121616948676};
        double[] hidden13 = {9.486834209177378};
        double[] hidden14 = {0.07133816914725088};
        double[] hidden15 = {-0.2888158540370778};
        double[] hidden16 = {1.6638009117715027};

        double[] outn = {1};

        double[][] inputLayer = {inneuron1,inneuron2,inneuron3,inneuron4,inneuron5,inneuron6,inneuron7,inneuron8,inneuron9,inneuron10,inneuron11,inneuron12,inneuron13,inneuron14,inneuron15,inneuron16,inneuron17,inneuron18};
        double[][] hiddenLayer = {hidden1,hidden2,hidden3,hidden4,hidden5,hidden6,hidden7,hidden8,hidden9,hidden10,hidden11,hidden12,hidden13,hidden14,hidden15,hidden16};
        double[][] outLayer = {outn};

        NetworkArrays net = new NetworkArrays(inputLayer,hiddenLayer,outLayer,learningrate,upper,lower);


        int correct =0;
        int falseNegative=0;
        int falsePositive=0;
        for (int i = 0; i < control.length; i++) {
            double out =net.forwardProp(control[i]);
            double correctedOutput = net.correctedOut(upper,lower,out);
            double target = control[i][control[i].length-1];
            if(correctedOutput == target){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " Correct");
                correct++;
            }
            if(correctedOutput == 1.0 && target== 0.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " false positive");
                falsePositive++;
            }
            if (correctedOutput == 0.0 && target == 1.0){
                System.out.println(out + " , " + correctedOutput + " target: " + target + " false negative");
                falseNegative++;
            }


            System.out.println("output: " + out + " " + "Target " + control[i][control[i].length-1]);
        }
        System.out.println("Correct: " + correct);
        System.out.println("False negative: " + falseNegative);
        System.out.println("False positive: " + falsePositive);


    }
     */
    @Test
    void trainWithTestData49() throws IOException {
        String path1 = "C:\\Users\\madsk\\Documents\\trainingsets\\bothsetsfixcollected.csv";
        String path2 = "C:\\Users\\madsk\\Documents\\trainingsets\\testset14.csv";
        double[][] trainingLoad = NetworkArrays.readInputFromFile(path1);
        //double[][] control = NetworkArrays.readInputFromFile(path2);

        double[][] shuffledTraining = Network.shuffleInputs(trainingLoad);
        double[][] training = new double[49][];
        double[][] control = new double[14][];

        //med shuffle
        for (int i = 0; i < 49; i++) {
            training[i] = shuffledTraining[i];
        }
        for (int i = 49; i < shuffledTraining.length ; i++) {
            control[i-49] = shuffledTraining[i]; // ej
        }
        assertFalse(Arrays.equals(training[training.length-1],control[0]));

        //uden shuffle
/*        for (int i = 0; i < 49; i++) {
            training[i] = trainingLoad[i];
        }
        for (int i = 49; i < trainingLoad.length ; i++) {
            control[i-49] = trainingLoad[i]; // ej
        }*/
        //assertFalse(Arrays.equals(training[training.length-1],control[0]));

        int inputSize = training[0].length-1;
        int hiddensize = 10;
        int outsize = 1;

        double min = -0.3;
        double max= 0.3;

        double lower = 0.2;
        double upper = 0.9;
        double learningrate = 0.0001;
        double[][] inputLayer = generateLayer(inputSize,hiddensize,min,max);
        double[][] hiddenlayer = generateLayer(hiddensize,outsize,min,max);
        double[][] outputLayer = generateLayer(outsize,0,min,max); //

        NetworkArrays net = new NetworkArrays(inputLayer,hiddenlayer,outputLayer,learningrate,upper,lower);

        System.out.println(inputLayer[0].length);
        System.out.println(learningrate);
        net.train(training,control);
        net.upper = 0.5;
        net.lower = 0.5;
        int expectedTrue = 0;
        int expectedFalse = 0;

        for (int i = 0; i < control.length; i++) {
            if (control[i][control[i].length-1] == 1.0){
                expectedTrue++;
            }
            if (control[i][control[i].length-1] == 0.0){
                expectedFalse++;
            }
        }
        int truePositive = 0;
        int trueNegative = 0;
        int falseNegative = 0;
        int falsePositive = 0;
        int leftover= 0;
        System.out.println("Control time owo");
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


            System.out.println("output: " + out + " " + "Target " + control[i][control[i].length-1]);
        }
        leftover = control.length-falseNegative-falsePositive-trueNegative-truePositive;
        System.out.println("Control is of length " + control.length);
        System.out.println("true positive: " + truePositive + " out of " + expectedTrue);
        System.out.println("True negative: " + trueNegative + "out of " + expectedFalse);
        System.out.println("False negative: " + falseNegative);
        System.out.println("False positive: " + falsePositive);
        System.out.println("leftover: " + leftover);

    }

    @Test
    void testBest10OnfullSet() throws FileNotFoundException {

        String path = "C:\\Users\\madsk\\Documents\\trainingsets\\trainingGoodfix.csv";
        double[] in1 ={0.06969807884312343,-0.21038738858245795,0.011247888407666008,-0.25638856580879765,-0.19567050565148297,5.009547159037065,0.2589533082454277,0.24486229880252178,-0.26061822075013724,0.016313231979510897};
        double[] in2={-0.2123608913251537,0.41806517563229495,-0.03401206256054405,-0.4118461352173203,0.23781147681994544,-0.06199470984324194,0.23995261951453234,-0.22478636837090063,-0.23446040894420092,-0.4065761243100491};
        double[] in3={-0.13547434734130542,-1.7719104895369728,0.1562519735508096,1.7638855121430816,0.2891853507026735,0.062084603894223954,0.25190835714898246,-0.27199750311058374,-0.24799063413310815,1.7253613567624073};
        double[] in4={0.22534465000528853,0.11542693673798038,0.02116688342451073,0.19545559458333517,-0.09920694833899804,0.2981241988904281,0.15257838746078767,0.250188235366466,-0.272841183563954,-0.007540535146547733};
        double[] in5={-0.16487138909915636,0.2730946429901602,-0.16572900714399746,0.07150732284525488,0.21595732812083257,-0.1997348978282041,-0.27701217055547317,-0.266647442390821,0.04890393389263881,-0.21110164482390004};
        double[] in6={-0.21409068979365,-0.021133738160463857,-0.5404730246920556,-0.5686275505174891,0.14781891176307266,-4.518129926753042,0.29364174116663816,0.12764873285647563,0.2448999458318824,-0.5791304620571653};
        double[] in7={-0.22018399893814256,-0.17364909622714836,0.10503247065639827,0.4909228773424874,0.1272415242085497,5.943887820388063,0.16670411213572744,0.03169819893854212,-0.23242125417307882,0.3792373420669711};
        double[] in8={0.26513354126309874,-0.22406049483278115,-0.5458162462948759,0.340777396569685,-0.23768969946699367,0.3515504542440831,-0.04118414960129895,0.28322048971383285,-0.184560048875161,0.3788769008649729};
        double[] in9={-0.04336444652742517,-0.0998380142686433,0.5101978687572871,0.5118372209795979,0.2608580965032095,-0.003190513003577394,-0.004896127942927318,-0.11097729547160305,-0.046857726870406284,0.4498850169305787};
        double[] in10={0.06005553456505524,-0.33312106381441925,-0.34353648284160004,0.35139084373163126,-0.11758398062064336,-7.880211891510874,-0.10292219790525874,0.29752038933708336,-0.05220387985448467,0.46729775285306924};
        double[] in11={-0.1948874566367711,0.08594648306191738,0.08985972539826159,0.1508035852530341,-0.028137889943280092,-0.02594636903282982,-0.0777396720835842,-0.2956319739635646,0.1548143624085509,-0.20024957142370453};
        double[] in12={0.0034080448985503863,0.16483781744373324,-0.26421676850943826,-0.05338837101778505,0.016116794504832173,-1.2401106385384713,-0.1725458628778004,0.030437043807047644,0.026358997425844948,-0.15065387452229814};
        double[] in13={0.2577868393438779,-0.03705697996438724,0.019566166587103518,-0.2405511535326966,-0.14114584473948794,5.673879540107983,-0.008910096910137848,-0.06261278495452086,0.01996039963192674,0.11330786867756089};
        double[] in14={0.05134665046327993,0.21606396147596021,0.2877409417106203,0.20366389928100437,0.09145272275606155,6.46857858669434,0.2780577998643003,-0.06379795204789693,0.18089838331395244,0.05360242909192993};
        double[] in15={-0.21058311012993156,0.19320655692383987,0.3345547215212554,-0.019962211709170175,-0.15127128517734723,6.540321550362578,0.19206856867598363,-0.2660770373803637,-0.12938266175646293,-0.09036505692484408};
        double[] in16={0.009081709608799254,-0.20520678147099025,-0.22315347188895784,-0.05757194658873221,-0.2587824105950253,-3.1205281745162234,0.18331181437162306,0.1868801327057338,0.040987118911964986,-0.0661456513385536};
        double[] in17={0.16026934475095694,0.04920988431570684,-0.07581367770928721,-0.418596478660406,0.21842124005933464,1.4219396417707892,-0.03562038872869755,0.28888596030069635,0.11941855498794703,-0.007813438260576394};
        double[] in18={-0.1576681802154529,-0.11559455200698326,-0.30300877576113355,0.31159236331078033,0.18735285493846818,-3.4356153838858865,0.20991425186939008,-0.26972955677749944,0.08633336504552719,0.1753900647599738};
        double[] hi1={0.2838567904744517};
        double[] hi2 ={-3.069567737755945};
        double[] hi3 ={1.9611400650361483};
        double[] hi4={3.1236819655866994};
        double[] hi5={-1.504714148652476};
        double[] hi6={15.018234809351629};
        double[] hi7={-1.3623711431211811};
        double[] hi8={0.029590147643048983};
        double[] hi9={-0.23966634321762165};
        double[] hi10={2.6958993935795843};
        double[] out = {0};
        double[][] inputLayer = {in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11,in12,in13,in14,in15,in16,in17,in18};
        double[][] hiddenLayer = {hi1,hi2,hi3,hi4,hi5,hi6,hi7,hi8,hi9,hi10};
        double[][] outoutLayer = {out};

        double[][] training = NetworkArrays.readInputFromFile(path);
        int inputSize = training[0].length-1;
        int hiddensize = 10;
        int outsize = 1;

        double min = -0.3;
        double max= 0.3;

        double lower = 0.3;
        double upper = 0.7;
        double learningrate = 0.0001;

        NetworkArrays net = new NetworkArrays(inputLayer,hiddenLayer,outoutLayer,learningrate,upper,lower);
        int actualPositive = 0;
        int actualNegative= 0;
        int truePositive = 0;
        int trueNegative = 0;
        int falsePositve = 0;
        int falseNegative = 0;
        for (int i = 0; i < training.length; i++) {
            double target = training[i][training[i].length-1];
            if (target == 1.0){
                actualPositive++;
            }
            if (target ==0.0){
                actualNegative++;
            }
            double output = net.forwardProp(training[i]);
            double correctedOutput = net.correctedOut(upper,lower,output);
            System.out.println("output " + output + " " + correctedOutput + " " + target);
            if(correctedOutput == 1.0 && target == 1.0){
                truePositive++;
            }
            if(correctedOutput == 0.0 && target== 0.0){
                trueNegative++;
            }
            if (correctedOutput == 0.0 && target == 1.0){
                falseNegative++;
            }
            if (correctedOutput == 1.0 && target == 0.0){
                falsePositve++;
            }
        }

        double missedPositive = actualPositive-truePositive;
        double missedNegative = actualNegative - trueNegative;
        System.out.println("Control is of length " + training.length);
        System.out.println("true positive: " + truePositive + " out of " + actualPositive + " " + truePositive/actualPositive);
        System.out.println("True negative: " + trueNegative + "out of " + actualNegative);
        System.out.println("False negative: " + falseNegative);
        System.out.println("False positive: " + falsePositve);
        System.out.println("missed positive " +missedPositive);
        System.out.println("missed negative " + missedNegative);

    }

    @Test
    void trainBIG() throws IOException {
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
        int leftover= 0;
        System.out.println("Control time owo");
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


            //System.out.println("output: " + out + " " + "Target " + control[i][control[i].length-1]);
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