import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Network {
    static double[][] shuffleInputs(double[][] in) {
        //Shuffle the inputs for training
        List<double[]> tempIn = new ArrayList<double[]>();
        tempIn.addAll(Arrays.asList(in));

        Collections.shuffle(tempIn);
        double[][] shuffledInputs = tempIn.toArray(new double[in.length][]);
        return shuffledInputs;
    }
}