
#include <DenseLayer.hpp>
#include <Matrix.hpp>
#include <algorithm>
#include <iostream>
#include <print>
#include <cmath>

using DenseLayer = bpy::DenseLayer;
using Matrix = bpy::Matrix; // creates an alias named Matrix

double relu(double x) { return std::max(0.0, x); }
// Activation Function: Sigmoid (for final probability mapping)
double sigmoid(double x) {return 1.0 / (1.0 + std::exp(-x));}

int main() {
    try {
        std::println("=== Neural Network Initialization ===\n");

        // 1. Define Network Dimensions
        const std::size_t input_features = 4; // e.g., sepal length, sepal width, petal length, petal width
        const std::size_t hidden_neurons = 8; // Size of our hidden processing layer
        const std::size_t output_neurons = 3; // e.g., 3 iris plant classifications (Setosa, Versicolor, Virginica)

        // 2. Create the Layer Stack
        // Pass the global `relu` function pointer to handle non-linearity automatically
        std::println("[INFO] Building a 3-layer Network Configuration...");
        
        DenseLayer hidden_layer(input_features, hidden_neurons, relu);

        std::println("\n--- Freshly Manufactured Hidden Layer Weights ---");
        hidden_layer.GetWeights().print();

        // If you want a completely new batch of weights mid-execution:
        std::println("[INFO] Regenerating weight distributions...");
        hidden_layer.remanufacture_weights(-0.5, 0.5);
        hidden_layer.GetWeights().print();

    // OR
        //DenseLayer hidden_layer(input_features, hidden_neurons, relu);
        
        DenseLayer output_layer(hidden_neurons, output_neurons, relu);

        // 3. Generate Mock Input Data
        // A [1 x 4] matrix representing a single sample vector
        Matrix input_sample(1, input_features);
        input_sample.randomize(0.0, 5.0); // Simulate positive measurements

        std::println( "\n--- Input Layer Data ---");
        input_sample.print();

        // 4. Perform Forward Pass Evaluation
        std::println( "--- Processing Network Layers ---");
        
        // Pass input into Hidden Layer -> Outputs [1 x 8]
        Matrix hidden_output = hidden_layer.forward(input_sample);
        std::println( "[OK] Hidden Layer calculated. Dimensions: {} x {}", hidden_output.Rows(), hidden_output.Cols());

        // Pass hidden output into Output Layer -> Outputs [1 x 3]
        Matrix final_predictions = output_layer.forward(hidden_output);
        std::println("[OK] Output Layer calculated. Dimensions: {} x {}", final_predictions.Rows(), final_predictions.Cols());

        // 5. Output Results
        std::println("\n--- Final Network Predictions (Post-ReLU) ---");
        final_predictions.print();

        output_layer.save_to_csv("model/weights.csv","model/bias.csv");

    } 
    catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Network Execution Failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}