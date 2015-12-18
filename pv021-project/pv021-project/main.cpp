#include "Test.hpp"
#include <fstream>

string TRAIN_DATA_SET = "/Users/jozefmarko/Documents/School/Neurónové siete/train-images-idx3-ubyte";
string TRAIN_LABELS = "/Users/jozefmarko/Documents/School/Neurónové siete/train-labels-idx1-ubyte";
string TEST_DATA_SET = "Users/jozefmarko/Documents/School/Neurónové siete/t10k-images-idx3-ubyte";
string TEST_LABELS = "Users/jozefmarko/Documents/School/Neurónové siete/t10k-labels-idx1-ubyte";
string OUTPUT_PATH = "/Users/jozefmarko/Development/Github/RBM-Autoencoder/pv021-project/pv021-project/outputs/";
string NAME_OF_OUTPUT_FILE = "autoencoder_errors";
vector<int> HIDDEN_LAYERS = {20};
vector<int> REPETITIONS = {1,3};
vector<float> LEARNING_RATES = {0.01, 0.1, 0.2};
vector<float> MOMENTUM = {0, 0.9};
//vector<int> HIDDEN_LAYERS = {1,2,3,4,5,7,9,11,13,15,17,20,23,26,30,40, 200};

int main(int argc, const char * argv[]) {
    
    ifstream train_data_set, train_labels, test_data_set, test_labels;
    train_data_set.open(TRAIN_DATA_SET);
    train_labels.open(TRAIN_LABELS);
    test_data_set.open(TEST_DATA_SET);
    test_labels.open(TEST_LABELS);
    
    for(int hidden_layers : HIDDEN_LAYERS){
        ofstream output_file;
        Test * test = new Test(1);
        //output_file.open(OUTPUT_PATH + NAME_OF_OUTPUT_FILE + "_" + std::to_string(hidden_layers));
        test->test(hidden_layers, cout, train_data_set, train_labels, test_data_set, test_labels);
    }
    return 0;
}
