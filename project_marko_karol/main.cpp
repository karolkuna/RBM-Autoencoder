#include "Test.hpp"
#include <fstream>

string TRAIN_DATA_SET = "/Users/jozefmarko/Documents/School/Neurónové siete/train-images-idx3-ubyte";
string TRAIN_LABELS = "/Users/jozefmarko/Documents/School/Neurónové siete/train-labels-idx1-ubyte";
string TEST_DATA_SET = "/Users/jozefmarko/Documents/School/Neurónové siete/t10k-images-idx3-ubyte";
string TEST_LABELS = "/Users/jozefmarko/Documents/School/Neurónové siete/t10k-labels-idx1-ubyte";
string OUTPUT_PATH = "/Users/jozefmarko/Development/Github/RBM-Autoencoder/pv021-project/pv021-project/outputs/";
string NAME_OF_OUTPUT_FILE = "autoencoder_errors";
vector<int> HIDDEN_LAYERS = {200}; // hidden neurons
vector<int> REPETITIONS = {4};
vector<float> LEARNING_RATES = {0.1};
vector<float> MOMENTUM = {0};
vector<float> NEURONS_IN_RBM = {300};
bool USE_COUT = true;
bool USE_AUTOENCODER = true; // if false for prec. is used RBM
//vector<int> HIDDEN_LAYERS = {1,2,3,4,5,7,9,11,13,15,17,20,23,26,30,40, 200};

int main(int argc, const char * argv[]) {
    
    ifstream train_data_set, train_labels, test_data_set, test_labels;
    train_data_set.open(TRAIN_DATA_SET);
    train_labels.open(TRAIN_LABELS);
    test_data_set.open(TEST_DATA_SET);
    test_labels.open(TEST_LABELS);
    
    for(int hidden_layers : HIDDEN_LAYERS){
        for(float momentum : MOMENTUM){
            for(float learning_rate : LEARNING_RATES){
                for(int repetitions : REPETITIONS)
                    for(int neurons : NEURONS_IN_RBM){
                        ofstream output_file;
                        train_data_set.clear();
                        train_data_set.seekg(0, ios::beg);
                        train_labels.clear();
                        train_labels.seekg(0, ios::beg);
                        test_data_set.clear();
                        test_data_set.seekg(0, ios::beg);
                        test_labels.clear();
                        test_labels.seekg(0, ios::beg);
                        Test * test = new Test(repetitions, momentum, learning_rate, hidden_layers, neurons, USE_AUTOENCODER);
                        if(USE_COUT){
                            cout<<"repetitions: "<<repetitions<<endl;
                            cout<<"momentum: "<<momentum<<endl;
                            cout<<"learning_rate: "<<learning_rate<<endl;
                            cout<<"hidden_neurons: "<<hidden_layers<<endl;
                            cout<<"neurons in RBM: "<<neurons<<endl;
                            test->test(hidden_layers, cout, train_data_set, train_labels, test_data_set, test_labels);
                        }
                        else{
                            output_file<<"repetitions: "<<repetitions<<endl;
                            output_file<<"momentum: "<<momentum<<endl;
                            output_file<<"learning_rate: "<<learning_rate<<endl;
                            output_file<<"hidden_neurons: "<<hidden_layers<<endl;
                            output_file<<"neurons in RBM: "<<neurons<<endl;
                            output_file.open(OUTPUT_PATH + NAME_OF_OUTPUT_FILE + "_" + std::to_string(hidden_layers)+"_" +  std::to_string(repetitions) +"_" +  std::to_string(learning_rate)+"_" +  std::to_string(momentum)+ "_" +  std::to_string(neurons));
                            test->test(hidden_layers, output_file, train_data_set, train_labels, test_data_set, test_labels);
                        }
                    }
            }
        }
    }
    return 0;
}
