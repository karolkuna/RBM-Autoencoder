//
//  Test.hpp
//  pv021-project
//
//  Created by Jozef Marko on 17/12/15.
//  Copyright © 2015 Karol Kuna. All rights reserved.
//

#ifndef Test_hpp
#define Test_hpp

#include <stdio.h>
#include "MyNetwork.hpp"
#include "AutoEncoder.hpp"
#include "RBM.hpp"

class Test {
    int repetition_m, hidden_neurons, neurons;
    float momentum, learning_rate;
    bool autoencoder;
    void trainCharasteristicNetwork(ostream& output, TrainingInterface& network, char* data_set);
public:
    Test(int repetition, float momentum, float learning_rate, int hidden_neurons, int neurons, bool autoencoder) : repetition_m(repetition), momentum(momentum), learning_rate(learning_rate), neurons(neurons), hidden_neurons(hidden_neurons), autoencoder(autoencoder){
    }
    int test(int num_hidden_layers, ostream& output_file, ifstream& train_data_set, ifstream& train_labels, ifstream& test_data_set, ifstream& test_labels);
    void trainNormalNetwork(ostream& output_file, int examples, int repetitions, bool with_training, MyNetwork& network, char * data_set, char * labels, RBM& rbm);
};


#endif /* Test_hpp */
