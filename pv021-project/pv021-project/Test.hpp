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
#include "AutoEncoder.hpp"
#include "RBM.hpp"

class Test {
    int repetition_m;
    void trainCharasteristicNetwork(ostream& output, TrainingInterface& network, char* data_set);
public:
    Test(int repetition) : repetition_m(repetition){
    }
    int test(int num_hidden_layers, ostream& output_file, ifstream& train_data_set, ifstream& train_labels, ifstream& test_data_set, ifstream& test_labels);
    void trainNormalNetwork(ostream& output_file, int examples, int repetitions, bool with_training, MyNetwork& network, char * data_set, char * labels, RBM& rbm);
};


#endif /* Test_hpp */
