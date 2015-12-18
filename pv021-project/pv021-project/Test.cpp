//
//  Test.cpp
//  pv021-project
//
//  Created by Jozef Marko on 17/12/15.
//  Copyright © 2015 Jozef Marko. All rights reserved.
//

#include "Test.hpp"
#include <iostream>
#include <fstream>
#include <ctime>

int CLASSES = 10;
int EXAMPLES = 60000;
int EXAMPLES_TESTS = 10000;
int HEADER_LABELS = 8;
int HEADER_IMAGES = 16;
int IMAGE_SIZE = 28*28;
int CONTENT_SIZE = EXAMPLES * IMAGE_SIZE;
int CONTENT_SIZE_TEST = EXAMPLES_TESTS * IMAGE_SIZE;
int NUM_OF_INPUT = IMAGE_SIZE/50;
vector<int> MILESTONES = {1,10,20,40,60,80,100,150,200,250,300,400,500,700,1000,2000,5000,10000,20000,40000,60000};

void Test::trainCharasteristicNetwork(ostream& output_file, TrainingInterface& network, char* data_set){
    MemoryBlock input(IMAGE_SIZE);
    
    int examples = EXAMPLES;
    float total_error = 0;
    for(int i = 0; i < repetition_m; i++){
        float round_error = 0;
        vector<int>::iterator it = MILESTONES.begin();
        for (int image = 0; image < examples; image++) {
            for(int pixel = image*IMAGE_SIZE; pixel<(image + 1) * IMAGE_SIZE; pixel++){
                input.data[pixel-image*IMAGE_SIZE] = round(((int)((unsigned char)data_set[pixel]))/(float)255);
            }
            float error = network.Train(input, 1);
            total_error+=error;
            round_error+=error;
            if(it != MILESTONES.end() && *it == image + 1){
                it++;
                float rounds = i*EXAMPLES + image + 1;
                output_file<<"Celkovy priemerny squared error po "<<rounds<<" kolach: "<<total_error/rounds<<" , priemerny squared error od "<<i*EXAMPLES<<" kola: "<<round_error/(image + 1)<<endl;
            }
        }
    }
}

void Test::trainNormalNetwork(ostream& output_file, int examples, int repetitions, bool with_training, MyNetwork& network, char * data_set, char * labels, RBM& rbm){
    MemoryBlock input(IMAGE_SIZE);
    MemoryBlock features(NUM_OF_INPUT);
    MemoryBlock output(CLASSES);
    
    float total_error = 0;
    int wrong_answers = 0;
    for(int i = 0; i < repetitions; i++){
        float round_error = 0;
        float round_wrong_answers = 0;
        vector<int>::iterator it = MILESTONES.begin();
        for (int image = 0; image < examples; image++) {
            for(int pixel = image*IMAGE_SIZE; pixel<(image + 1) * IMAGE_SIZE; pixel++){
                input.data[pixel-image*IMAGE_SIZE] = round(((int)((unsigned char)data_set[pixel]))/(float)255);
                cout<<input.data[pixel-image*IMAGE_SIZE]<<endl;
            }
            
            for (int c = 0; c < CLASSES; c++) {
                if(c == (labels[image])){
                    output.data[c] = 1;
                }
                else{
                    output.data[c] = 0;
                }
            }
            
            rbm.Encode(input, features);
            pair<float,int> result = with_training ? network.TrainWithResult(features, output) : network.JustResult(features, output);
            total_error+=result.first;
            round_error+=result.first;
            
            if(result.second != labels[image]){
                ++wrong_answers;
                ++round_wrong_answers;
            }
            
            if(it != MILESTONES.end() && *it == image + 1){
                it++;
                float rounds = i*EXAMPLES + image + 1;
                output_file<<"Celkovy priemerny squared error po "<<rounds<<" kolach: "<<total_error/rounds<<" , priemerny squared error od "<<i*EXAMPLES<<" kola: "<<round_error/(image + 1)<<endl;
                output_file<<"Celkovy miss rate po "<<rounds<<" kolach: "<<wrong_answers/rounds<<" , priemerny squared error od "<<i*EXAMPLES<<" kola: "<<round_wrong_answers/(image + 1)<<endl;
                
            }
        }
    }
}

int Test::test(int num_hidden_layers, ostream& output_file, ifstream& train_data_set, ifstream& train_labels, ifstream& test_data_set, ifstream& test_labels) {
    
    long long timestamp = time(0);
    
    char * data_set = (char*)malloc(CONTENT_SIZE);
    train_data_set.read(data_set, HEADER_IMAGES); // header
    train_data_set.read(data_set, CONTENT_SIZE);

    char * labels = (char*)malloc(EXAMPLES);
    train_labels.read(labels, HEADER_LABELS); // header
    train_labels.read(labels, EXAMPLES);
    
    output_file<<"----Traning RBM----"<<endl;
    RBM rbm(IMAGE_SIZE, NUM_OF_INPUT, 0.08f, 0.0f);
    trainCharasteristicNetwork(output_file, rbm, data_set);
    
    output_file<<endl<<endl<<"----Traning ForwardNetwork----"<<endl;
    LogisticFunction logistic;
    MyNetwork autoEncoder(NUM_OF_INPUT, num_hidden_layers, CLASSES, &logistic, 0.2511f, 0.0f);
    trainNormalNetwork(output_file, EXAMPLES, repetition_m, true, autoEncoder, data_set, labels, rbm);
    
    output_file<<endl<<endl<<"----Testing ForwardNetwork----"<<endl;
    delete data_set;
    delete labels;
    data_set = (char*)malloc(CONTENT_SIZE_TEST);
    test_data_set.read(data_set, HEADER_IMAGES); // header
    test_data_set.read(data_set, CONTENT_SIZE_TEST);
    
    labels = (char*)malloc(EXAMPLES_TESTS);
    test_labels.read(labels, HEADER_LABELS); // header
    test_labels.read(labels, EXAMPLES_TESTS);
    trainNormalNetwork(output_file, EXAMPLES_TESTS, 1, true, autoEncoder, data_set, labels, rbm);
    
    MemoryBlock input(IMAGE_SIZE);
    MemoryBlock output(CLASSES);
    
    output_file<<endl<<"Total time: "<<time(0) - timestamp<<endl;
    
    return 0;
}
