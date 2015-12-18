#include "AutoEncoder.hpp"

MyNetwork::MyNetwork(int visibleUnits, int hiddenUnits, int outputUnits, ActivationFunction* activationFunction, float learningRate, float momentumRate) {
        
    network = new FeedforwardNetwork(visibleUnits, {hiddenUnits}, outputUnits, activationFunction, learningRate, momentumRate);
        backprop = new Backpropagation(network);
}
    
MyNetwork::~MyNetwork() {
    delete network;
    delete backprop;
}

pair<float, int> MyNetwork::TrainWithResult(const MemoryBlock& input, const MemoryBlock& output){
    network->Propagate(input);
    backprop->Train(output);
    return make_pair(backprop->error, backprop->best_pick);
}

pair<float, int> MyNetwork::JustResult(const MemoryBlock& input, const MemoryBlock& output){
    network->Propagate(input);
    float error = 0;
    float best_pick_prob = network->output.data[0];
    int best_pick = 0;
    for (int i = 0; i < output.size; i++) {
        if(best_pick_prob < network->output.data[i]){
            best_pick_prob = network->output.data[i];
            best_pick = i;
        }
        error += (output.data[i] - network->output.data[i]) * (output.data[i] - network->output.data[i]);
    }
    return make_pair(error, best_pick);
}