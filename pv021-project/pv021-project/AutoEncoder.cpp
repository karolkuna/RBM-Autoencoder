#include "AutoEncoder.hpp"

AutoEncoder::AutoEncoder(int visibleUnits, int hiddenUnits, ActivationFunction* activationFunction, float learningRate, float momentumRate) {
        
    network = new FeedforwardNetwork(visibleUnits, {hiddenUnits}, visibleUnits, activationFunction, learningRate, momentumRate);
        backprop = new Backpropagation(network);
}
    
AutoEncoder::~AutoEncoder() {
    delete network;
    delete backprop;
}
    
float AutoEncoder::Train(const MemoryBlock& input) {
    network->Propagate(input);
    backprop->Train(input);
    return backprop->error;
}
    
void AutoEncoder::Encode(const MemoryBlock& input, MemoryBlock& features) {
    network->layers[0]->SetActivation(input);
    network->layers[1]->PropagateForward();
    network->layers[1]->activation.CopyTo(features);
}
    
void AutoEncoder::Decode(const MemoryBlock& features, MemoryBlock& reconstruction) {
    network->layers[1]->SetActivation(features);
    network->layers[2]->PropagateForward();
    network->layers[2]->activation.CopyTo(reconstruction);
}