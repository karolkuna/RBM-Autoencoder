#ifndef AutoEncoder_hpp
#define AutoEncoder_hpp

#include "NeuralPredictionLibrary/FeedforwardNetwork.h"
#include "NeuralPredictionLibrary/Backpropagation.h"

class MyNetwork {
public:
    FeedforwardNetwork* network;
    Backpropagation* backprop;
    
    MyNetwork(int visibleUnits, int hiddenUnits, int outputUnits, ActivationFunction* activationFunction, float learningRate, float momentumRate);
    ~MyNetwork();
    
    pair<float,int> TrainWithResult(const MemoryBlock& input, const MemoryBlock& output);
    pair<float,int> JustResult(const MemoryBlock& input, const MemoryBlock& output);
};

#endif /* AutoEncoder_hpp */
