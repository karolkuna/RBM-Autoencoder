#ifndef AutoEncoder_hpp
#define AutoEncoder_hpp

#include "NeuralPredictionLibrary/FeedforwardNetwork.h"
#include "NeuralPredictionLibrary/Backpropagation.h"
#include "TrainingInterface.hpp"

class AutoEncoder : public TrainingInterface{
public:
    FeedforwardNetwork* network;
    Backpropagation* backprop;
    
    AutoEncoder(int visibleUnits, int hiddenUnits, ActivationFunction* activationFunction, float learningRate, float momentumRate);
    ~AutoEncoder();
    
    float Train(const MemoryBlock& input);
    float Train(const MemoryBlock& input, int steps);
    void Encode(const MemoryBlock& input, MemoryBlock& features);
    void Decode(const MemoryBlock& features, MemoryBlock& reconstruction);
};

#endif /* AutoEncoder_hpp */