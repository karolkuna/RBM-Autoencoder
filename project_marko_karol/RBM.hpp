#ifndef RBM_hpp
#define RBM_hpp

#include "NeuralPredictionLibrary/MemoryBlock.h"
#include "NeuralPredictionLibrary/ActivationFunctions.h"
#include "TrainingInterface.hpp"

class RBM : public TrainingInterface{
    int visibleUnits, hiddenUnits;
    MemoryBlock visibleStates;
    MemoryBlock hiddenStates;
    MemoryBlock weights;
    MemoryBlock weightDeltas;
    MemoryBlock positiveGradient;
    MemoryBlock negativeGradient;
    
    LogisticFunction logistic;

    std::random_device generator;
    std::uniform_real_distribution<float> uniform;
    
    void ActivateHidden(bool sampleBinaryState);
    void ActivateVisible(bool sampleBinaryState);
    void ComputeGradient(MemoryBlock& gradient);
    void UpdateWeights();
    
public:
    float learningRate, momentum;
    
    RBM(int visibleUnits, int hiddenUnits, float learningRate, float momentum);
    
    float Train(const MemoryBlock& input, int gibbsSamplingSteps);
    void Encode(const MemoryBlock& input, MemoryBlock& features);
    void Decode(const MemoryBlock& features, MemoryBlock& reconstruction);
};


#endif /* RBM_hpp */
