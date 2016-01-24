#include "RBM.hpp"

void RBM::ActivateHidden(bool sampleBinaryState) {
    for (int to = 0; to < hiddenUnits; to++) {
        float weightedSum = 0.0f;
        for (int from = 0; from < visibleStates.size; from++) {
            int weightId = to * visibleStates.size + from;
            weightedSum += visibleStates.data[from] * weights.data[weightId];
        }
        
        if (sampleBinaryState)
            hiddenStates.data[to] = logistic(weightedSum) > uniform(generator) ? 1.0f : 0.0f;
        else
            hiddenStates.data[to] = logistic(weightedSum);
    }
}

void RBM::ActivateVisible(bool sampleBinaryState) {
    for (int to = 0; to < visibleUnits; to++) {
        float weightedSum = 0.0f;
        for (int from = 0; from < hiddenStates.size; from++) {
            int weightId = from * visibleStates.size + to;
            weightedSum += hiddenStates.data[from] * weights.data[weightId];
        }
        
        if (sampleBinaryState)
            visibleStates.data[to] = logistic(weightedSum) > uniform(generator) ? 1.0f : 0.0f;
        else
            visibleStates.data[to] = logistic(weightedSum);
    }
}

void RBM::ComputeGradient(MemoryBlock& gradient) {
    for (int to = 0; to < hiddenStates.size; to++) {
        for (int from = 0; from < visibleStates.size; from++) {
            gradient.data[to * visibleStates.size + from] = visibleStates.data[from] * hiddenStates.data[to];
        }
    }
}

void RBM::UpdateWeights() {
    for (int w = 0; w < weights.size; w++) {
        weightDeltas.data[w] = learningRate * (positiveGradient.data[w] - negativeGradient.data[w]) + momentum * weightDeltas.data[w];
        weights.data[w] += weightDeltas.data[w];
    }
}

RBM::RBM(int visibleUnits, int hiddenUnits, float learningRate, float momentum) {
    this->visibleUnits = visibleUnits;
    this->hiddenUnits = hiddenUnits;
    this->learningRate = learningRate;
    this->momentum = momentum;
    
    visibleStates = MemoryBlock(visibleUnits + 1); // + 1 because of bias unit
    hiddenStates = MemoryBlock(hiddenUnits + 1);
    
    // set bias unit state to 1
    visibleStates.data[visibleUnits] = 1;
    hiddenStates.data[visibleUnits] = 1;
    
    weights = MemoryBlock(visibleStates.size * hiddenStates.size);
    weights.GenerateNormal(0.0f, 0.01f);
    
    weightDeltas = MemoryBlock(weights.size);
    weightDeltas.Fill(0.0f);
    positiveGradient = MemoryBlock(weights.size);
    negativeGradient = MemoryBlock(weights.size);
    
    uniform = std::uniform_real_distribution<float>(0.0f, 1.0f);
}

// input should be binary
float RBM::Train(const MemoryBlock& input, int gibbsSamplingSteps) {
    input.CopyTo(visibleStates, 0, 0, visibleUnits);
    
    ActivateHidden(true);
    ComputeGradient(positiveGradient);
    
    for (int k = 0; k < gibbsSamplingSteps - 1; k++) {
        ActivateVisible(false);
        ActivateHidden(true);
    }
    
    ActivateVisible(false);
    ActivateHidden(false);
    ComputeGradient(negativeGradient);
    
    UpdateWeights();
    
    float error = 0.0f;
    for (int i = 0; i < visibleUnits; i++) {
        error += (visibleStates.data[i] - input.data[i]) * (visibleStates.data[i] - input.data[i]);
    }
    
    return error;
}


void RBM::Encode(const MemoryBlock& input, MemoryBlock& features) {
    input.CopyTo(visibleStates, 0, 0, visibleUnits);
    ActivateHidden(false);
    hiddenStates.CopyTo(features, 0, 0, hiddenUnits);
}

void RBM::Decode(const MemoryBlock& features, MemoryBlock& reconstruction) {
    features.CopyTo(hiddenStates, 0, 0, hiddenUnits);
    ActivateVisible(false);
    visibleStates.CopyTo(reconstruction, 0, 0, visibleUnits);
}
