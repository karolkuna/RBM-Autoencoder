#include <iostream>
#include "AutoEncoder.hpp"
#include "RBM.hpp"

int main(int argc, const char * argv[]) {
    MemoryBlock input(2);
    input.data[0] = 0;
    input.data[1] = 1;
    
    LogisticFunction logistic;
    
    RBM rbm(2, 3, 0.1f, 0.0f);
    AutoEncoder autoEncoder(2, 3, &logistic, 0.1f, 0.0f);
    
    for (int step = 0; step < 100; step++) {
        printf("RBM error: %f\n", rbm.Train(input, 1));
        printf("AutoEncoder error: %f\n\n", autoEncoder.Train(input));
    }
    
    MemoryBlock features(3);
    MemoryBlock reconstruction(2);
    
    
    printf("\nRBM features:\n");
    rbm.Encode(input, features);
    features.Print();
    printf("RBM reconstruction:\n");
    rbm.Decode(features, reconstruction);
    reconstruction.Print();
    
    printf("\nAutoEncoder features:\n");
    autoEncoder.Encode(input, features);
    features.Print();
    printf("AutoEncoder reconstruction:\n");
    autoEncoder.Decode(features, reconstruction);
    reconstruction.Print();
    
    return 0;
}
