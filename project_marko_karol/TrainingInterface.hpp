//
//  TrainingInterface.hpp
//  pv021-project
//
//  Created by Jozef Marko on 18/12/15.
//  Copyright © 2015 Jozef Marko. All rights reserved.
//

#ifndef TrainingInterface_hpp
#define TrainingInterface_hpp

#include <stdio.h>
#include "MemoryBlock.h"

class TrainingInterface {
public:
    virtual float Train(const MemoryBlock& input, int gibbsSamplingSteps) = 0;

};

#endif /* TrainingInterface_hpp */
