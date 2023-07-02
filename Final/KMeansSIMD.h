#ifndef FINAL_KMEANSSIMD_H
#define FINAL_KMEANSSIMD_H

#include "KMeans.h"

#define SIMD_UNALIGNED 1
#define SIMD_ALIGNED 2

class KMeansSIMD : public KMeans{
    void calculate() override;
    void updateCentroids() override;
    void calculateSIMD();
    void calculateAVX();
    void calculateAVX512();
    float calculateDistanceSIMD(float *dataItem, float *centroidItem);
    void updateCentroidsSIMD();
    void changeMemory() override;
public:
    explicit KMeansSIMD(int k, int method = 0);
    void fit() override;
    ~KMeansSIMD();
};

#endif 
