#ifndef FINAL_KMEANSOMP_H
#define FINAL_KMEANSOMP_H

#include "KMeans.h"

#define OMP 1
#define OMP_SIMD 2

class KMeansOMP : public KMeans{
    int threadNum{};
    void calculate() override;
    void updateCentroids() override;
    void changeMemory() override;
    void calculateSerial();
    void calculateOMP();
    void calculateOMPSIMD();
    float calculateDistanceSIMD(float *dataItem, float *centroidItem);
    void updateCentroidsSerial();
    void updateCentroidsOMP();
    void updateCentroidsOMPSIMD();

public:
    explicit KMeansOMP(int k, int method = 0);
    ~KMeansOMP();
    void fit() override;
    void setThreadNum(int threadNumber);

};

#endif //FINAL_KMEANSOMP_H
