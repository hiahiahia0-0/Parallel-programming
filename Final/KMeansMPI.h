#ifndef FINAL_KMEANSMPI_H
#define FINAL_KMEANSMPI_H

#include "KMeans.h"

#define DATA_COMM 111
#define CENTROID_COMM 222
#define LABEL_COMM 333

#define MPI_BLOCK 1
#define MPI_CYCLE 2
#define MPI_SIMD 3
#define MPI_OMP 4
#define MPI_OMP_SIMD 5

class KMeansMPI : public KMeans{
    float* dataMemory{};
    float* centroidMemory{};
    int tasks{};
    int rank{};
    static int size;
    int threadNum = 4;
    void calculate() override;
    void updateCentroids() override;
    float calculateDistance(const float *dataItem, const float *centroidItem) const override;
    float calculateDistanceSerial(const float *dataItem, const float *centroidItem) const;
    float calculateDistanceSIMD(const float *dataItem, const float *centroidItem) const;
    void changeMemory() override;
    void fitMPI_Block();
    void fitMPI_CYCLE();
    void fitMPI_SIMD();
    void fitMPI_OMP();
    void fitMPI_OMP_SIMD();
    void fitSerial();
    void calculateSerial();
    void calculateMultiThread();
    void calculateSingleThread();
    void updateCentroidsSerial();
    void updateCentroidsSIMD();
    void updateCentroidsOMP();
    void updateCentroidsOMP_SIMD();

public:
    explicit KMeansMPI(int k, int method = 0);
    ~KMeansMPI();
    void fit() override;
    void setThreadNum(int threadNumber);
};


#endif //FINAL_KMEANSMPI_H
