#ifndef FINAL_KMEANSPTHREAD_H
#define FINAL_KMEANSPTHREAD_H

#include "KMeans.h"
#include <pthread.h>
#include <semaphore.h>

#define PTHREAD_Serial 1
#define PTHREAD_SIMD 2

class KMeansPthread : public KMeans{

    typedef struct
    {
        int thread_id;
    } threadParam_t;

    int threadNum{};
    sem_t sem{};
    pthread_barrier_t barrier{};
    pthread_mutex_t lock{};
    int taskIndex = 0;
    int taskNum = 4;
    void calculate() override;
    void updateCentroids() override;
    void changeMemory() override;
    void* threadFuncDiv(void* param);
    void* threadFuncSIMD(void* param);
    float calculateDistanceSIMD(float* dataItem, float* centroidItem);

public:
    explicit KMeansPthread(int k, int mehtod = 0);
    ~KMeansPthread();
    void fit() override;
    void setThreadNum(int threadNumber);
    void initThread();
    void updateCentroidsSerial();
    void updateCentroidsSIMD();
};

#endif //FINAL_KMEANSPTHREAD_H
