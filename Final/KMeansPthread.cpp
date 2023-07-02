#include "KMeansPthread.h"
#include <pmmintrin.h> 
#include <cstring>

KMeansPthread::KMeansPthread(int k, int mehtod) : KMeans(k, mehtod) {
}

KMeansPthread::~KMeansPthread() = default;

void KMeansPthread::initThread() {
    void *(*threadFunc)(void *);
    switch(method){
        case PTHREAD_Serial:
            threadFunc = threadFuncDiv;
            break;
        case PTHREAD_SIMD:
            threadFunc = threadFuncSIMD;
            break;
        default:
            threadFunc = nullptr;
            break;
    }
    sem_init(&sem, 0, 0);
    pthread_barrier_init(&barrier, nullptr, threadNum);
    pthread_mutex_init(&lock, nullptr);
    auto* threadParams = new threadParam_t[threadNum];
    auto* threads = new pthread_t[threadNum];
    for(int i = 0; i < threadNum; i++){
        threadParams[i].thread_id = i;
        pthread_create(&threads[i], nullptr, threadFunc, (void *)(&threadParams[i]));
    }
    for(int i = 0; i < threadNum; i++){
        pthread_join(threads[i], nullptr);
    }
    sem_destroy(&sem);
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&lock);
    delete[] threadParams;
    delete[] threads;
}


void *KMeansPthread::threadFuncDiv(void *param) {
    auto* threadParam = (threadParam_t*)param;
    int thread_id = threadParam->thread_id;
    for(int l = 0; l < this->L; l++){
        for (int i = thread_id; i < this->N; i += threadNum) {
            float min = 1e9;
            int minIndex = 0;
            for(int k=0;k<this->K;k++) {
                float dis = calculateDistance(this->data[i], this->centroids[k]);
                if(dis < min) {
                    min = dis;
                    minIndex = k;
                }
            }
            this->clusterLabels[i] = minIndex;
        }
        if(thread_id == 0){
            updateCentroids();
        }
        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}

void *KMeansPthread::threadFuncSIMD(void *param) {
    auto* threadParam = (threadParam_t*)param;
    int thread_id = threadParam->thread_id;
    for(int l = 0; l < this->L; l++){
        for (int i = thread_id; i < this->N; i += threadNum) {
            float min = 1e9;
            int minIndex = 0;
            for(int k=0;k<this->K;k++) {
                float dis = calculateDistanceSIMD(this->data[i], this->centroids[k]);
                if(dis < min) {
                    min = dis;
                    minIndex = k;
                }
            }
            this->clusterLabels[i] = minIndex;
        }
        if(thread_id == 0){
            updateCentroids();
        }
        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}

void KMeansPthread::calculate(){
}

float KMeansPthread::calculateDistanceSIMD(float* dataItem, float* centroidItem){
    float dis = 0;
    for(int i = 0; i < this->D - this->D % 4; i+=4) {
        __m128 tmpData, centroid;
        tmpData = _mm_loadu_ps(&dataItem[i]);
        centroid = _mm_loadu_ps(&centroidItem[i]);
        __m128 diff = _mm_sub_ps(tmpData, centroid);
        __m128 square = _mm_mul_ps(diff, diff);
        __m128 sum = _mm_hadd_ps(square, square);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        dis += _mm_cvtss_f32(sum);
    }
    for(int i = this->D - this->D % 4; i < this->D; i++) {
        dis += (dataItem[i] - centroidItem[i]) * (dataItem[i] - centroidItem[i]);
    }
    return dis;
}

void KMeansPthread::updateCentroids(){
    switch(this->method){
        case PTHREAD_Serial:
            updateCentroidsSerial();
            break;
        case PTHREAD_SIMD:
            updateCentroidsSIMD();
            break;
    }
}

void KMeansPthread::updateCentroidsSerial() {
    memset(this->clusterCount, 0, sizeof(int) * K);
    for(int i=0;i<this->N;i++){
        int cluster = this->clusterLabels[i];
        for(int j=0;j<this->D;j++){
            this->centroids[cluster][j] += this->data[i][j];
        }
        this->clusterCount[cluster]++;
    }

    for(int i=0;i<this->K;i++){
        for(int j=0;j<this->D;j++){
            this->centroids[i][j] /= (float)this->clusterCount[i];
        }
    }
}

void KMeansPthread::updateCentroidsSIMD() {
    memset(this->clusterCount, 0, sizeof(int) * K);
    for(int i=0;i<this->N;i++){
        int cluster = this->clusterLabels[i];
        this->clusterCount[cluster]++;
        for(int j=0;j<this->D - this->D % 4;j+=4){
            __m128 tmpData = _mm_loadu_ps(&this->data[i][j]);
            __m128 centroid = _mm_loadu_ps(&this->centroids[cluster][j]);
            __m128 sum = _mm_add_ps(tmpData, centroid);
            _mm_storeu_ps(&this->centroids[cluster][j], sum);
        }
        for(int j=this->D - this->D % 4;j<this->D;j++){
            this->centroids[cluster][j] += this->data[i][j];
        }
    }

    for(int i=0;i<this->K;i++){
        for(int j=0;j<this->D - this->D % 4;j+=4){
            __m128 tmpData = _mm_loadu_ps(&this->centroids[i][j]);
            __m128 count = _mm_loadu_ps(reinterpret_cast<const float *>(&this->clusterCount[i]));
            __m128 mean = _mm_div_ps(tmpData, count);
            _mm_storeu_ps(&this->centroids[i][j], mean);
        }
        for(int j=this->D - this->D % 4;j<this->D;j++){
            this->centroids[i][j] /= this->clusterCount[i];
        }
    }
}

void KMeansPthread::setThreadNum(int threadNumber) {
    this->threadNum = threadNumber;
}

void KMeansPthread::changeMemory(){
}

void KMeansPthread::fit(){
    initCentroidsRandom();
    initThread();
}