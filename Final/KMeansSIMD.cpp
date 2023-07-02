#include <xmmintrin.h> 
#include <pmmintrin.h> 
#include <cstring>
#include "KMeansSIMD.h"

KMeansSIMD::KMeansSIMD(int k, int method) : KMeans(k, method) {
}

KMeansSIMD::~KMeansSIMD(){
    data = new float*[this->N];
    for(int i = 0; i < this->N; i++)
        data[i] = new float[this->D];
    centroids = new float*[this->K];
    for(int i = 0; i < this->K; i++)
        centroids[i] = new float[this->D];
    clusterCount = new int[this->K];
}

void KMeansSIMD::fit() {
    initCentroidsRandom();
    for (int i = 0; i < this->L; i++) {
        calculate();
        updateCentroids();
    }
}

void KMeansSIMD::calculate() {
    switch (method) {
        case SIMD_UNALIGNED:
        case SIMD_ALIGNED:
            calculateSIMD();
            break;
        default:
            break;
    }
}

void KMeansSIMD::calculateSIMD() {
    for (int i = 0; i < this->N; i++) {
        float min = 1e9;
        int minIndex = 0;
        for (int k = 0; k < this->K; k++) {
            float dis = calculateDistanceSIMD(this->data[i], this->centroids[k]);
            if (dis < min) {
                min = dis;
                minIndex = k;
            }
        }
        this->clusterLabels[i] = minIndex;
    }
}

float KMeansSIMD::calculateDistanceSIMD(float *dataItem, float *centroidItem) {
    float dis = 0;
    for(int i = 0; i < this->D - this->D % 4; i+=4) {
        __m128 tmpData, centroid;
        if(this->method == SIMD_UNALIGNED){
            tmpData = _mm_loadu_ps(&dataItem[i]);
            centroid = _mm_loadu_ps(&centroidItem[i]);
        }
        else{
            tmpData = _mm_load_ps(&dataItem[i]);
            centroid = _mm_load_ps(&centroidItem[i]);
        }
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


void KMeansSIMD::updateCentroids() {
    switch (method) {
        case SIMD_UNALIGNED:
        case SIMD_ALIGNED:
            updateCentroidsSIMD();
            break;
        default:
            break;
    }
}

void KMeansSIMD::updateCentroidsSIMD() {
    memset(this->clusterCount, 0, sizeof(int) * K);
    for(int i=0;i<this->N;i++){
        int cluster = this->clusterLabels[i];
        this->clusterCount[cluster]++;
        if(method == SIMD_UNALIGNED){
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
        else{
            for(int j=0;j<this->D - this->D % 4;j+=4){
                __m128 tmpData = _mm_load_ps(&this->data[i][j]);
                __m128 centroid = _mm_load_ps(&this->centroids[cluster][j]);
                __m128 sum = _mm_add_ps(tmpData, centroid);
                _mm_store_ps(&this->centroids[cluster][j], sum);
            }
            for(int j=this->D - this->D % 4;j<this->D;j++){
                this->centroids[cluster][j] += this->data[i][j];
            }
        }
    }

    for(int i=0;i<this->K;i++){
        if(method == SIMD_UNALIGNED){
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
        else{
            for(int j=0;j<this->D - this->D % 4;j+=4){
                __m128 tmpData = _mm_load_ps(&this->centroids[i][j]);
                __m128 count = _mm_loadu_ps(reinterpret_cast<const float *>(&this->clusterCount[i]));
                __m128 mean = _mm_div_ps(tmpData, count);
                _mm_store_ps(&this->centroids[i][j], mean);
            }
            for(int j=this->D - this->D % 4;j<this->D;j++){
                this->centroids[i][j] /= this->clusterCount[i];
            }
        }
    }
}

void KMeansSIMD::changeMemory() {
    if (method % 2 == 0) {
        auto **newData = (float **) malloc(sizeof(float *) * this->N);
        for (int i = 0; i < this->N; i++) {
            newData[i] = (float *) _aligned_malloc(sizeof(float) * this->D, 64);
        }
        for (int i = 0; i < this->N; i++) {
            for (int j = 0; j < this->D; j++) {
                newData[i][j] = data[i][j];
            }
        }
        for (int i = 0; i < this->N; i++) {
            delete[] data[i];
        }
        delete[] data;
        data = newData;
        centroids = (float **) malloc(sizeof(float *) * this->K);
        for(int i = 0; i < this->K; i++) {
            centroids[i] = (float *) _aligned_malloc(sizeof(float) * this->D, 64);
        }
        clusterCount = (int *) _aligned_malloc(sizeof(int) * this->K, 64);
    }
}
