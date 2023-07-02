#include <pmmintrin.h> 
#include <omp.h>
#include <cstring>
#include "KMeansOMP.h"

KMeansOMP::KMeansOMP(int k, int method) : KMeans(k, method) {
}

KMeansOMP::~KMeansOMP()= default;

void KMeansOMP::calculate() {
    switch(method){
        case OMP:
            calculateOMP();
            break;
        case OMP_SIMD:
            calculateOMPSIMD();
            break;
        default:
            calculateSerial();
            break;
    }
}

void KMeansOMP::calculateSerial(){
    for(int i=0;i<this->N;i++){
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
}

void KMeansOMP::calculateOMP() {
    int i, k;
    float min = 1e9;
    int minIndex = 0;
    float dis;
#pragma omp parallel num_threads(threadNum) default(none) \
    private(i, k, min, minIndex, dis) \
    shared(this->data, this->clusterLabels, this->centroids, this->N, this->K)
    {
#pragma omp for schedule(static)
        {
            for (i = 0; i < this->N; i++) {
                for (k = 0; k < this->K; k++) {
                    dis = calculateDistance(this->data[i], this->centroids[k]);
                    if (dis < min) {
                        min = dis;
                        minIndex = k;
                    }
                }
                this->clusterLabels[i] = minIndex;
            }
        }
    }
}

void KMeansOMP::calculateOMPSIMD() {
    int i, k;
    float min = 1e9;
    int minIndex = 0;
    float dis;
#pragma omp parallel num_threads(threadNum) default(none) \
    private(i, k, min, minIndex, dis) \
    shared(this->data, this->clusterLabels, this->centroids, this->N, this->K)
    {
#pragma omp for schedule(static)
        {
            for (i = 0; i < this->N; i++) {
                for (k = 0; k < this->K; k++) {
                    dis = calculateDistanceSIMD(this->data[i], this->centroids[k]);
                    if (dis < min) {
                        min = dis;
                        minIndex = k;
                    }
                }
                this->clusterLabels[i] = minIndex;
            }
        }
    }
}

float KMeansOMP::calculateDistanceSIMD(float *dataItem, float *centroidItem) {
    float dis = 0;
    for(int i = 0; i < this->D - this->D % 4; i+=4) {
        __m128 data, centroid;
        data = _mm_load_ps(&dataItem[i]);
        centroid = _mm_load_ps(&centroidItem[i]);
        __m128 diff = _mm_sub_ps(data, centroid);
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


void KMeansOMP::updateCentroids() {
    switch(method){
        case OMP:
            updateCentroidsOMP();
            break;
        case OMP_SIMD:
            updateCentroidsOMPSIMD();
            break;
        default:
            updateCentroidsSerial();
            break;
    }
}

void KMeansOMP::updateCentroidsSerial() {
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

void KMeansOMP::updateCentroidsOMP() {
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i,j,cluster;
#pragma omp parallel default(none) num_threads(threadNum) \
    private(i, j, cluster) \
    shared(this->data, this->clusterLabels, this->clusterLabels, this->clusterCount, this->N, this->K)
    {
#pragma omp for schedule(static)
        {
            for (i = 0; i < this->N; i++) {
                cluster = this->clusterLabels[i];
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[cluster][j] += this->data[i][j];
                    }
                }
                this->clusterCount[cluster]++;
            }
        }
#pragma omp for schedule(static)
        {
            for (i = 0; i < this->K; i++) {
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[i][j] /= (float) this->clusterCount[i];
                    }
                }
            }
        }
    }
}

void KMeansOMP::updateCentroidsOMPSIMD() {
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i,j,cluster;
    __m128 tmpData, centroid, sum;
#pragma omp parallel default(none) num_threads(threadNum) \
    private(i, j, cluster, tmpData, centroid, sum) \
    shared(this->data, this->clusterLabels, this->clusterLabels, this->clusterCount, this->N, this->K)
    {
#pragma omp for schedule(guided)
        {
            for (i = 0; i < this->N; i++) {
                cluster = this->clusterLabels[i];
                this->clusterCount[cluster]++;
                for(j=0; j<this->D - this->D % 4;j+=4){
                    tmpData = _mm_loadu_ps(&this->data[i][j]);
                    centroid = _mm_loadu_ps(&this->centroids[cluster][j]);
                    sum = _mm_add_ps(tmpData, centroid);
                    _mm_storeu_ps(&this->centroids[cluster][j], sum);
                }
                for(j=this->D - this->D % 4;j<this->D;j++){
                    this->centroids[cluster][j] += this->data[i][j];
                }
            }
        }
#pragma omp for schedule(guided)
        {
            for (i = 0; i < this->K; i++) {
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[i][j] /= (float) this->clusterCount[i];
                    }
                }
            }
        }
    }
}

void KMeansOMP::fit() {
    initCentroidsRandom();
    for (int i = 0; i < this->L; i++) {
        calculate();
        updateCentroids();
    }
}

void KMeansOMP::changeMemory() {}

void KMeansOMP::setThreadNum(int threadNumber){
    this->threadNum = threadNumber;
}
