#include "KMeans.h"
#include <iostream>
#include <cstring>
#include <random>

using namespace std;

KMeans::KMeans(int k) : KMeans(k, 0){
}

KMeans::KMeans(int k, int method) {
    this->K = k;
    this->clusterCount = new int[k];
    this->method = method;
    memset(this->clusterCount, 0, sizeof(int) * k);
}

void KMeans::printResult() {
    for(int i = 0; i < this->K; i++)
        cout << "Cluster " << i << ": " << this->clusterCount[i] << " points" << endl;
}

int KMeans::getClusterNumber() const{
    return this->K;
}

KMeans::~KMeans() {
    if(data!=nullptr){
        for(int i = 0; i < this->N; i++)
            delete[] data[i];
        delete[] data;
    }
    if(centroids!=nullptr){
        for(int i = 0; i < this->K; i++)
            delete[] centroids[i];
        delete[] centroids;
    }
    if(clusterLabels!=nullptr){
        delete[] clusterLabels;
    }
    if(clusterCount!=nullptr){
        delete[] clusterCount;
    }
}

void KMeans::initData(float** inputData, int n, int d) {
    this->N = n;
    this->D = d;
    this->data = new float*[this->N];
    for(int i = 0; i < this->N; i++)
    {
        this->data[i] = new float[this->D];
        for(int j = 0; j < this->D; j++)
            this->data[i][j] = inputData[i][j];
    }
    this->clusterLabels = new int[this->N];
    changeMemory();
}

void KMeans::initCentroidsRandom() {
    this->centroids = new float*[this->K];
    default_random_engine e;
    uniform_int_distribution<unsigned> u(0, this->N - 1);
    for(int i = 0; i < this->K; i++)
    {
        this->centroids[i] = new float[this->D];
        int index = u(e) % this->N;
        for(int j = 0; j < this->D; j++)
            this->centroids[i][j] = this->data[index][j];
    }
}

float KMeans::calculateDistance(const float *dataItem, const float *centroid) const {
    float dis = 0;
    for(int i = 0; i < this->D; i++)
        dis += (dataItem[i] - centroid[i]) * (dataItem[i] - centroid[i]);
    return dis;
}
