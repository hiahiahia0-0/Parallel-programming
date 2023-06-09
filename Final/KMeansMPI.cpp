#include <mpi.h>
#include <cmath>
#include <pmmintrin.h>
#include <cstring>
#include "KMeansMPI.h"

KMeansMPI::KMeansMPI(int k, int method) : KMeans(k, method) {
}

KMeansMPI::~KMeansMPI() {
    delete[] dataMemory;
    delete[] centroidMemory;
    data = new float *[this->N];
    for (int i = 0; i < this->N; i++) {
        data[i] = new float[this->D];
    }
    centroids = new float *[this->K];
    for (int i = 0; i < this->K; i++) {
        centroids[i] = new float[this->D];
    }
}

void KMeansMPI::fitMPI_Block() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i == size - 1 ? this->N % tasks : tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (this->rank != 0) {
            calculate();
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
        else {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = (source - 1) * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

void KMeansMPI::fitMPI_CYCLE() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i == size - 1 ? this->N % tasks : tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (this->rank != 0) {
            calculate();
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
        else {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = (source - 1) * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

void KMeansMPI::fitMPI_SIMD() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i != this->N / tasks ? tasks : this->N % tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (this->rank != 0) {
            calculate();
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
        else {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = (source - 1) * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

void KMeansMPI::fitMPI_OMP() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i != this->N / tasks ? tasks : this->N % tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (this->rank != 0) {
            calculate();
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
        else {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = (source - 1) * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

void KMeansMPI::fitMPI_OMP_SIMD() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i != this->N / tasks ? tasks : this->N % tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (this->rank != 0) {
            calculate();
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
        else {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = (source - 1) * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

void KMeansMPI::fitSerial() {
    initCentroidsRandom();
    for (int i = 0; i < this->L; i++) {
        calculate();
        updateCentroids();
    }
}

void KMeansMPI::calculate() {
    switch (this->method) {
        case MPI_OMP:
        case MPI_OMP_SIMD:
            calculateMultiThread();
            break;
        case MPI_BLOCK:
        case MPI_CYCLE:
        case MPI_SIMD:
            calculateSingleThread();
            break;
        default:
            calculateSerial();
            break;
    }
}

void KMeansMPI::calculateSerial() {
    for (int i = 0; i < this->N; i++) {
        float min = 1e9;
        int minIndex = 0;
        for (int k = 0; k < this->K; k++) {
            float dis = calculateDistance(this->data[i], this->centroids[k]);
            if (dis < min) {
                min = dis;
                minIndex = k;
            }
        }
        this->clusterLabels[i] = minIndex;
    }
}

void KMeansMPI::calculateMultiThread() {
    int begin = this->rank * this->tasks;
    int end = this->rank == size - 1 ? this->N : begin + this->tasks;
    int i, k;
    float min = 1e9;
    int minIndex = 0;
    float dis;
#pragma omp parallel num_threads(this->threadNum) default(none) \
    private(i, k, min, minIndex, dis) \
    shared(this->data, this->clusterLabels, this->centroids, this->N, this->K, begin, end)
    {
#pragma omp for schedule(static)
        {
            for (i = begin; i < end; i++) {
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

void KMeansMPI::calculateSingleThread() {
    int begin = (this->rank - 1) * this->tasks;
    int end = this->rank == size - 1 ? this->N : begin + this->tasks;
    for (int i = begin; i < end; i++) {
        float min = 1e9;
        int minIndex = 0;
        for (int k = 0; k < this->K; k++) {
            float dis = calculateDistance(this->data[i], this->centroids[k]);
            if (dis < min) {
                min = dis;
                minIndex = k;
            }
        }
        this->clusterLabels[i] = minIndex;
    }
}


float KMeansMPI::calculateDistance(const float *dataItem, const float *centroidItem) const {
    switch (this->method) {
        case MPI_BLOCK:
        case MPI_CYCLE:
        case MPI_OMP:
        default:
            return calculateDistanceSerial(dataItem, centroidItem);
        case MPI_SIMD:
        case MPI_OMP_SIMD:
            return calculateDistanceSIMD(dataItem, centroidItem);
    }
}

float KMeansMPI::calculateDistanceSerial(const float *dataItem, const float *centroidItem) const {
    float dis = 0;
    for (int i = 0; i < this->D; i++)
        dis += (dataItem[i] - centroidItem[i]) * (dataItem[i] - centroidItem[i]);
    return dis;
}

float KMeansMPI::calculateDistanceSIMD(const float *dataItem, const float *centroidItem) const {
    float dis = 0;
    for (int i = 0; i < this->D - this->D % 4; i += 4) {
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
    for (int i = this->D - this->D % 4; i < this->D; i++) {
        dis += (dataItem[i] - centroidItem[i]) * (dataItem[i] - centroidItem[i]);
    }
    return dis;
}

void KMeansMPI::updateCentroids() {
    switch (this->method) {
        case MPI_BLOCK:
        case MPI_CYCLE:
        default:
            updateCentroidsSerial();
            break;
        case MPI_SIMD:
            updateCentroidsSIMD();
            break;
        case MPI_OMP:
            updateCentroidsOMP();
            break;
        case MPI_OMP_SIMD:
            updateCentroidsOMP_SIMD();
            break;
    }
}

void KMeansMPI::updateCentroidsSerial() {
    memset(this->clusterCount, 0, sizeof(int) * K);
    for (int i = 0; i < this->N; i++) {
        int cluster = this->clusterLabels[i];
        for (int j = 0; j < this->D; j++) {
            this->centroids[cluster][j] += this->data[i][j];
        }
        this->clusterCount[cluster]++;
    }r
    for (int i = 0; i < this->K; i++) {
        for (int j = 0; j < this->D; j++) {
            this->centroids[i][j] /= (float) this->clusterCount[i];
        }
    }
}

void KMeansMPI::updateCentroidsSIMD() {
    memset(this->clusterCount, 0, sizeof(int) * K);
    for (int i = 0; i < this->N; i++) {
        int cluster = this->clusterLabels[i];
        this->clusterCount[cluster]++;
        for (int j = 0; j < this->D - this->D % 4; j += 4) {
            __m128 tmpData = _mm_loadu_ps(&this->data[i][j]);
            __m128 centroid = _mm_loadu_ps(&this->centroids[cluster][j]);
            __m128 sum = _mm_add_ps(tmpData, centroid);
            _mm_storeu_ps(&this->centroids[cluster][j], sum);
        }
        for (int j = this->D - this->D % 4; j < this->D; j++) {
            this->centroids[cluster][j] += this->data[i][j];
        }
    }
    for (int i = 0; i < this->K; i++) {
        for (int j = 0; j < this->D - this->D % 4; j += 4) {
            __m128 tmpData = _mm_loadu_ps(&this->centroids[i][j]);
            __m128 count = _mm_loadu_ps(reinterpret_cast<const float *>(&this->clusterCount[i]));
            __m128 mean = _mm_div_ps(tmpData, count);
            _mm_storeu_ps(&this->centroids[i][j], mean);
        }
        for (int j = this->D - this->D % 4; j < this->D; j++) {
            this->centroids[i][j] /= (float) this->clusterCount[i];
        }
    }
}

void KMeansMPI::updateCentroidsOMP() {
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i, j, cluster;
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

void KMeansMPI::updateCentroidsOMP_SIMD() {
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i, j, cluster;
    __m128 tmpData, centroid, sum, count, mean;
#pragma omp parallel default(none) num_threads(threadNum) \
    private(i, j, cluster, tmpData, centroid, sum, count, mean) \
    shared(this->data, this->clusterLabels, this->clusterLabels, this->clusterCount, this->N, this->K)
    {
#pragma omp for schedule(guided)
        {
            for (i = 0; i < this->N; i++) {
                cluster = this->clusterLabels[i];
                this->clusterCount[cluster]++;
                for (j = 0; j < this->D - this->D % 4; j += 4) {
                    tmpData = _mm_loadu_ps(&this->data[i][j]);
                    centroid = _mm_loadu_ps(&this->centroids[cluster][j]);
                    sum = _mm_add_ps(tmpData, centroid);
                    _mm_storeu_ps(&this->centroids[cluster][j], sum);
                }
                for (j = this->D - this->D % 4; j < this->D; j++) {
                    this->centroids[cluster][j] += this->data[i][j];
                }
            }
        }
#pragma omp for schedule(guided)
        {
            for (i = 0; i < this->K; i++) {
                for (j = 0; j < this->D - this->D % 4; j += 4) {
                    tmpData = _mm_loadu_ps(&this->centroids[i][j]);
                    count = _mm_loadu_ps(reinterpret_cast<const float *>(&this->clusterCount[i]));
                    mean = _mm_div_ps(tmpData, count);
                    _mm_storeu_ps(&this->centroids[i][j], mean);
                }
                for (j = this->D - this->D % 4; j < this->D; j++) {
                    this->centroids[i][j] /=(float) this->clusterCount[i];
                }
            }
        }
    }
}


void KMeansMPI::setThreadNum(int threadNumber) {
    this->threadNum = threadNumber;
}

void KMeansMPI::changeMemory() {
    dataMemory = new float[this->N * this->D];
    for (int i = 0; i < this->N; i++) {
        for (int j = 0; j < this->D; j++) {
            dataMemory[i * this->D + j] = this->data[i][j];
        }
    }
    for (int i = 0; i < this->N; i++) {
        delete[] data[i];
    }
    delete[] data;
    for (int i = 0; i < this->N; i++) {
        data[i] = &dataMemory[i * this->D];
    }
    centroidMemory = new float[this->K * this->D];
    for (int i = 0; i < this->K; i++) {
        for (int j = 0; j < this->D; j++) {
            centroidMemory[i * this->D + j] = this->centroids[i][j];
        }
    }
    for (int i = 0; i < this->K; i++) {
        delete[] centroids[i];
    }
    delete[] centroids;
    for (int i = 0; i < this->K; i++) {
        centroids[i] = &centroidMemory[i * this->D];
    }
}

void KMeansMPI::fit() {
    switch (method) {
        case MPI_BLOCK:
            fitMPI_Block();
            break;
        case MPI_CYCLE:
            fitMPI_CYCLE();
            break;
        case MPI_SIMD:
            fitMPI_SIMD();
            break;
        case MPI_OMP:
            fitMPI_OMP();
            break;
        case MPI_OMP_SIMD:
            fitMPI_OMP_SIMD();
            break;
        default:
            fitSerial();
            break;
    }
}
