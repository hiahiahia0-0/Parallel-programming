#include <iostream>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include <xmmintrin.h> 
#include <pthread.h>  
#include <semaphore.h>
#include <omp.h>

using namespace std;

int NUM_THREADS = 8;
int N;
const int L = 100;
const int LOOP = 10;
float **data;
float **matrix = NULL;

void init_data()
{
    data = new float *[N], matrix = new float *[N];
    for (int i = 0; i < N; i++)
        data[i] = new float[N], matrix[i] = new float[N];
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
            data[i][j] = rand() * 1.0 / RAND_MAX * L;
    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++)
            for (int k = 0; k < N; k++)
                data[j][k] += data[i][k];
}

void init_matrix()
{
    if(matrix!=NULL)
        for (int i = 0; i < N; i++)
            delete[] matrix[i];
    delete[] matrix;
    matrix = new float *[N];
    for (int i = 0; i < N; i++)
        matrix[i] = new float[N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = data[i][j];
}

void calculate_serial()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

void calculate_SIMD()
{
    for (int k = 0; k < N; k++)
    {
        __m128 Akk = _mm_set_ps1(matrix[k][k]);
        int j;
        for (j = k + 1; j + 3 < N; j += 4)
        {
            __m128 Akj = _mm_loadu_ps(matrix[k] + j);
            Akj = _mm_div_ps(Akj, Akk);
            _mm_storeu_ps(matrix[k] + j, Akj);
        }
        for (; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            __m128 Aik = _mm_set_ps1(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4)
            {
                __m128 Akj = _mm_loadu_ps(matrix[k] + j);
                __m128 Aij = _mm_loadu_ps(matrix[i] + j);
                __m128 AikMulAkj = _mm_mul_ps(Aik, Akj);
                Aij = _mm_sub_ps(Aij, AikMulAkj);
                _mm_storeu_ps(matrix[i] + j, Aij);
            }
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

void calculate_openmp_single_SIMD()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(1) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

void calculate_openmp_SIMD()
{
    int i, j, k;
    float tmp;
    __m128 Akk, Akj, Aik, Aij, AikMulAkj;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp, Akk, Akj, Aik, Aij, AikMulAkj) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
        Akk = _mm_set_ps1(matrix[k][k]);
        int j;
        tmp = matrix[k][k];
#pragma omp single
        {
            for (j = k + 1; j + 3 < N; j += 4)
            {
                Akj = _mm_loadu_ps(matrix[k] + j);
                Akj = _mm_div_ps(Akj, Akk);
                _mm_storeu_ps(matrix[k] + j, Akj);
            }
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (int i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
            Aik = _mm_set_ps1(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4)
            {
                Akj = _mm_loadu_ps(matrix[k] + j);
                Aij = _mm_loadu_ps(matrix[i] + j);
                AikMulAkj = _mm_mul_ps(Aik, Akj);
                Aij = _mm_sub_ps(Aij, AikMulAkj);
                _mm_storeu_ps(matrix[i] + j, Aij);
            }
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

void calculate_openmp_row()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

void calculate_openmp_column()
{
    int i, j, k;
#pragma omp parallel num_threads(NUM_THREADS), default(none), private(i, j, k), shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp for schedule(simd \
                         : guided)
        for (j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
            for (i = k + 1; i < N; i++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
        }
#pragma omp single
        {
            matrix[k][k] = 1;
            for (i = k + 1; i < N; i++)
            {
                matrix[i][k] = 0;
            }
        }
    }
}

void print_matrix()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_result(int time)
{
    cout  << time / LOOP <<endl;
    print_matrix();
}

void test(int n)
{
    N = n;
    cout << "=================================== " << N << " ===================================" << endl;
    struct timeval start;
    struct timeval end;
    float time = 0;
    init_data();
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_serial();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "serial:" << time / LOOP << "ms" << endl;
    print_result(time);

    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_SIMD();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "SIMD:" << time / LOOP << "ms" << endl;
    print_result(time);

    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_single_SIMD();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_single_SIMD:" << time / LOOP << "ms" << endl;
    print_result(time);

    
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_SIMD();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_SIMD:" << time / LOOP << "ms" << endl;
    print_result(time);
    
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_row();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_row:" << time / LOOP << "ms" << endl;
    print_result(time);
    
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_column();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_column:" << time / LOOP << "ms" << endl;
    print_result(time);
}



int main()
{
    int n=10;
    cin>>n;
    test(n);
    system("pause");
    return 0;
}