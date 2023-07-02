#include <semaphore.h>
#include <xmmintrin.h> 
#include <emmintrin.h> 
#include <pmmintrin.h> 
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h> 
#include <immintrin.h> 
#include <iostream>
#include <sys/time.h>
#include <pthread.h>
using namespace std;

const int N = 3000;
const int L = 100;

float matrix[N][N];

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

float data[N][N];
void init_data()
{
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

void calculate_SSE()
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

sem_t sem_Division;
pthread_barrier_t barrier;
const int THREAD_NUM = 8;
typedef struct
{
    int t_id;
} threadParam_t;
void *threadFunc_SSE(void *param)
{
    threadParam_t *thread_param_t = (threadParam_t *)param;
    int t_id = thread_param_t->t_id;
    for (int k = 0; k < N; k++)
    {
        if (t_id == 0)
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
            matrix[k][k] = 1.0;
        }
        else
        {
            sem_wait(&sem_Division);
        }
        if (t_id == 0)
        {
            for (int i = 1; i < THREAD_NUM; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            for (int i = k + t_id; i < N; i += (THREAD_NUM - 1))
            {
                __m128 Aik = _mm_set_ps1(matrix[i][k]);
                int j = k + 1;
                for (; j + 3 < N; j += 4)
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
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

void calculate_pthread_SSE()
{
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, THREAD_NUM);
    pthread_t threads[THREAD_NUM];
    threadParam_t thread_param_t[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc_SSE, (void *)(&thread_param_t[i]));
    }
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
}

const int LOOP = 1;
int main()
{
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

    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_SSE();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "SSE:" << time / LOOP << "ms" << endl;

    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_pthread_SSE();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "pthread_SSE:" << time / LOOP << "ms" << endl;
}

