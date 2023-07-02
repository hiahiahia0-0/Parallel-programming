#ifndef FINAL_KMEANS_H
#define FINAL_KMEANS_H

class KMeans {
    virtual void calculate() = 0;
    virtual void updateCentroids() = 0;
    virtual void changeMemory() = 0;

protected:
    float **data{};                         
    int N{};                               
    int D{};                                
    int K{};                                
    int L = 500;                            
    float **centroids{};                   
    int *clusterCount{};                    
    int *clusterLabels{};                   
    int method;                  
               
    void initCentroidsRandom();
    virtual float calculateDistance(const float *, const float *) const;

public:
    explicit KMeans(int k);
    KMeans(int k, int method);
    ~KMeans();
    void initData(float **data, int n, int d);
    virtual void fit() = 0;
    void printResult();
    int getClusterNumber() const;
};


#endif //FINAL_KMEANS_H
