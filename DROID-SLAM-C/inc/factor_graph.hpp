#ifndef FACTOR_GRAPH_HPP
#define FACTOR_GRAPH_HPP

#include <vector>
#include <mutex>
#include <memory>
#include <array>
#include <functional>

#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

#include "dataStructures.h"
#include "datastructure_slam.h"
// =========================
// FactorGraph
// =========================

class FactorGraph
{
public:
    using UpdateOperator = std::function<void(const float* fnetBuff,
                                              const float* cnetBuff,
                                              const float* gmapBuff,
                                              int ii, int jj)>;

    // =========================
    // Constructor / Destructor
    // =========================
    // FactorGraph();


    FactorGraph(std::vector<DROID_SLAM_Prediction>* predBuffer,
                UpdateOperator updateOp,
                int maxFactors = 48,
                bool upsample = false);

    ~FactorGraph();

    // =========================
    // Factor Management
    // =========================
    bool addProximityFactors(int t0, 
                            int t1, 
                            int rad, 
                            int nms,
                            float beta, 
                            float thresh, 
                            bool remove);

    void addNeighborhoodFactors(int i0, int i1, int r);
    


    void addFactors(const std::vector<int>& ii_in,
                    const std::vector<int>& jj_in,
                    bool remove);


    void rmFactors(const std::vector<bool>& mask, bool store);

    void rmKeyframe(int idx);

    std::pair<Tensor5D, Tensor5D> reproject(const std::vector<int>& ii, 
                                            const std::vector<int>& jj);

    
    void ba(Tensor5D& target,
            Tensor4D& weight,
            const std::vector<float>& eta,
            const std::vector<int>& ii,
            const std::vector<int>& jj,
            int t0,
            int t1,
            int itrs,
            float lm,
            float ep,
            bool motion_only);


    void upsample(const std::vector<int>& ix, const std::vector<int>& mask);

    float computeDistance(int ii, int jj, float beta);
    float frameDistance(int ii, int jj, float beta);
    float bilinear(const std::vector<float>& disp, float u, float v, int W, int H);
    void setIntrinsics(float fx, float fy, float cx, float cy);
    // =========================
    // Graph Update
    // =========================
    void update(int t0, int t1, int itrs, bool use_inactive, float EP, bool motion_only);

    // =========================
    // Getter
    // =========================
    inline const std::vector<int>& getActiveEdges() const { return m_activeEdges; }
    inline const std::vector<int>& allEdges() const { return m_edges; }

public:
    // Tensors
    Tensor5D m_coords0;
    Tensor5D m_target;
    Tensor4D m_weight;
    Tensor4D m_net;
    Tensor4D m_inp;

    // Damping
    std::vector<float> m_damping;

    // Video-related data
    std::vector<std::vector<float>> m_disps;       // per-frame disparity maps (flattened H*W)
    std::vector<std::vector<float>> m_disps_up;       // per-frame disparity maps (flattened H*W)
    std::vector<SE3> m_poses;                      // per-frame poses
    std::vector<Intrinsics> m_intrinsics;          // per-frame camera intrinsics

    int m_B = 1;
    int m_H = 41; 
    int m_W = 73;
    int m_updateAge = 0;
    bool m_upsample = false;

    float m_fx;
    float m_fy;
    float m_cx;
    float m_cy;


    // Frame edges
    std::vector<int> m_ii;   // edge_i indices
    std::vector<int> m_ii_bad;   // edge_i bad indices
    std::vector<int> m_ii_inac;   // edge_i inactive indices
    std::vector<int> m_jj;   // edge_j indices
    std::vector<int> m_jj_bad;   // edge_j bad indices
    std::vector<int> m_jj_inac;   // edge_j inactive indices
    std::vector<int> m_age;


    // Placeholder CPU function
    UpdateResult update_op(const Tensor4D& net, const Tensor4D& inp, const Tensor4D& corr,
                        const Tensor5D& motn, const std::vector<int>& ii, const std::vector<int>& jj);
 
private:
    // =========================
    // Internal Helper Functions
    // =========================
    bool _computeCorrelation(int ii, int jj);
    bool _runUpdateOperator(int ii, int jj);
    bool _isValidIndex(int idx) const;
    void _filterRepeatedEdges(std::vector<int>& ii,std::vector<int>& jj);
    void update_damping_and_target(Tensor5D& target,
                                    Tensor4D& weight,
                                    std::vector<float>& damping,
                                    const std::vector<int>& ii,
                                    const std::vector<int>& jj,
                                    Tensor5D& delta,
                                    float EP);

private:
    // =========================
    // External Inputs
    // =========================
    std::vector<DROID_SLAM_Prediction>* m_predBuffer = nullptr;
    UpdateOperator m_updateOp;


    // =========================
    // Internal State
    // =========================
    std::vector<int> m_edges;        // all edge ids
    std::vector<int> m_activeEdges;  // filtered edges

    int m_maxFactors = 48;

    std::mutex m_mutex;

};

#endif // FACTOR_GRAPH_HPP
