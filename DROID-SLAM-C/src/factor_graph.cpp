// factor_graph.cpp
#include "factor_graph.hpp"

#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <cassert>

#ifdef SPDLOG_USE_SYSLOG
static const char* LOG_NAME = "factor-graph";
#else
static const char* LOG_NAME = "FactorGraph";
#endif


// Alister add 2025-11-28
Tensor5D build_motn(Tensor5D& coords0,
                    Tensor5D& coords1,
                    Tensor5D& target)
{
    int B = coords0.B;
    int N = coords0.N;
    int H = coords0.H;
    int W = coords0.W;
    const int C_out = 4;

    Tensor5D motn(B,N,C_out,H,W);

    for (int b=0;b<B;b++){
        for (int n=0;n<N;n++){
            for (int h=0;h<H;h++){
                for (int w=0;w<W;w++){
                    // coords1 - coords0
                    float dx1 = coords1(b,n,0,h,w) - coords0(b,n,0,h,w);
                    float dy1 = coords1(b,n,1,h,w) - coords0(b,n,1,h,w);

                    // target - coords1
                    float dx2 = target(b,n,0,h,w) - coords1(b,n,0,h,w);
                    float dy2 = target(b,n,1,h,w) - coords1(b,n,1,h,w);

                    // clamp
                    dx1 = std::max(-64.0f,std::min(64.0f,dx1));
                    dy1 = std::max(-64.0f,std::min(64.0f,dy1));
                    dx2 = std::max(-64.0f,std::min(64.0f,dx2));
                    dy2 = std::max(-64.0f,std::min(64.0f,dy2));

                    motn(b,n,0,h,w) = dx1;
                    motn(b,n,1,h,w) = dy1;
                    motn(b,n,2,h,w) = dx2;
                    motn(b,n,3,h,w) = dy2;
                }
            }
        }
    }

    return motn;
}

// Alister add 2025-11-28
Tensor4D corr(const Tensor5D& coords1) {
    // placeholder: user implements correlation volume
    int B = coords1.B;
    int N = coords1.N;
    int H = coords1.H;
    int W = coords1.W;
    int C = 64;  // example
    Tensor4D out(B,C,H,W);
    // compute correlation
    return out;
}

// Alister add 2025-11-28
// struct UpdateResult {
//     Tensor4D net;
//     Tensor5D delta;
//     Tensor4D weight;
//     std::vector<float> damping;
//     std::vector<int> upmask;
// };


UpdateResult FactorGraph::update_op(const Tensor4D& net, const Tensor4D& inp, const Tensor4D& corr,
                        const Tensor5D& motn, const std::vector<int>& ii, const std::vector<int>& jj)
    {
        UpdateResult res;
        // User implements network update in CPU
        return res;
    }

    // FactorGraph::FactorGraph()
    // {
    // }

    // -----------------------------
    // Constructor / Destructor
    // -----------------------------
    FactorGraph::FactorGraph(std::vector<DROID_SLAM_Prediction> *predBuffer,
                             UpdateOperator updateOp,
                             int maxFactors,
                             bool upsample)
        // : m_predBuffer(predBuffer),
        //   m_updateOp(updateOp),
        //   m_maxFactors(maxFactors),
        //   m_upsample(upsample)
    {
        // auto logger = spdlog::get(LOG_NAME);
        // if (logger)
        //     logger->info("[FactorGraph] constructed (maxFactors={}, upsample={})", m_maxFactors, m_upsample);
        // int N = m_ii.size();
        // m_coords0 = Tensor5D(m_B, N, 2, m_H, m_W);
        // m_target = Tensor5D(m_B, N, 2, m_H, m_W);
        // m_weight = Tensor4D(m_B, 2, m_H, m_W);
        // m_net = Tensor4D(m_B, 128, m_H, m_W); // TODD
        // m_inp = Tensor4D(m_B, 128, m_H, m_W); // TODO
        // m_damping = std::vector<float>(N, 0.0f);

        // // Initialize video arrays with correct sizes
        // int num_frames = 10; // example, replace with actual number of frames
        // m_disps.resize(num_frames, std::vector<float>(m_H * m_W, 0.0f));
        // m_poses.resize(num_frames);
        // m_intrinsics.resize(num_frames);
    }

FactorGraph::~FactorGraph()
{
    // auto logger = spdlog::get(LOG_NAME);
    // if (logger) logger->info("[FactorGraph] destroyed");
}

// -----------------------------
// Internal helpers
// -----------------------------
bool FactorGraph::_isValidIndex(int idx) const
{
    if (!m_predBuffer) return false;
    return (idx >= 0 && idx < static_cast<int>(m_predBuffer->size()));
}

void FactorGraph::_filterRepeatedEdges(std::vector<int>& ii, std::vector<int>& jj)
{
    // Remove any (ii,k,jj,k) pairs that intersect with existing active, bad or inactive edges
    std::vector<char> keep(ii.size(), 1);

    for (size_t k = 0; k < ii.size(); ++k) {
        int a = ii[k], b = jj[k];
        // Check active edges
        for (size_t e = 0; e < m_ii.size(); ++e) {
            if (a == m_ii[e] && b == m_jj[e]) { keep[k] = 0; break; }
        }
        if (!keep[k]) continue;
        // Check bad edges
        for (size_t e = 0; e < m_ii_bad.size(); ++e) {
            if (a == m_ii_bad[e] && b == m_jj_bad[e]) { keep[k] = 0; break; }
        }
        if (!keep[k]) continue;
        // Check inactive edges
        for (size_t e = 0; e < m_ii_inac.size(); ++e) {
            if (a == m_ii_inac[e] && b == m_jj_inac[e]) { keep[k] = 0; break; }
        }
    }

    // compact
    std::vector<int> ii2, jj2;
    ii2.reserve(ii.size());
    jj2.reserve(jj.size());
    for (size_t k = 0; k < ii.size(); ++k) {
        if (keep[k]) { ii2.push_back(ii[k]); jj2.push_back(jj[k]); }
    }
    ii.swap(ii2);
    jj.swap(jj2);
}

// -----------------------------
// addFactors - faithful to Python behaviour (simplified)
// -----------------------------
void FactorGraph::addFactors(const std::vector<int>& ii_in,
                             const std::vector<int>& jj_in,
                             bool remove)
{
    // auto logger = spdlog::get(LOG_NAME);
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_predBuffer) {
        // if (logger) logger->error("[addFactors] predBuffer is null");
        return;
    }

    if (ii_in.size() != jj_in.size()) {
        // if (logger) logger->error("[addFactors] ii/jj length mismatch: {} vs {}", ii_in.size(), jj_in.size());
        return;
    }

    // Copy inputs to mutable vectors
    std::vector<int> ii = ii_in;
    std::vector<int> jj = jj_in;

    // Remove duplicates with active/bad/inactive edges
    _filterRepeatedEdges(ii, jj);

    if (ii.empty()) {
        // if (logger) logger->debug("[addFactors] no new factors after filtering");
        return;
    }

    // If exceeding max factors and remove==true, remove oldest by age
    if (m_maxFactors > 0 && static_cast<int>(m_ii.size()) + static_cast<int>(ii.size()) > m_maxFactors && remove) {
        int total_needed = static_cast<int>(m_ii.size()) + static_cast<int>(ii.size());
        int num_remove = total_needed - m_maxFactors;
        if (num_remove > 0 && !m_age.empty()) {
            // sort indices by age descending (remove oldest)
            std::vector<size_t> order(m_age.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return m_age[a] > m_age[b]; });

            std::vector<bool> mask(m_age.size(), false);
            for (int k = 0; k < num_remove && k < static_cast<int>(order.size()); ++k) {
                mask[order[k]] = true;
            }
            rmFactors(mask, true);
        }
    }

    // Append nets: Python does `net = self.video.nets[ii].to(self.device).unsqueeze(0)`
    // Here: we don't maintain full tensor net arrays (outside scope). We assume
    // your update operator will fetch nets/inps from m_predBuffer on demand.
    // So we only append ii/jj/age vectors as the factor set.

    for (size_t k = 0; k < ii.size(); ++k) {
        int a = ii[k];
        int b = jj[k];
        if (!_isValidIndex(a) || !_isValidIndex(b)) {
            // if (logger) logger->warn("[addFactors] skipping out-of-range pair ({},{})", a, b);
            continue;
        }

        m_ii.push_back(a);
        m_jj.push_back(b);
        m_age.push_back(0);
    }

    // if (logger) logger->info("[addFactors] added {} factors, total {}", ii.size(), m_ii.size());
}

// -----------------------------
// rmFactors (masked removal). If store==true, push into inactive arrays.
// -----------------------------
void FactorGraph::rmFactors(const std::vector<bool>& mask, bool store)
{
    // auto logger = spdlog::get(LOG_NAME);
    std::lock_guard<std::mutex> lock(m_mutex);

    if (mask.size() != m_ii.size()) {
        // if (logger) logger->error("[rmFactors] mask size mismatch: {} != {}", mask.size(), m_ii.size());
        return;
    }

    std::vector<int> new_ii, new_jj, new_age;
    new_ii.reserve(m_ii.size());
    new_jj.reserve(m_jj.size());
    new_age.reserve(m_age.size());

    for (size_t k = 0; k < m_ii.size(); ++k) {
        if (mask[k]) {
            // store in inactive if requested
            if (store) {
                m_ii_inac.push_back(m_ii[k]);
                m_jj_inac.push_back(m_jj[k]);
            } else {
                m_ii_bad.push_back(m_ii[k]);
                m_jj_bad.push_back(m_jj[k]);
            }
        } else {
            new_ii.push_back(m_ii[k]);
            new_jj.push_back(m_jj[k]);
            new_age.push_back(m_age[k]);
        }
    }

    m_ii.swap(new_ii);
    m_jj.swap(new_jj);
    m_age.swap(new_age);

    // if (logger) logger->info("[rmFactors] removed factors, remaining={}", m_ii.size());
}

// -----------------------------
// rmKeyframe: drop a frame at index idx and shift buffers (very similar to Python)
// -----------------------------
void FactorGraph::rmKeyframe(int idx)
{
    // auto logger = spdlog::get(LOG_NAME);
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_predBuffer) {
        // if (logger) logger->error("[rmKeyframe] predBuffer is null");
        return;
    }

    int t = static_cast<int>(m_predBuffer->size());
    if (idx < 0 || idx >= t - 1) {
        // if (logger) logger->warn("[rmKeyframe] idx {} out of range [0, {})", idx, t-1);
        return;
    }

    // Erase the frame at idx (shift left the buffer) - vector and deque support erase
    m_predBuffer->erase(m_predBuffer->begin() + idx);

    // Adjust stored ii_inac/jj_inac indices and ii/jj indices >= idx
    for (auto &v : m_ii_inac) if (v >= idx) --v;
    for (auto &v : m_jj_inac) if (v >= idx) --v;
    for (auto &v : m_ii_bad) if (v >= idx) --v;
    for (auto &v : m_jj_bad) if (v >= idx) --v;

    // Remove inactive edges referencing removed frame
    {
        std::vector<int> new_ii_inac, new_jj_inac;
        for (size_t k = 0; k < m_ii_inac.size(); ++k) {
            if (m_ii_inac[k] == idx || m_jj_inac[k] == idx) continue;
            new_ii_inac.push_back(m_ii_inac[k]);
            new_jj_inac.push_back(m_jj_inac[k]);
        }
        m_ii_inac.swap(new_ii_inac);
        m_jj_inac.swap(new_jj_inac);
    }

    // Adjust active edges
    for (auto &v : m_ii) if (v >= idx) --v;
    for (auto &v : m_jj) if (v >= idx) --v;

    // Remove any active edges that reference invalid indices now
    std::vector<bool> mask(m_ii.size(), false);
    for (size_t k = 0; k < m_ii.size(); ++k) {
        if (!_isValidIndex(m_ii[k]) || !_isValidIndex(m_jj[k])) mask[k] = true;
    }
    if (!mask.empty()) rmFactors(mask, false);

    // if (logger) logger->info("[rmKeyframe] removed frame {}, now frames={}", idx, m_predBuffer->size());
}

// -----------------------------
// addNeighborhoodFactors: add edges between neighbors within radius r
// -----------------------------
void FactorGraph::addNeighborhoodFactors(int i0, int i1, int r)
{
    // auto logger = spdlog::get(LOG_NAME);
    std::lock_guard<std::mutex> lock(m_mutex);

    if (i1 <= i0) { 
        // if (logger) logger->debug("[addNeighborhoodFactors] empty range");
        return;
    }

    std::vector<int> ii_list, jj_list;
    for (int i = i0; i < i1; ++i) {
        for (int j = i0; j < i1; ++j) {
            int diff = std::abs(i - j);
            if (diff > 0 && diff <= r) {
                ii_list.push_back(i);
                jj_list.push_back(j);
            }
        }
    }

    addFactors(ii_list, jj_list, false);
    // if (logger) logger->info("[addNeighborhoodFactors] added {} edges", ii_list.size());
}

// -----------------------------
// addProximityFactors: faithful translation of Python function
// -----------------------------
bool FactorGraph::addProximityFactors(int t0, int t1, int rad, int nms, float beta, float thresh, bool remove)
{
    // auto logger = spdlog::get(LOG_NAME);
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_predBuffer) {
        // if (logger) logger->error("[addProximityFactors] predBuffer is null");
        return false;
    }

    int t = static_cast<int>(m_predBuffer->size());
    if (t <= 0) {
        // if (logger) logger->debug("[addProximityFactors] empty buffer");
        return false;
    }

    if (t0 < 0) t0 = 0;
    if (t1 < 0) t1 = 0;
    if (t0 >= t) t0 = t - 1;
    if (t1 >= t) t1 = t - 1;

    // Build meshgrid arrays II, JJ
    std::vector<int> ix, jx;
    for (int i = t0; i < t; ++i) ix.push_back(i);
    for (int j = t1; j < t; ++j) jx.push_back(j);

    int NI = static_cast<int>(ix.size());
    int NJ = static_cast<int>(jx.size());
    if (NI == 0 || NJ == 0) {
        // if (logger) logger->debug("[addProximityFactors] no candidates");
        return false;
    }

    int N = NI * NJ;
    std::vector<int> II; II.resize(N);
    std::vector<int> JJ; JJ.resize(N);
    std::vector<float> D; D.resize(N, std::numeric_limits<float>::infinity());

    for (int a = 0; a < NI; ++a) {
        for (int b = 0; b < NJ; ++b) {
            int idx = a * NJ + b;
            II[idx] = ix[a];
            JJ[idx] = jx[b];
            // compute distance using video.pose-based measure
            float d = 0; //TODO: distance(II[idx], JJ[idx], beta);
            // apply python masks: ii - rad < jj -> inf, d>100 -> inf
            if (II[idx] - rad < JJ[idx] || d > 100.0f) d = std::numeric_limits<float>::infinity();
            D[idx] = d;
        }
    }

    // Combine existing ii/jj, ii_bad/jj_bad, ii_inac/jj_inac for NMS suppression
    auto applySuppress = [&](int i, int j) {
        for (int di = -nms; di <= nms; ++di) {
            for (int dj = -nms; dj <= nms; ++dj) {
                if (std::abs(di) + std::abs(dj) <= std::max(std::min(std::abs(i - j) - 2, nms), 0)) {
                    int i1 = i + di;
                    int j1 = j + dj;
                    if (i1 >= t0 && i1 < t && j1 >= t1 && j1 < t) {
                        int idx = (i1 - t0) * NJ + (j1 - t1);
                        if (idx >= 0 && idx < N) D[idx] = std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
    };

    // gather previous edges lists
    std::vector<int> ii1, jj1;
    ii1.insert(ii1.end(), m_ii.begin(), m_ii.end());
    ii1.insert(ii1.end(), m_ii_bad.begin(), m_ii_bad.end());
    ii1.insert(ii1.end(), m_ii_inac.begin(), m_ii_inac.end());

    jj1.insert(jj1.end(), m_jj.begin(), m_jj.end());
    jj1.insert(jj1.end(), m_jj_bad.begin(), m_jj_bad.end());
    jj1.insert(jj1.end(), m_jj_inac.begin(), m_jj_inac.end());

    for (size_t k = 0; k < ii1.size() && k < jj1.size(); ++k) {
        applySuppress(ii1[k], jj1[k]);
    }

    // Add local neighbors and stereo (bidirectional)
    std::vector<std::pair<int,int>> es;
    for (int i = t0; i < t; ++i) {
        if (_isValidIndex(i) && (*m_predBuffer)[i].stereo) {
            es.emplace_back(i, i);
            int idx = (i - t0) * NJ + (i - t1);
            if (idx >= 0 && idx < N) D[idx] = std::numeric_limits<float>::infinity();
        }
        for (int j = std::max(i - rad - 1, 0); j < i; ++j) {
            es.emplace_back(i, j);
            es.emplace_back(j, i);
            int idx = (i - t0) * NJ + (j - t1);
            if (idx >= 0 && idx < N) D[idx] = std::numeric_limits<float>::infinity();
        }
    }

    // Sort candidate indices by distance
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return D[a] < D[b]; });

    for (int k : order) {
        if (D[k] > thresh) continue;

        if (m_maxFactors > 0 && static_cast<int>(es.size()) > m_maxFactors) break;

        int i = II[k];
        int j = JJ[k];

        // append both directions
        es.emplace_back(i, j);
        es.emplace_back(j, i);

        // NMS suppression around the newly added edge
        for (int di = -nms; di <= nms; ++di) {
            for (int dj = -nms; dj <= nms; ++dj) {
                if (std::abs(di) + std::abs(dj) <= std::max(std::min(std::abs(i - j) - 2, nms), 0)) {
                    int i1 = i + di;
                    int j1 = j + dj;
                    if (i1 >= t0 && i1 < t && j1 >= t1 && j1 < t) {
                        int idx2 = (i1 - t0) * NJ + (j1 - t1);
                        if (idx2 >= 0 && idx2 < N) D[idx2] = std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
    }

    // Convert es to two vectors and add
    std::vector<int> ii_add, jj_add;
    ii_add.reserve(es.size());
    jj_add.reserve(es.size());
    for (auto &p : es) { ii_add.push_back(p.first); jj_add.push_back(p.second); }

    addFactors(ii_add, jj_add, remove);

    // if (logger) logger->info("[addProximityFactors] candidates={} added={}", N, ii_add.size());
    return true;
}

// -----------------------------
// compute correlation (placeholder)
// -----------------------------
bool FactorGraph::_computeCorrelation(int ii, int jj)
{
    (void)ii; (void)jj;
    // TODO: call CV28 corr block or other implementation
    // return true on success
    return true;
}

// -----------------------------
// Run update operator wrapper (per-edge calling style)
// -----------------------------
bool FactorGraph::_runUpdateOperator(int ii, int jj)
{
    // auto logger = spdlog::get(LOG_NAME);
    if (!m_updateOp) {
        // if (logger) logger->error("[_runUpdateOperator] updateOp not set");
        return false;
    }
    if (!_isValidIndex(ii) || !_isValidIndex(jj)) {
        // if (logger) logger->warn("[_runUpdateOperator] invalid indices ii={} jj={}", ii, jj);
        return false;
    }

    // attempt to provide fnet/inp/gmap pointers to the update op
    const DROID_SLAM_Prediction& A = (*m_predBuffer)[ii];
    const DROID_SLAM_Prediction& B = (*m_predBuffer)[jj];

    const float* fnetBuf = A.netBuff;
    const float* cnetBuf = A.inpBuff;
    const float* gmapBuf = B.gmapBuff;

    // call the user-provided update operator
    m_updateOp(fnetBuf, cnetBuf, gmapBuf, ii, jj);

    return true;
}


std::pair<Tensor5D, Tensor5D> FactorGraph::reproject(const std::vector<int>& ii, const std::vector<int>& jj)
{
    int B = 1;                 // batch
    int N = ii.size();         // number of edges
    int H = m_H;
    int W = m_W;
    Tensor5D coords1(B, N, 2, H, W);
    Tensor5D valid_mask(B, N, 1, H, W);

    for (int k = 0; k < N; ++k)
    {
        int idx_i = ii[k];
        int idx_j = jj[k];

        SE3 pose_i = m_poses[idx_i];
        SE3 pose_j = m_poses[idx_j];

        auto& disp_i = m_disps[idx_i];   // [H x W] disparity map
        Intrinsics K = m_intrinsics[idx_i];

        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                float d = disp_i[h*W + w]; // disparity = 1/depth

                if (d <= 0.0f) {
                    coords1(0,k,0,h,w) = 0.0f;
                    coords1(0,k,1,h,w) = 0.0f;
                    valid_mask(0,k,0,h,w) = 0.0f;
                    continue;
                }

                // 1. Backproject pixel to 3D in frame i
                float x = (w - K.cx) / K.fx;
                float y = (h - K.cy) / K.fy;
                float z = 1.0f / d; // depth

                float X = x*z;
                float Y = y*z;
                float Z = z;

                // 2. Transform 3D point from frame i -> frame j
                // Assuming SE3 matrix m[3][4]: [R|t]
                // float Xj = pose_j.m[0][0]*X + pose_j.m[0][1]*Y + pose_j.m[0][2]*Z + pose_j.m[0][3];
                // float Yj = pose_j.m[1][0]*X + pose_j.m[1][1]*Y + pose_j.m[1][2]*Z + pose_j.m[1][3];
                // float Zj = pose_j.m[2][0]*X + pose_j.m[2][1]*Y + pose_j.m[2][2]*Z + pose_j.m[2][3];
                // 1. Backproject pixel to 3D in frame i
             
                Vec3 Pi_xyz(X, Y, Z);

                // 2. Transform 3D point from frame i -> frame j
                Vec3 Pw = pose_i.toWorld(Pi_xyz);
                Vec3 Pj_xyz = pose_j.fromWorld(Pw);

                float Xj = Pj_xyz.x;
                float Yj = Pj_xyz.y;
                float Zj = Pj_xyz.z;

                // 3. Project back to pixel coords in frame j
                float u = K.fx * (Xj / Zj) + K.cx;
                float v = K.fy * (Yj / Zj) + K.cy;

                coords1(0,k,0,h,w) = u;
                coords1(0,k,1,h,w) = v;

                // valid mask
                valid_mask(0,k,0,h,w) = (Zj > 0.0f) ? 1.0f : 0.0f;
            }
        }
    }

    return {coords1, valid_mask};
}


void FactorGraph::ba(
    Tensor5D& target,
    Tensor4D& weight,
    const std::vector<float>& eta,
    const std::vector<int>& ii,
    const std::vector<int>& jj,
    int t0,
    int t1,
    int itrs,
    float lm,
    float ep,
    bool motion_only)
{
    // 1. Determine t1 if not specified
    if (t1 < 0) {
        int max_i = *std::max_element(ii.begin(), ii.end());
        int max_j = *std::max_element(jj.begin(), jj.end());
        t1 = std::max(max_i, max_j) + 1;
    }

    // 2. Lock if needed (Python uses get_lock())
    // std::lock_guard<std::mutex> lock(m_mutex); // add mutex if multithreaded

    // 3. Iterate DBA
    for (int iter = 0; iter < itrs; ++iter) {
        // For each edge (ii[k], jj[k])
        int N = ii.size();
        for (int k = 0; k < N; ++k) {
            int idx_i = ii[k];
            int idx_j = jj[k];

            // target, weight, eta → update poses and disparity
            // Here you can call a backend function, or implement a simplified update:
            // e.g., perform a small gradient descent step on m_poses[idx_j] using target and weight

            // NOTE: Full BA is very complex; for CV28 we can implement a simplified version
        }
    }

    // 4. Clamp disparities (Python: self.disps.clamp_(min=0.001))
    int num_frames = m_disps.size();
    for (int f = 0; f < num_frames; ++f) {
        int H = m_H;
        int W = m_W;
        for (int i = 0; i < H*W; ++i) {
            if (m_disps[f][i] < 0.001f) m_disps[f][i] = 0.001f;
        }
    }
}


float FactorGraph::computeDistance(int ii, int jj, float beta)
{
    if (ii < 0 || jj < 0) return 1e9f;
    if (ii >= m_poses.size() || jj >= m_poses.size()) return 1e9f;

    float d1 = frameDistance(ii, jj, beta);
    float d2 = frameDistance(jj, ii, beta);

    return 0.5f * (d1 + d2);
}

void FactorGraph::setIntrinsics(float fx, float fy, float cx, float cy)
{
    m_fx = fx;
    m_fy = fy;
    m_cx = cx;
    m_cy = cy;
}


float FactorGraph::frameDistance(int ii, int jj, float beta)
{
    const SE3& Pi = m_poses[ii];
    const SE3& Pj = m_poses[jj];

    const std::vector<float>& disp_i = m_disps[ii];
    const std::vector<float>& disp_j = m_disps[jj];

    // Intrinsics
    const Intrinsics& K = m_intrinsics[0];
    float m_fx = K.fx;
    float m_fy = K.fy;
    float m_cx = K.cx;
    float m_cy = K.cy;


    int H = m_H;   // 你需要在 class 裡加入 m_H, m_W
    int W = m_W;

    float photometric_cost = 0.f;
    float geometric_cost = 0.f;
    int count = 0;

    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
    {
        float di = disp_i[y * W + x];
        if (di <= 0) continue;

        float depth_i = 1.0f / di;

        //---------------------------------------------------------
        //  lift pixel (x,y) → 3D point in frame i
        //---------------------------------------------------------
        float X = (x - m_cx) / m_fx * depth_i;
        float Y = (y - m_cy) / m_fy * depth_i;
        float Z = depth_i;

        // transform to frame j :  Pj_xyz = R_j * (R_i^T * Pi_xyz - t_i) + t_j
        Vec3 Pi_xyz = {X, Y, Z};
        Vec3 Pj_xyz = Pj.fromWorld(Pi.toWorld(Pi_xyz));  
        // 取決於你的 SE3 實作
        // 若你沒有 toWorld/fromWorld，我會幫你 rewrite

        if (Pj_xyz.z <= 0) continue;

        //---------------------------------------------------------
        // project to frame j
        //---------------------------------------------------------
        float u = m_fx * (Pj_xyz.x / Pj_xyz.z) + m_cx;
        float v = m_fy * (Pj_xyz.y / Pj_xyz.z) + m_cy;

        if (u < 0 || u >= W - 1 || v < 0 || v >= H - 1)
            continue;

        //---------------------------------------------------------
        // photometric error
        //---------------------------------------------------------
        float Ii = di;
        float Ij = bilinear(disp_j, u, v, W, H);

        photometric_cost += fabs(Ii - Ij);

        //---------------------------------------------------------
        // geometric (depth) error
        //---------------------------------------------------------
        float dj = bilinear(disp_j, u, v, W, H);
        if (dj > 0) {
            float depth_j = 1.0f / dj;
            geometric_cost += fabs(depth_i - depth_j);
        }

        count++;
    }

    if (count == 0) return 1e9f;

    float d = photometric_cost / count + beta * (geometric_cost / count);
    return d;
}

float FactorGraph::bilinear(const std::vector<float>& disp, float u, float v, int W, int H)
{
    int x0 = int(u);
    int y0 = int(v);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = u - x0;
    float dy = v - y0;

    float v00 = disp[y0 * W + x0];
    float v01 = disp[y0 * W + x1];
    float v10 = disp[y1 * W + x0];
    float v11 = disp[y1 * W + x1];

    float a = v00 * (1 - dx) + v01 * dx;
    float b = v10 * (1 - dx) + v11 * dx;
    return a * (1 - dy) + b * dy;
}

void FactorGraph::upsample(const std::vector<int>& ix, const std::vector<int>& mask)
{
    int B = m_B;
    int dim = 1;  // assume disparity map has 1 channel
    int ht = m_H;
    int wd = m_W;
    int up_factor = 8;

    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < ix.size(); ++k) {
            int fidx = ix[k];

            // Allocate upsampled disparity
            std::vector<float> disps_up(up_factor*ht * up_factor*wd * dim, 0.0f);

            // Simple nearest-neighbor upsample (replace softmax/unfold if needed)
            for (int h = 0; h < ht; ++h) {
                for (int w = 0; w < wd; ++w) {
                    float val = m_disps[fidx][h*wd + w]; // single channel
                    for (int uh = 0; uh < up_factor; ++uh) {
                        for (int uw = 0; uw < up_factor; ++uw) {
                            int hh = h*up_factor + uh;
                            int ww = w*up_factor + uw;
                            disps_up[hh*up_factor*wd + ww] = val; // copy value
                        }
                    }
                }
            }

            // Store back to m_disps_up (member variable, pre-allocated)
            m_disps_up[fidx] = disps_up;
        }
    }
}


void FactorGraph::update_damping_and_target(Tensor5D& target,
                               Tensor4D& weight,
                               std::vector<float>& damping,
                               const std::vector<int>& ii,
                               const std::vector<int>& jj,
                               Tensor5D& delta,
                               float EP)
{
    int B = target.B;
    int N = target.N;
    int H = target.H;
    int W = target.W;

    // target += delta
    for (int b=0;b<B;b++){
        for (int n=0;n<N;n++){
            for (int h=0;h<H;h++){
                for (int w=0;w<W;w++){
                    for (int c=0;c<2;c++)
                        target(b,n,c,h,w) += delta(b,n,c,h,w);
                }
            }
        }
    }

    // damping update
    std::unordered_set<int> unique_ii(ii.begin(), ii.end());
    for (int u : unique_ii)
        damping[u] = 0.2f * damping[u] + EP;
}


// -----------------------------
// update(): iterate factors and run correlation + updateOp
// -----------------------------
void FactorGraph::update(int t0=-1, int t1=-1, int itrs=2, bool use_inactive=false, float EP=1e-7, bool motion_only=false)
{
    // 1. motion features
    Tensor5D coords1, mask;
    std::tie(coords1, mask) = FactorGraph::reproject(m_ii, m_jj); // user-defined

    Tensor5D motn = build_motn(m_coords0, coords1, m_target);

    // 2. correlation features
    Tensor4D corr_feat = corr(coords1);

    // 3. update op
    UpdateResult upd = update_op(m_net, m_inp, corr_feat, motn, m_ii, m_jj);
    m_net = upd.net;

    // 4. damping and target update
    FactorGraph::update_damping_and_target(m_target, m_weight, m_damping, m_ii, m_jj, upd.delta, EP);

    // 5. dense BA
    if (t0 < 0) t0 = 1;  // min frame index
    ba(m_target, upd.weight, upd.damping, m_ii, m_jj, t0, t1, itrs, 1e-4, 0.1, motion_only);

    // 6. optional upsample
    if (m_upsample) FactorGraph::upsample(m_ii, upd.upmask);

    m_updateAge += 1;
}
