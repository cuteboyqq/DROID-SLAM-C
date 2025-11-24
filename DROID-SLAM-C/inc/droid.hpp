#pragma once
#include <vector>
#include <string>
#include <memory>
#include <thread>
// #include "droid_net.hpp"
#include "depth_video.hpp"
#include "motion_filter.hpp"
#include "droid_frontend.hpp"
#include <deque>
#include <condition_variable>
#include <chrono>
// Ambarella CV28
#include <eazyai.h>
#include <opencv2/core.hpp>
// #include "droid_backend.hpp"
// #include "pose_trajectory_filler.hpp"

struct DroidArgs {
    std::string weights;
    bool disable_vis;
    int image_size;
    int buffer;
    bool stereo;
    float filter_thresh;
};

class Droid {
public:
    Droid(const DroidArgs& args);
    ~Droid();

    // Main tracking call
    void track(double tstamp, 
               const std::vector<uint8_t>& image,
               const std::vector<float>& depth = {}, 
               const Eigen::Vector4d& intrinsics = {});

    // Terminate and return camera trajectory
    std::vector<std::vector<float>> terminate();

    void updateInputFrame(ea_tensor_t* imgTensor, int frameIdx);
    

    // Thread Management
    bool m_bInferenced      = true;
    bool m_bProcessed       = true;
    bool m_threadTerminated = false;
    bool m_threadStarted    = false;
    bool m_bDone            = false;

private:
    void load_weights(const std::string& weights);
    bool _runDroidFunc();

private:
    char*       	m_ptrModelPath  = NULL;
    ea_net_t*       net             = NULL;
    ea_net_t*       fnet            = NULL;
    ea_net_t*       cnet            = NULL;
    ea_net_t*       update          = NULL;
    ea_tensor_t* 	m_img           = NULL; //TODO:
    ea_tensor_t* 	m_inputTensor   = NULL;
    DepthVideo video;
    MotionFilter filterx;
    DroidFrontend frontend;
    // DroidBackend backend;
    // PoseTrajectoryFiller traj_filler;

    DroidArgs args;

    std::thread visualizer_thread;
    bool disable_vis;

    // === Thread Management === //
    std::thread             m_threadInference;
    mutable std::mutex      m_pred_mutex;
    mutable std::mutex      m_mutex;
    std::condition_variable m_condition;
    // WakeCallback            m_wakeFunc;

    std::deque<std::pair<int, ea_tensor_t*>> m_inputFrameBuffer;
};
