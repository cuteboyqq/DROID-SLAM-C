/*
  (C) 2023-2025 Wistron NeWeb Corporation (WNC) - All Rights Reserved

  This software and its associated documentation are the confidential and
  proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
  may not be copied, modified, distributed, or otherwise disclosed to third
  parties without the express written consent of the Company.

  Unauthorized reproduction, distribution, or disclosure of this software and
  its associated documentation or the information contained herein is a
  violation of applicable laws and may result in severe legal penalties.
*/

#ifndef __DroidFrontend__
#define __DroidFrontend__

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
//#include <atomic>
#include "factor_graph.hpp"
#include "dataStructures.h"

class DroidFrontend
{
public:
    DroidFrontend(std::vector<DROID_SLAM_Prediction>* predBuffer,FactorGraph* factorGraph);

    DroidFrontend();
    ~DroidFrontend();
    void runThread();
    void stopThread();

private:
    // Thread Worker
    void _runFrontendFunc();

    // Core SLAM steps
    void _initialize();
    void _update();
    void _initNextState();

private:
    std::vector<DROID_SLAM_Prediction>* m_predictionBuffer;
    std::deque<std::pair<int, DROID_SLAM_Prediction>> m_slam_predictionBuffer;
    FactorGraph* m_graph;

    std::thread m_threadFrontend;
    std::mutex m_mutex;
    std::condition_variable m_condition;

    // std::atomic<bool> m_threadTerminated{false};
    // std::atomic<bool> m_threadStarted{false};
    bool m_threadTerminated = false;
    bool m_threadStarted = false;

    bool m_isInitialized = false;

    int m_t0 = 0;
    int m_t1 = 0;

    int m_warmup = 8;
    int m_iters1 = 1;
    int m_iters2 = 0;
    int m_maxAge = 20;

    float m_beta = 0.3f;
    float m_keyframeThresh = 0.5f;
    int m_frontendWindow = 20;
    int m_depthWindow = 3;
};

#endif
