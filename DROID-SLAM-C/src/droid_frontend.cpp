/*
  (C) 2025-2026 Wistron NeWeb Corporation (WNC) - All Rights Reserved

  This software and its associated documentation are the confidential and
  proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
  may not be copied, modified, distributed, or otherwise disclosed to third
  parties without the express written consent of the Company.

  Unauthorized reproduction, distribution, or disclosure of this software and
  its associated documentation or the information contained herein is a
  violation of applicable laws and may result in severe legal penalties.
*/

#include "droid_frontend.hpp"
#include <spdlog/spdlog.h>

DroidFrontend::DroidFrontend(std::vector<DROID_SLAM_Prediction>* predBuffer, FactorGraph* factorGraph)       
    : m_predictionBuffer(predBuffer),
      m_graph(factorGraph)
{
    assert(predBuffer != nullptr);
    assert(factorGraph != nullptr);

    if (m_predictionBuffer->empty()) {
    std::cerr << "prediction buffer is empty!" << std::endl;
    }

}

DroidFrontend::DroidFrontend()
{
}

// 記得 destructor 釋放
DroidFrontend::~DroidFrontend() {
    delete m_graph;
}

void DroidFrontend::runThread()
{
    m_threadFrontend = std::thread(&DroidFrontend::_runFrontendFunc, this);
}


void DroidFrontend::stopThread()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_threadTerminated = true;
    }

    m_condition.notify_all();

    if (m_threadFrontend.joinable())
        m_threadFrontend.join();
}


void DroidFrontend::_runFrontendFunc()
{
// #ifdef SPDLOG_USE_SYSLOG
//     auto logger = spdlog::get("slam-frontend");
// #else
//     auto logger = spdlog::get("SLAM-Frontend");
// #endif

//     logger->info("DroidFrontend thread started");

    while (!m_threadTerminated)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        // Wait until:
        //   1. Thread is started
        //   2. Prediction buffer has frames
        //   3. Or request to terminate
        m_condition.wait(lock,
            [this]()
            {
                return m_threadTerminated ||
                      (m_threadStarted && !m_predictionBuffer->empty());
            });

        if (m_threadTerminated)
            break;

        int counter = m_predictionBuffer->size();
        if (counter == 0)
            continue;

        // ---- Initialization Stage ----
        if (!m_isInitialized && counter >= m_warmup)
        {
            // logger->info("Initializing SLAM...");
            _initialize();
            _initNextState();
            continue;
        }

        // ---- Running Updates ----
        if (m_isInitialized && m_t1 < counter - 1)
        {
            _update();
            _initNextState();
        }
        lock.unlock();
    } // while

    // logger->info("DroidFrontend thread terminated");
}


void DroidFrontend::_initialize()
{
    int counter = m_predictionBuffer->size();
    m_t0 = 0;
    m_t1 = counter - 1;

    // Add initial neighborhood factors
    m_graph->addNeighborhoodFactors(m_t0, m_t1, 3);

    for (int i = 0; i < 8; i++)
        m_graph->update(1, -1, 2, true, 1e-7, false);

    m_graph->addProximityFactors(0, 0, 2, 2, 0.3, m_keyframeThresh, false);

    for (int i = 0; i < 8; i++)
        m_graph->update(1, -1, 2, true, 1e-7, false);

    // Copy pose/disp from previous frame
    auto& prev = m_predictionBuffer->at(m_t1 - 1);
    auto& curr = m_predictionBuffer->at(m_t1);

    curr.pose = prev.pose;
    memcpy(curr.disps, prev.disps, sizeof(float) * 128 * 128);

    m_isInitialized = true;
    m_threadStarted = true;
}


void DroidFrontend::_update()
{
    m_t1++;

    auto& prev = m_predictionBuffer->at(m_t1 - 1);
    auto& curr = m_predictionBuffer->at(m_t1);

    // 1. Remove aged factors
    int m_maxAge = 20;
    std::vector<bool> mask(m_graph->m_age.size(), false);  // default: keep all

    for (size_t k = 0; k < m_graph->m_age.size(); ++k) {
        if (m_graph->m_age.size() > m_maxAge) {
            mask[k] = true;  // mark for removal
        }
    }

    // now call rmFactors
    m_graph->rmFactors(mask, false);  // or true if you want to store inactive

    // 2. Add proximity factors
    m_graph->addProximityFactors(
        m_t1 - 5,
        std::max(m_t1 - m_frontendWindow, 0),
        2,
        2,
        m_keyframeThresh,
        0.3,
        true);
    
    // 3. Depth: use sensor depth if available
    for (int i = 0; i < 128 * 128; i++)
    {
        if (curr.disps[i] <= 0.0f)
            curr.disps[i] = prev.disps[i];
    }

    // 4. First optimization loop
    for (int i = 0; i < m_iters1; i++)
        m_graph->update(-1, -1, 2, true, 1e-7, false);

    // 5. Compute motion
    float d = m_graph->computeDistance(m_t1 - 4, m_t1 - 2, m_beta);

    // 6. Decide whether to remove keyframe
    if (d < 2 * m_keyframeThresh)
    {
        m_graph->rmKeyframe(m_t1 - 3);
        m_t1--;
        return;
    }

    // 7. Second optimization
    for (int i = 0; i < m_iters2; i++)
        m_graph->update(-1, -1, 2, true, 1e-7, false);
}

void DroidFrontend::_initNextState()
{
    auto& prev = m_predictionBuffer->at(m_t1 - 1);
    auto& curr = m_predictionBuffer->at(m_t1);

    curr.pose = prev.pose;

    // median depth from last 3 frames
    for (int i = 0; i < 128 * 128; i++)
    {
        float a = m_predictionBuffer->at(m_t1 - 3).disps[i];
        float b = m_predictionBuffer->at(m_t1 - 2).disps[i];
        float c = m_predictionBuffer->at(m_t1 - 1).disps[i];

        float median = std::max(std::min(a, b),
                                std::min(std::max(a, b), c));

        curr.disps[i] = median;
    }
}
