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

#ifndef __MotionFilter__
#define __MotionFilter__

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <fstream>

// Ambarella CV28
#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

// WNC
#include "bounding_box.hpp" //TODO:
#include "dla_config.hpp"
#include "img_util.hpp" //TODO:
#include "dataStructures.h"
#include "logger.hpp"
#include "utils.hpp"
#include "yolov8_decoder.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// constexpr int MODEL_HEIGHT     = 288;
// constexpr int MODEL_WIDTH      = 512;
constexpr int VSLAM_MODEL_HEIGHT  = 328;
constexpr int VSLAM_MODEL_WIDTH   = 584;
// constexpr int FRAME_HEIGHT     = 288;
// constexpr int FRAME_WIDTH      = 512;
// constexpr int NUM_OBJ_CLASSES  = 2;
// constexpr int NUM_LANE_CLASSES = 3;
// constexpr int NUM_POSE_CLASSES = 1;

class MotionFilter
{
public:
    using WakeCallback = std::function<void()>;
    MotionFilter(Config_S* config, WakeCallback wakeFunc = nullptr);
    ~MotionFilter();

    // Multi-threading
    void runThread();
    void stopThread();
    void updateInputFrame(ea_tensor_t* imgTensor, int frameIdx);
    void notifyProcessingComplete();
    
    // Prediction
    bool getLastestPrediction(YOLOv8_Prediction& pred, int& frameIdx);

    // Utility Functions
    bool isInputBufferEmpty() const;
    bool isPredictionBufferEmpty() const;
    void updateTensorPath(const std::string& path);
    bool createDirectory(const std::string& path);
    bool directoryExists(const std::string& path);

    // Historical Mode //TODO:
    // bool run_sequential(cv::Mat& imgFrame, YOLOv8_Prediction& pred);

    // Debug
    void getDebugProfiles(float& inferenceTime, int& inputBufferSize, int& outBufferSize);

    // Inference
#if defined(CV28) || defined(CV28_SIMULATOR)
    std::deque<std::pair<int, ea_tensor_t*>> m_inputFrameBuffer;
#else
    std::deque<std::pair<int, cv::Mat>> m_inputFrameBuffer;
#endif

    // Thread Management
    bool m_bInferenced      = true;
    bool m_bProcessed       = true;
    bool m_threadTerminated = false;
    bool m_threadStarted    = false;
    bool m_bDone            = false;

    // Historical Mode
    int         m_inputMode          = DETECTION_MODE_LIVE; // ADAS_MODE_LIVE or ADAS_MODE_HISTORICAL
    std::string m_dbg_dateTime       = "";
    std::string m_dbg_rawImgsDirPath = "";

    // Debug
    int         m_saveRawImage       = 0;



#if defined(CV28) || defined(CV28_SIMULATOR)
    ea_tensor_t* cvmat_to_ea_tensor(const cv::Mat& img, ea_tensor_t* like_tensor);
    ea_tensor_t* nhwc_to_nchw(ea_tensor_t* nhwc_tensor);
#endif

private:

#if defined(CV28) || defined(CV28_SIMULATOR)
    bool _checkSavedTensor(int frameIdx);
    bool _loadInput(ea_tensor_t* imgTensor);
    int _preProcessingMemory(ea_tensor_t* imgTensor);
    bool _run(ea_tensor_t* imgTensor, int frameIdx);
    bool _runInferenceFunc();
    void _initModelIO();
    bool _releaseModel();
    bool _releaseInputTensor();
    bool _releaseOutputTensor();
    bool _releaseTensorBuffers();
#endif

#if defined(SAVE_OUTPUT_TENSOR)
    bool _saveOutputTensor(int frameIdx);
#endif

    // === Thread Management === //
    std::thread             m_threadInference;
    mutable std::mutex      m_pred_mutex;
    mutable std::mutex      m_mutex;
    std::condition_variable m_condition;
    WakeCallback            m_wakeFunc;

#if defined(CV28) || defined(CV28_SIMULATOR)
    char*       	m_ptrModelPath       = NULL;
    char*       	m_fnet_ptrModelPath  = NULL;
    char*       	m_cnet_ptrModelPath  = NULL;
    char*       	m_calibPath          = NULL;
    ea_net_t*   	m_model              = NULL;
    ea_net_t*   	m_fnet_model         = NULL;
    ea_net_t*   	m_cnet_model         = NULL;
    ea_tensor_t* 	m_img                = NULL; //TODO:
    ea_tensor_t* 	m_inputTensor        = NULL;
    ea_tensor_t* 	m_fnet_inputTensor   = NULL; // DROID-SLAM
    ea_tensor_t* 	m_cnet_inputTensor   = NULL; // DROID-SLAM
    std::vector<ea_tensor_t*> 	m_outputTensors;

    // DROID-SLAM, Alister add 2025-11-24
    std::vector<ea_tensor_t*> 	m_fnet_outputTensors;
    std::vector<ea_tensor_t*> 	m_cnet_outputTensors;
#endif

    // I/O information
    int m_inputChannel    = 0;
    int m_inputHeight     = 0;
    int m_inputWidth      = 0;
    std::string m_inputTensorName = "images";

    std::string m_fnet_inputTensorName = "images";
    std::string m_cnet_inputTensorName = "images";

    // DROID-SLAM I/O information, Alister add 2025-11-24
    // fnet I/O
    int m_fnet_inputChannel = 0;
    int m_fnet_inputHeight  = 0;
    int m_fnet_inputWidth   = 0;
    // cnet I/O
    int m_cnet_inputChannel = 0;
    int m_cnet_inputHeight  = 0;
    int m_cnet_inputWidth   = 0;

    // Number of BBox
    int m_numAnchorBox    = 0;

    // Number of keypoints
    int m_numKeypoints    = 17; 

    // Prediction Buffer
    std::deque<std::pair<int, YOLOv8_Prediction>> m_predictionBuffer;
    YOLOv8_Prediction m_pred;

    // Prediction Buffer for DROID-slam
    std::deque<std::pair<int, DROID_SLAM_Prediction>> m_slam_predictionBuffer;
    DROID_SLAM_Prediction m_slam_pred;
    std::vector<double> m_calib;


    std::vector<double> loadCalib(const std::string &path); 

    bool process_one_image( cv::Mat image,
                            FrameData& outFrame);

    FrameData m_frameData;


    // Buffer Size
    int m_boxBufferSize   = 0;
    int m_confBufferSize  = 0;
    int m_classBufferSize = 0;
    int m_kptsBufferSize  = 0;

    // DROID Slam Buffer Size, Alister modified 2025-11-24
    int m_netBufferSize    = 0;
    int m_inpBufferSize    = 0;
    int m_gmapBufferSize   = 0;

    // Buffer
    float* m_objBoxBuff;
    float* m_objConfBuff;
    float* m_objClsBuff;

    float* m_laneBoxBuff;
    float* m_laneConfBuff;
    float* m_laneClsBuff;

    float* m_poseBoxBuff;
    float* m_poseConfBuff;
    float* m_poseClsBuff;
    float* m_poseKptsBuff;

    // Driod-Slam Buffer, Alister add 2025-11-21
    float*  m_netBuff;
    float*  m_inpBuff;
    float*  m_gmapBuff;

    // Output Tensor List
    // output_names=['obj_box', 'obj_conf', 'obj_cls', 'lane_box', 'lane_conf', 'lane_cls', 'pose_box', 'pose_conf', 'pose_cls', 'pose_kpt']

    std::vector<std::string> m_outputTensorList = {
        // "dbox",  "conf",  "cls_id"              // Object Detection
        "obj_box", "obj_conf", "obj_cls",   // Object Detection
        "lane_box", "lane_conf", "lane_cls", // Lane Detection
        "pose_box", "pose_conf", "pose_cls", "pose_kpt" // Pose Detection
    };

    // DROID-Slam gmap output name
    std::vector<std::string> m_fnet_outputTensorList = {
        "fnet_features"
    };
    // DROID-Slam cnet output name
    std::vector<std::string> m_cnet_outputTensorList = {
        "net", "inp"
    };


    // Read Saved Tensor
    std::string m_tensorPath;

    // Debug
    float m_inferenceTime = 0.0f;
    bool m_estimateTime = false;
};

#ifdef __cplusplus
}
#endif

#endif

