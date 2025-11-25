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

#include "motion_filter.hpp"

MotionFilter::MotionFilter(Config_S *config, WakeCallback wakeFunc)
    : m_numAnchorBox((MODEL_WIDTH * MODEL_HEIGHT / 64) + (MODEL_WIDTH * MODEL_HEIGHT / 256)
                     + (MODEL_WIDTH * MODEL_HEIGHT / 1024)),
      m_boxBufferSize(4 * m_numAnchorBox),
      m_confBufferSize(m_numAnchorBox),
      m_classBufferSize(m_numAnchorBox),
	  m_kptsBufferSize(m_numKeypoints * 3 * m_numAnchorBox),
      m_estimateTime(config->stShowProcTimeConfig.AIModel),
	  m_netBufferSize(1 * 128 * (384 / 8) * (512 / 8)), // DROID_SLAM
	  m_inpBufferSize(1 * 128 * (384 / 8) * (512 / 8)), // DROID_SLAM
	  m_gmapBufferSize(1 * 128 * (384 / 8) * (512 / 8)) // DROID_SLAM

{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::syslog_logger_mt("yolov8", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
    auto logger = spdlog::stdout_color_mt("yolov8");
    logger->set_pattern("[%n] [%^%l%$] %v");
#endif

    logger->set_level(config->stDebugConfig.AIModel ? spdlog::level::debug : spdlog::level::info);

    // ==================================
    // (Ambarella CV28) Model Initialization
    // ==================================
#if defined(CV28) || defined(CV28_SIMULATOR)

    m_ptrModelPath    = const_cast<char *>(config->modelPath.c_str()); //TODO:

	// DROID net/inp/gmap model path, Alister modified 2025-11-24
	m_fnet_ptrModelPath    = const_cast<char *>(config->fnetModelPath.c_str()); //TODO:
	m_cnet_ptrModelPath    = const_cast<char *>(config->cnetModelPath.c_str()); //TODO:


	// DROID-SLAM Load calib file, Alister add 2025-11-24
	m_calibPath = const_cast<char *>(config->calibPath.c_str()); //TODO:

    m_calib = loadCalib(m_calibPath);
    if (m_calib.size() < 4) {
        logger->error("Calibration file must have at least 4 numbers.");
    }

		// Initialize network parameters
		ea_net_params_t net_params;
		memset(&net_params, 0 , sizeof(net_params));
		// Set GPU ID to -1 to use CPU
		net_params.acinf_gpu_id = -1;

		// Initialize DROID_SLAM network parameters,Alister add 2025-11-24
		ea_net_params_t cnet_params;
		memset(&cnet_params, 0 , sizeof(cnet_params));
		// Set GPU ID to -1 to use CPU
		cnet_params.acinf_gpu_id = -1;

		ea_net_params_t fnet_params;
		memset(&fnet_params, 0 , sizeof(fnet_params));
		// Set GPU ID to -1 to use CPU
		fnet_params.acinf_gpu_id = -1;

		// Create network instance  
		m_model 		= ea_net_new(&net_params);
		// Create DROID SLAM network instance, Alister add 2025-11-24
		m_cnet_model 	= ea_net_new(&cnet_params);
		m_fnet_model 	= ea_net_new(&fnet_params);

    if (m_model == NULL)
    {
			logger->error("Creating YOLOv8 model failed");
    }else{
		logger->info("Creating YOLOv8 model successful");
	}
		// RVAL_ASSERT(m_model != NULL); //TODO:

	// Creating DROID-SLANM model check, Alister add 2025-11-22
	if (m_fnet_model == NULL)
	{
		logger->error("Creating DROID SLAM fnet model failed");
	}else{
		logger->info("Creating DROID SLAM fnet model successful");
	}

	if (m_cnet_model == NULL)
	{
		logger->error("Creating DROID SLAM cnet model failed");
	}else{
		logger->info("Creating DROID SLAM cnet model successful");
	}

		m_img 					= NULL; //TODO:
		m_inputTensor 	= NULL;
		m_cnet_inputTensor = NULL; // DROID-SLAM
		m_fnet_inputTensor = NULL; // DROID-SLAM

		m_outputTensors 	 = std::vector<ea_tensor_t*>(m_outputTensorList.size()); //TODO:
		// DROID_SLAM 2025-11-24 Alister modified
		m_fnet_outputTensors = std::vector<ea_tensor_t*>(m_fnet_outputTensorList.size()); //TODO:
		m_cnet_outputTensors = std::vector<ea_tensor_t*>(m_cnet_outputTensorList.size()); //TODO:
		

		m_objBoxBuff  = new float[m_boxBufferSize]; //TODO:
		m_objConfBuff = new float[m_confBufferSize]; //TODO:
		m_objClsBuff  = new float[m_classBufferSize]; //TODO:

		// Allocate Lane Box Detection Buffers
		m_laneBoxBuff  = new float[m_boxBufferSize];
		m_laneConfBuff = new float[m_confBufferSize];
		m_laneClsBuff  = new float[m_classBufferSize];

		// Allocate Lane Point Detection Buffers
		m_poseBoxBuff  = new float[m_boxBufferSize];
		m_poseConfBuff = new float[m_confBufferSize];
		m_poseClsBuff  = new float[m_classBufferSize];
		m_poseKptsBuff = new float[m_kptsBufferSize];

		// DROID_SLAM buffer 2025-11-22 add
		m_netBuff  = new float[m_netBufferSize];
		m_inpBuff  = new float[m_inpBufferSize];
		m_gmapBuff = new float[m_gmapBufferSize];
#endif
    // ==================================

    m_wakeFunc = wakeFunc;

    // === Historical Feed === //
    m_saveRawImage = config->stDebugConfig.saveRawImages;
    m_inputMode    = config->HistoricalFeedModeConfig.inputMode;
    // m_visualMode   = config->HistoricalFeedModeConfig.visualizeMode; // TODO: Not implemented yet
    utils::getDateTime(m_dbg_dateTime);
    m_dbg_rawImgsDirPath = config->stDebugConfig.rawImgsDirPath + "/" + m_dbg_dateTime;

		// TODO:
    if (config->stDebugConfig.saveRawImages)
    {
			if (!directoryExists(m_dbg_rawImgsDirPath))
			{
				if (!createDirectory(m_dbg_rawImgsDirPath))
				{
						std::cerr << "---------------FUNC : HISTORICAL::HISTORICAL(ADAS_Config_S *config) "
											<< m_dbg_rawImgsDirPath << std::endl;
						std::cerr << "                  Failed to create directory: " << m_dbg_rawImgsDirPath << std::endl;
				}
			}
    }

    // Init Model Input/Output Tensor
    _initModelIO();
};

bool MotionFilter::_releaseModel()
{
	if (m_model)
	{
		ea_net_free(m_model);
		m_model = NULL;
	}
	
	return true;
}

bool MotionFilter::createDirectory(const std::string &path)
{
    return mkdir(path.c_str(), 0755) == 0; // Create directory with rwxr-xr-x permissions
}

bool MotionFilter::directoryExists(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        return false; // Directory doesn't exist
    }
    return (info.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

MotionFilter::~MotionFilter() // clear object memory
{
#ifdef SPDLOG_USE_SYSLOG
    spdlog::drop("yolov8");
#else
    spdlog::drop("yolov8");
#endif

	// ==================================
	// Ambarella CV28
	// ==================================
#if defined(CV28) || defined(CV28_SIMULATOR)

	_releaseInputTensor();
	_releaseOutputTensor();
	_releaseTensorBuffers();
	_releaseModel();

#endif
	// ==================================
};

// ============================================
//               Tensor Settings
// ============================================
void MotionFilter::_initModelIO()
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("yolov8");
#else
    auto logger = spdlog::get("yolov8");
#endif

	// ==================================
	// (Ambarella CV28) Create Model Output Buffers
	// ==================================
#if defined(CV28) || defined(CV28_SIMULATOR)

	int rval = EA_SUCCESS;
	logger->info("-------------------------------------------");
	logger->info("Configure Model Input/Output");


	// Configure model path
	logger->info("-------------------------------------------");
	logger->info("Load Model");
	logger->info("Model Path: {}", m_ptrModelPath);
	// DROID-SLAM
	logger->info("fnet Model Path: {}", m_fnet_ptrModelPath);
	logger->info("cnet Model Path: {}", m_cnet_ptrModelPath);

	// Check if model path exists before loading
	if (m_ptrModelPath == nullptr || strlen(m_ptrModelPath) == 0) {
		logger->error("Model path is null or empty");
		return;
	}
	
	// Check if file exists
	FILE* file = fopen(m_ptrModelPath, "r");
	if (file == nullptr) {
		logger->error("Model file does not exist at path: {}", m_ptrModelPath);
		return;
	}
	fclose(file);

	logger->info("yolov8 Model file exists, proceeding with loading");
	
	// cnet, Alister add 2025-11-24 
	FILE* file_cnet = fopen(m_cnet_ptrModelPath, "r");
	if (file_cnet == nullptr) {
		logger->error("cnet Model file does not exist at path: {}", m_cnet_ptrModelPath);
		return;
	}
	fclose(file_cnet);
	logger->info("cnet Model file exists, proceeding with loading");

	// fnet, Alister add 2025-11-24 
	FILE* file_fnet = fopen(m_fnet_ptrModelPath, "r");
	if (file_fnet == nullptr) {
		logger->error("fnet Model file does not exist at path: {}", m_fnet_ptrModelPath);
		return;
	}
	fclose(file_fnet);
	logger->info("fnet Model file exists, proceeding with loading");


	//--------------------
	//ea_net_config_output
	//--------------------
	// Configure input tensor
	logger->info("-------------------------------------------");
	logger->info("Configure Input Tensor");
	logger->info("Input Name: {}", m_inputTensorName);
	rval = ea_net_config_input(m_model, m_inputTensorName.c_str());

	logger->info("-------------------------------------------");
	logger->info("Configure Output Tensors");

	for (size_t i = 0; i < m_outputTensorList.size(); ++i) {
		logger->info("Output Name: {}", m_outputTensorList[i]);
		rval = ea_net_config_output(m_model, m_outputTensorList[i].c_str());
		// RVAL_ASSERT(rval == EA_SUCCESS);
	}
	// RVAL_ASSERT(rval == EA_SUCCESS);
	//------------------
	// DROID-SLAM fnet
	//-----------------
	logger->info("-------------------------------------------");
	logger->info("Configure fnet Input Tensor");
	logger->info("Input Name: {}", m_inputTensorName);
	rval = ea_net_config_input(m_fnet_model, m_fnet_inputTensorName.c_str());

	// DROID-SLAM , Alister 2025-11-24
	// logger->info("-------------------------------------------");
	logger->info("Configure Output Tensors");
	for (size_t i = 0; i < m_fnet_outputTensorList.size(); ++i) {
		logger->info("Output Name: {}", m_fnet_outputTensorList[i]);
		rval = ea_net_config_output(m_fnet_model, m_fnet_outputTensorList[i].c_str());
	}
	//------------------
	// DROID-SLAM cnet
	//------------------
	logger->info("-------------------------------------------");
	logger->info("Configure cnet Input Tensor");
	logger->info("Input Name: {}", m_inputTensorName);
	rval = ea_net_config_input(m_cnet_model, m_cnet_inputTensorName.c_str());

	// Configure output tensors
	// Configure output tensors

	// logger->info("-------------------------------------------");
	logger->info("Configure Output Tensors");
	for (size_t i = 0; i < m_cnet_outputTensorList.size(); ++i) {
		logger->info("Output Name: {}", m_cnet_outputTensorList[i]);
		rval = ea_net_config_output(m_cnet_model, m_cnet_outputTensorList[i].c_str());
	}
	//---------------------------------------------------------------------------------------------------------


	//-------------
	// ea_net_load
	//--------------
	rval = ea_net_load(m_model, EA_NET_LOAD_FILE, (void *)m_ptrModelPath, 1/*max_batch*/);
	logger->info("ea_net_load yolov8 model finished");
	//DROID_SLAM, Alister modified 2025-11-24
	
	rval = ea_net_load(m_fnet_model , EA_NET_LOAD_FILE, (void *)m_fnet_ptrModelPath, 1/*max_batch*/);
	logger->info("ea_net_load fnet model finished");

	rval = ea_net_load(m_cnet_model , EA_NET_LOAD_FILE, (void *)m_cnet_ptrModelPath, 1/*max_batch*/);
	logger->info("ea_net_load cnet model finished");
	// RVAL_ASSERT(rval == EA_SUCCESS);	

	
	//------------------
	// ea_net_input
	//------------------

	// Get input tensor
	logger->info("-------------------------------------------");
	logger->info("Create Model Input Tensor");
	m_inputTensor   		= ea_net_input(m_model, m_inputTensorName.c_str());
	logger->info("m_inputTensor ptr = {}", (void*)m_inputTensor);
	logger->info("yolov8 m_inputTensor finished");
	
	// RVAL_ASSERT(m_inputTensor != NULL);
	m_inputHeight 	= ea_tensor_shape(m_inputTensor)[EA_H];
	m_inputWidth  	= ea_tensor_shape(m_inputTensor)[EA_W];
	m_inputChannel 	= ea_tensor_shape(m_inputTensor)[EA_C];
	logger->info("Input H: {}", m_inputHeight);
	logger->info("Input W: {}", m_inputWidth);
	logger->info("Input C: {}", m_inputChannel);


	// DROID-SLAM cnet, Alister modified 2025-11-24
	m_cnet_inputTensor    = ea_net_input(m_cnet_model , m_cnet_inputTensorName.c_str());
	logger->info("m_cnet_inputTensor ptr = {}", (void*)m_cnet_inputTensor);
	if (!m_cnet_inputTensor) {
		logger->error("ea_net_input returned NULL for cnet");
		return;
	}
	logger->info("DROID-SLAM m_cnet_inputTensor finished");
	// DROID_SLAM cnet input size, Alister add 2025-11-24
	m_cnet_inputHeight 		= ea_tensor_shape(m_cnet_inputTensor)[EA_H];
	m_cnet_inputWidth  		= ea_tensor_shape(m_cnet_inputTensor)[EA_W];
	m_cnet_inputChannel 	= ea_tensor_shape(m_cnet_inputTensor)[EA_C];
	logger->info("cnet Input H: {}", m_cnet_inputHeight);
	logger->info("cnet Input W: {}", m_cnet_inputWidth);
	logger->info("cnet Input C: {}", m_cnet_inputChannel);

	// DROID-SLAM fnet, Alister modified 2025-11-24
	m_fnet_inputTensor    	= ea_net_input(m_fnet_model , m_fnet_inputTensorName.c_str());
	logger->info("m_fnet_inputTensor ptr = {}", (void*)m_fnet_inputTensor);
	if (!m_fnet_inputTensor) {
		logger->error("ea_net_input returned NULL for cnet");
		return;
	}
	logger->info("DROID-SLAM m_fnet_inputTensor finished");

	// DROID_SLAM fnet input size, Alister add 2025-11-24
	m_fnet_inputHeight 		= ea_tensor_shape(m_fnet_inputTensor)[EA_H];
	m_fnet_inputWidth  		= ea_tensor_shape(m_fnet_inputTensor)[EA_W];
	m_fnet_inputChannel 	= ea_tensor_shape(m_fnet_inputTensor)[EA_C];
	logger->info("fnet Input H: {}", m_fnet_inputHeight);
	logger->info("fnet Input W: {}", m_fnet_inputWidth);
	logger->info("fnet Input C: {}", m_fnet_inputChannel);




	// Get output tensors
	// Create a buffer to store image data
	logger->info("-------------------------------------------");
	logger->info("Create YOLOV8 Model Output Tensors");
	m_outputTensors[0] = ea_net_output_by_index(m_model, 0);
	m_outputTensors[1] = ea_net_output_by_index(m_model, 1);
	m_outputTensors[2] = ea_net_output_by_index(m_model, 2);

	m_outputTensors[3] = ea_net_output_by_index(m_model, 3);
	m_outputTensors[4] = ea_net_output_by_index(m_model, 4);
	m_outputTensors[5] = ea_net_output_by_index(m_model, 5);

	m_outputTensors[6] = ea_net_output_by_index(m_model, 6);
	m_outputTensors[7] = ea_net_output_by_index(m_model, 7);
	m_outputTensors[8] = ea_net_output_by_index(m_model, 8);
	m_outputTensors[9] = ea_net_output_by_index(m_model, 9);	

	for (size_t i=0; i<ea_net_output_num(m_model); i++)
	{
		const char* tensorName = static_cast<const char*>(ea_net_output_name(m_model, i));
		const size_t* tensorShape = static_cast<const size_t*>(ea_tensor_shape(ea_net_output_by_index(m_model, i)));
		size_t tensorSize = static_cast<size_t>(ea_tensor_size(ea_net_output_by_index(m_model, i)));
		
		std::string shapeStr = std::to_string(tensorShape[0]) + "x" + 
							   std::to_string(tensorShape[1]) + "x" + 
							   std::to_string(tensorShape[2]) + "x" + 
							   std::to_string(tensorShape[3]);
		
		logger->info("Output Tensor Name: {}", tensorName);
		logger->info("Output Tensor Shape: ({})", shapeStr);
		logger->info("Output Tensor Size: {}", tensorSize);
	}

	// DROID_SLAM, Alister modified 2025-11-24
	logger->info("-------------------------------------------");
	logger->info("Create DROID-SLAM Model Output Tensors");
	logger->info("-------------------------------------------");
	logger->info("fnet Model Output Tensors");
	logger->info("-------------------------------------------");
	m_fnet_outputTensors[0]  = ea_net_output_by_index(m_fnet_model , 0);
	for (size_t i=0; i<ea_net_output_num(m_fnet_model); i++)
	{
		const char* tensorName2 = static_cast<const char*>(ea_net_output_name(m_fnet_model, i));
		const size_t* tensorShape2 = static_cast<const size_t*>(ea_tensor_shape(ea_net_output_by_index(m_fnet_model, i)));
		size_t tensorSize2 = static_cast<size_t>(ea_tensor_size(ea_net_output_by_index(m_fnet_model, i)));
		
		std::string shapeStr2 = std::to_string(tensorShape2[0]) + "x" + 
							   std::to_string(tensorShape2[1]) + "x" + 
							   std::to_string(tensorShape2[2]) + "x" + 
							   std::to_string(tensorShape2[3]);
		
		logger->info("Output Tensor Name: {}", tensorName2);
		logger->info("Output Tensor Shape: ({})", shapeStr2);
		logger->info("Output Tensor Size: {}", tensorSize2);
	}

	logger->info("-------------------------------------------");
	logger->info("cnet Model Output Tensors");
	logger->info("-------------------------------------------");
	m_cnet_outputTensors[0]  = ea_net_output_by_index(m_cnet_model , 0);
	m_cnet_outputTensors[1]  = ea_net_output_by_index(m_cnet_model , 1);
	for (size_t i=0; i<ea_net_output_num(m_cnet_model); i++)
	{
		const char* tensorName3 = static_cast<const char*>(ea_net_output_name(m_cnet_model, i));
		const size_t* tensorShape3 = static_cast<const size_t*>(ea_tensor_shape(ea_net_output_by_index(m_cnet_model, i)));
		size_t tensorSize3 = static_cast<size_t>(ea_tensor_size(ea_net_output_by_index(m_cnet_model, i)));
		
		std::string shapeStr3 = std::to_string(tensorShape3[0]) + "x" + 
							   std::to_string(tensorShape3[1]) + "x" + 
							   std::to_string(tensorShape3[2]) + "x" + 
							   std::to_string(tensorShape3[3]);
		
		logger->info("Output Tensor Name: {}", tensorName3);
		logger->info("Output Tensor Shape: ({})", shapeStr3);
		logger->info("Output Tensor Size: {}", tensorSize3);
	}

#endif
	// ==================================
	return;
}


std::vector<double> MotionFilter::loadCalib(const std::string &path) 
{
    std::vector<double> calib;
    std::ifstream file(path);
    double v;
    while (file >> v) calib.push_back(v);
    return calib;
}

bool MotionFilter::process_one_image(
    cv::Mat image,
    FrameData& outFrame)
{

    double fx = m_calib[0];
    double fy = m_calib[1];
    double cx = m_calib[2];
    double cy = m_calib[3];

    // Build intrinsic matrix
    cv::Mat K = (cv::Mat_<double>(3,3) <<
					fx, 0,  cx,
					0,  fy, cy,
					0,  0,  1
				);

    // Undistort if calibration has extra values
    if (m_calib.size() > 4) {
        cv::Mat distCoeffs(m_calib.size() - 4, 1, CV_64F);
        for (int i = 4; i < m_calib.size(); ++i)
            distCoeffs.at<double>(i - 4) = m_calib[i];

        cv::Mat undistorted;
        cv::undistort(image, undistorted, K, distCoeffs);
        image = undistorted;
    }

    int h0 = image.rows;
    int w0 = image.cols;

    // Scale to match Python code (area = 384*512)
    float scale = std::sqrt((384.0 * 512.0) / (h0 * w0));
    int h1 = int(h0 * scale);
    int w1 = int(w0 * scale);

    cv::resize(image, image, cv::Size(w1, h1));

    // Crop to multiples of 8
    h1 = h1 - (h1 % 8);
    w1 = w1 - (w1 % 8);
    image = image(cv::Rect(0, 0, w1, h1));

    // Convert to CHW float32 (flattened 3 x H*W)
    // cv::Mat chw(3, h1 * w1, CV_32FC1);
    // for (int c = 0; c < 3; ++c) {
    //     for (int y = 0; y < h1; ++y) {
    //         for (int x = 0; x < w1; ++x) {
    //             cv::Vec3b px = image.at<cv::Vec3b>(y, x);
    //             chw.at<float>(c, y * w1 + x) = float(px[c]);
    //         }
    //     }
    // }

    // Adjust intrinsics (same logic as Python)
    outFrame.intr = {
        float(fx * (float(w1) / w0)),
        float(fy * (float(h1) / h0)),
        float(cx * (float(w1) / w0)),
        float(cy * (float(h1) / h0))
    };

    outFrame.image = image;
    outFrame.h = h1;
    outFrame.w = w1;

    return true;
}


// =================================================================================================
// Ambarella CV28 Tensor Functions
// =================================================================================================
#if defined(CV28) || defined(CV28_SIMULATOR)
bool MotionFilter::_checkSavedTensor(int frameIdx)
{
	auto logger = spdlog::get("yolov8");

	auto time_0 = std::chrono::high_resolution_clock::now();

	// Define file paths for each tensor
	std::string objBoxFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor0.bin";
	std::string objConfFilePath     = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor1.bin";
	std::string objClsFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor2.bin";

	std::string laneBoxFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor3.bin";
	std::string laneConfFilePath     = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor4.bin";
	std::string laneClsFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor5.bin";

	std::string poseBoxFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor6.bin";
	std::string poseConfFilePath     = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor7.bin";
	std::string poseClsFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor8.bin";
	std::string poseKptsFilePath     = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor9.bin";

	// Function to load tensor data from a binary file
	auto loadTensorFromBinaryFile = [](const std::string& filePath, float* buffer, size_t size) {
		std::ifstream inFile(filePath, std::ios::binary);
		if (inFile.is_open()) {
				inFile.read(reinterpret_cast<char*>(buffer), size * sizeof(float));
				inFile.close();
				return true;
		} else {
				std::cerr << "Failed to open file: " << filePath << std::endl;
				return false;
		}
	};

	// Check if tensor files exist and load them if they do
	// Usage : std::ifstream fileName(filePath, mode);
	// The mode std::ios::binary, indicating the files are read as binary data.
	// Declare file stream variables
	std::ifstream objBoxFile(objBoxFilePath, std::ios::binary);
	std::ifstream objConfFile(objConfFilePath, std::ios::binary);
	std::ifstream objClsFile(objClsFilePath, std::ios::binary);

	std::ifstream laneBoxFile(laneBoxFilePath, std::ios::binary);
	std::ifstream laneConfFile(laneConfFilePath, std::ios::binary);
	std::ifstream laneClsFile(laneClsFilePath, std::ios::binary);

	std::ifstream poseBoxFile(poseBoxFilePath, std::ios::binary);
	std::ifstream poseConfFile(poseConfFilePath, std::ios::binary);
	std::ifstream poseClsFile(poseClsFilePath, std::ios::binary);
	std::ifstream poseKptsFile(poseKptsFilePath, std::ios::binary);
	
	if (objBoxFile  && loadTensorFromBinaryFile(objBoxFilePath, m_objBoxBuff, m_boxBufferSize) &&
		objConfFile && loadTensorFromBinaryFile(objConfFilePath, m_objConfBuff, m_confBufferSize) &&
		objClsFile  && loadTensorFromBinaryFile(objClsFilePath, m_objClsBuff, m_classBufferSize) &&

		laneBoxFile  && loadTensorFromBinaryFile(laneBoxFilePath, m_laneBoxBuff, m_boxBufferSize) &&
		laneConfFile && loadTensorFromBinaryFile(laneConfFilePath, m_laneConfBuff, m_confBufferSize) &&
		laneClsFile  && loadTensorFromBinaryFile(laneClsFilePath, m_laneClsBuff, m_classBufferSize) &&

		poseBoxFile  && loadTensorFromBinaryFile(poseBoxFilePath, m_poseBoxBuff, m_boxBufferSize) &&
		poseConfFile && loadTensorFromBinaryFile(poseConfFilePath, m_poseConfBuff, m_confBufferSize) &&
		poseClsFile  && loadTensorFromBinaryFile(poseClsFilePath, m_poseClsBuff, m_classBufferSize) &&
		poseKptsFile && loadTensorFromBinaryFile(poseKptsFilePath, m_poseKptsBuff, m_kptsBufferSize))
	{
		// logger->info("Loaded tensors from binary files for frame index: {}", frameIdx);

		if (m_estimateTime)
		{
			auto time_1 = std::chrono::high_resolution_clock::now();
			logger->info("[_checkSavedTensor]: {} ms",
				std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
		}

		return true; // true means the tensors are already saved in the specific path
	}

	return false;
}

#if defined(SAVE_OUTPUT_TENSOR)
bool MotionFilter::_saveOutputTensor(int frameIdx)
{
	auto logger = spdlog::get("yolov8");

	std::string objBoxFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor0.bin";
	std::string objConfFilePath     = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor1.bin";
	std::string objClsFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor2.bin";

	std::string laneBoxFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor3.bin";
	std::string laneConfFilePath     = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor4.bin";
	std::string laneClsFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor5.bin";

	std::string poseBoxFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor6.bin";
	std::string poseConfFilePath     = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor7.bin";
	std::string poseClsFilePath      = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor8.bin";
	std::string poseKptsFilePath     = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor9.bin";

	// DROID-SLAM, Alister add 2025-11-24
	std::string netFilePath      	= m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor10.bin";
	std::string inpFilePath      	= m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor11.bin";
	std::string gmapFilePath      	= m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor12.bin";



	logger->debug("========================================");
	logger->debug("Model Frame Index: {}", frameIdx);
	logger->debug("Out Buffer Size: {}", m_predictionBuffer.size());
	logger->debug("========================================");

	// Save tensors to binary files
	auto saveTensorToBinaryFile = [](const std::string& filePath, float* buffer, size_t size) {
		std::ofstream outFile(filePath, std::ios::binary);
		if (outFile.is_open()) {
				outFile.write(reinterpret_cast<char*>(buffer), size * sizeof(float));
				outFile.close();
		} else {
				std::cerr << "Failed to open file: " << filePath << std::endl;
		}
	};
	// Save each tensor to its corresponding binary file
	saveTensorToBinaryFile(objBoxFilePath, 	m_pred.objBoxBuff, 	m_boxBufferSize);
	saveTensorToBinaryFile(objConfFilePath, m_pred.objConfBuff, m_confBufferSize);
	saveTensorToBinaryFile(objClsFilePath, 	m_pred.objClsBuff, 	m_classBufferSize);

	saveTensorToBinaryFile(laneBoxFilePath, m_pred.laneBoxBuff, 	m_boxBufferSize);
	saveTensorToBinaryFile(laneConfFilePath,m_pred.laneConfBuff, m_confBufferSize);
	saveTensorToBinaryFile(laneClsFilePath, m_pred.laneClsBuff, 	m_classBufferSize);

	saveTensorToBinaryFile(poseBoxFilePath, 	m_pred.poseBoxBuff,  m_boxBufferSize);
	saveTensorToBinaryFile(poseConfFilePath,    m_pred.poseConfBuff, m_confBufferSize);
	saveTensorToBinaryFile(poseClsFilePath, 	m_pred.poseClsBuff,  m_classBufferSize);
	saveTensorToBinaryFile(poseKptsFilePath, 	m_pred.poseKptsBuff, m_kptsBufferSize);


	// DROID-SLAM, Alister add 2025-11-24
	saveTensorToBinaryFile(netFilePath, 	m_slam_pred.netBuff,   m_netBufferSize);
	saveTensorToBinaryFile(inpFilePath, 	m_slam_pred.inpBuff,   m_inpBufferSize);
	saveTensorToBinaryFile(gmapFilePath, 	m_slam_pred.gmapBuff,  m_gmapBufferSize);

	return true;
}
#endif

bool MotionFilter::_releaseInputTensor()
{
	if (m_inputTensor)
	{
		m_inputTensor = nullptr;
	}
	return true;
}

bool MotionFilter::_releaseOutputTensor()
{
	for (size_t i=0; i<m_outputTensorList.size(); i++)
	{
		if (m_outputTensors[i])
		{
			m_outputTensors[i] = nullptr;
		}
	}
	return true;
}

bool MotionFilter::_releaseTensorBuffers()
{
	// Release Output Buffers
	delete[] m_objBoxBuff;   // Use delete[]
	delete[] m_objConfBuff; // Use delete[]
	delete[] m_objClsBuff;  // Use delete[]

	delete[] m_laneBoxBuff;   // Use delete[]
	delete[] m_laneConfBuff; // Use delete[]
	delete[] m_laneClsBuff;  // Use delete[]

	delete[] m_poseBoxBuff;   // Use delete[]
	delete[] m_poseConfBuff; // Use delete[]
	delete[] m_poseClsBuff;  // Use delete[]
	delete[] m_poseKptsBuff;  // Use delete[]

	// DROID-SLAM, Alister add 2025-11-24
	delete[] m_netBuff;
	delete[] m_inpBuff;
	delete[] m_gmapBuff;

	m_objBoxBuff   = nullptr;
	m_objConfBuff  = nullptr;
	m_objClsBuff   = nullptr;

	m_laneBoxBuff   = nullptr;
	m_laneConfBuff  = nullptr;
	m_laneClsBuff   = nullptr;

	m_poseBoxBuff   = nullptr;
	m_poseConfBuff  = nullptr;
	m_poseClsBuff   = nullptr;
	m_poseKptsBuff  = nullptr;

	// DROID-SLAM, Alister add 2025-11-24
	m_netBuff 		= nullptr;
	m_inpBuff		= nullptr;
	m_gmapBuff		= nullptr;

	return true;
}
#endif



ea_tensor_t* MotionFilter::cvmat_to_ea_tensor(const cv::Mat& img)
{
    int H = img.rows;
    int W = img.cols;
    int C = img.channels();

    // EA wants size_t for shape array
    size_t shape[3] = { (size_t)W, (size_t)H, (size_t)C };

    // dtype = EA_U8
    // shape = pointer to size_t array
    // pitch = 0 (default, contiguous)
    ea_tensor_t* tensor = ea_tensor_new(EA_U8, shape, 0);

    // Copy image data
    uint8_t* dst = (uint8_t*)ea_tensor_data(tensor);
    std::memcpy(dst, img.data, img.total() * img.elemSize());

    return tensor;
}


// =================================================================================================

// =================================================================================================
// Inference Entrypoint
// =================================================================================================
bool MotionFilter::_run(ea_tensor_t* imgTensor, int frameIdx)
{
#ifdef SPDLOG_USE_SYSLOG
	auto logger = spdlog::get("yolov8");
#else
	auto logger = spdlog::get("yolov8");
#endif

	auto time_0 = std::chrono::high_resolution_clock::now();
	auto time_1 = std::chrono::time_point<std::chrono::high_resolution_clock>{};
	auto time_2 = std::chrono::high_resolution_clock::now();

	logger->debug(" =========== Model Frame Index: {} ===========", frameIdx);
	logger->debug(" =========== Buffer Size: {} ===========", m_inputFrameBuffer.size());

	if (m_estimateTime)
	{
		logger->info("[AI Processing Time]");
		logger->info("-----------------------------------------");
	}

	// ==================================
	// Ambarella CV28 Inference
	// ==================================
#if defined(CV28) || defined(CV28_SIMULATOR)

	int rval = EA_SUCCESS;

	m_bProcessed = false;
	std::unique_lock<std::mutex> pred_lock(m_pred_mutex);

	// STEP 1: load input tensor
	// if (!_loadInput(imgTensor))
	// {
	// 	logger->error("Load Input Data Failed");
	// 	return false;
	// }
	
	cv::Mat img = imgUtil::convertTensorToMat(imgTensor);
	logger->info("[DROID-SLAM] convertTensorToMat finished");

	// DROID-SLAM process one image of cv::Mat format, TODO: Alister add 2025-11-24
	process_one_image(img, m_frameData);
	logger->info("[DROID-SLAM] process_one_image finished");


	// DROID-SLAM, convert cv::Mat into  ea_tensor* , Alister add 2025-11-24
	// cv::Mat processedImg =  m_frameData.image;	
	// ea_tensor_t* tensorImg = cvmat_to_ea_tensor(processedImg);
	// logger->info("[DROID-SLAM] cvmat_to_ea_tensor finished");

	if (!_loadInput(imgTensor))
	{
		logger->error("Load Input Data Failed");
		return false;
	}


	// STEP 2: inference
	// STEP 2-1: check if the tensors are already saved, pass the inference
	if (_checkSavedTensor(frameIdx))
	{
		// Allocate memory for all output tensors
		// Object Detection Buffers
		m_pred.objBoxBuff   = new float[m_boxBufferSize];
		m_pred.objConfBuff  = new float[m_confBufferSize];
		m_pred.objClsBuff   = new float[m_classBufferSize];
		// Lane Detection Buffers
		m_pred.laneBoxBuff   = new float[m_boxBufferSize];
		m_pred.laneConfBuff  = new float[m_confBufferSize];
		m_pred.laneClsBuff   = new float[m_classBufferSize];
		// Pose Detection Buffers
		m_pred.poseBoxBuff   = new float[m_boxBufferSize];
		m_pred.poseConfBuff  = new float[m_confBufferSize];
		m_pred.poseClsBuff   = new float[m_classBufferSize];
		m_pred.poseKptsBuff  = new float[m_kptsBufferSize];

		// DROID-SLAM Buffers, Alister add 2025-11-24
		m_slam_pred.netBuff  = new float[m_netBufferSize];
		m_slam_pred.inpBuff  = new float[m_inpBufferSize];
		m_slam_pred.gmapBuff = new float[m_gmapBufferSize];


		// Copy output tensors to prediction buffers
		std::memcpy(m_pred.objBoxBuff,   (float *)m_objBoxBuff, m_boxBufferSize * sizeof(float));
		std::memcpy(m_pred.objConfBuff,  (float *)m_objConfBuff, m_confBufferSize * sizeof(float));
		std::memcpy(m_pred.objClsBuff,   (float *)m_objClsBuff, m_classBufferSize * sizeof(float));

		std::memcpy(m_pred.laneBoxBuff,   (float *)m_laneBoxBuff, m_boxBufferSize * sizeof(float));
		std::memcpy(m_pred.laneConfBuff,  (float *)m_laneConfBuff, m_confBufferSize * sizeof(float));
		std::memcpy(m_pred.laneClsBuff,   (float *)m_laneClsBuff, m_classBufferSize * sizeof(float));

		std::memcpy(m_pred.poseBoxBuff,   (float *)m_poseBoxBuff, m_boxBufferSize * sizeof(float));
		std::memcpy(m_pred.poseConfBuff,  (float *)m_poseConfBuff, m_confBufferSize * sizeof(float));
		std::memcpy(m_pred.poseClsBuff,   (float *)m_poseClsBuff, m_classBufferSize * sizeof(float));
		std::memcpy(m_pred.poseKptsBuff,  (float *)m_poseKptsBuff, m_kptsBufferSize * sizeof(float));

		// DROID-SLAM, Copy output tensors to prediction buffers, Alister add 2025-11-24
		std::memcpy(m_slam_pred.netBuff,  (float *)m_netBuff  , m_netBufferSize * sizeof(float));
		std::memcpy(m_slam_pred.inpBuff,  (float *)m_inpBuff  , m_inpBufferSize * sizeof(float));
		std::memcpy(m_slam_pred.gmapBuff, (float *)m_gmapBuff , m_gmapBufferSize * sizeof(float));
	}
	else
	{
		// STEP 2-2: run inference using Ambarella's eazyai library
		if (m_estimateTime)
			time_1 = std::chrono::high_resolution_clock::now();

		if (EA_SUCCESS != ea_net_forward(m_model, 1))
		{
			_releaseInputTensor();
			_releaseOutputTensor();
			_releaseTensorBuffers();
			_releaseModel();
		}
		else
		{	logger->info("----------------------------------------------------------------");
			logger->info("ea_net_forward m_model finished");
			// TODO: Not implemented yet
			// if (m_saveRawImage || m_visualMode == 0)
			// {
			// 	//------------- Below save raw image into buffer-------------------------------
			// 	// cv::Mat history_Img = imgFrame.clone();
			// 	// m_historical->putInputFrameToBuffer(frameIdx, history_Img);
			// 	std::cout << "Save input tensor to image is not implemented yet" << std::endl;
			// }


			// Object Detection
			// Sync output tensors between VP and CPU
#if defined(CV28)
			rval = ea_tensor_sync_cache(m_outputTensors[0], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 0");
			}
			rval = ea_tensor_sync_cache(m_outputTensors[1], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 1");
			}
			rval = ea_tensor_sync_cache(m_outputTensors[2], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 2");
			}

			rval = ea_tensor_sync_cache(m_outputTensors[3], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 3");
			}
			rval = ea_tensor_sync_cache(m_outputTensors[4], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 4");
			}
			rval = ea_tensor_sync_cache(m_outputTensors[5], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 5");
			}


			rval = ea_tensor_sync_cache(m_outputTensors[6], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 6");
			}
			rval = ea_tensor_sync_cache(m_outputTensors[7], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 7");
			}
			rval = ea_tensor_sync_cache(m_outputTensors[8], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 8");
			}
			rval = ea_tensor_sync_cache(m_outputTensors[9], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 9");
			}
#endif

			m_outputTensors[0]	= ea_net_output_by_index(m_model, 0);
			m_outputTensors[1]	= ea_net_output_by_index(m_model, 1);
			m_outputTensors[2]	= ea_net_output_by_index(m_model, 2);

			m_outputTensors[3]	= ea_net_output_by_index(m_model, 3);
			m_outputTensors[4]	= ea_net_output_by_index(m_model, 4);
			m_outputTensors[5]	= ea_net_output_by_index(m_model, 5);

			m_outputTensors[6]	= ea_net_output_by_index(m_model, 6);
			m_outputTensors[7]	= ea_net_output_by_index(m_model, 7);
			m_outputTensors[8]	= ea_net_output_by_index(m_model, 8);
			m_outputTensors[9]	= ea_net_output_by_index(m_model, 9);

			// Allocate memory for all output tensors
			// Object Detection Buffers
			m_pred.objBoxBuff   = new float[m_boxBufferSize];
			m_pred.objConfBuff  = new float[m_confBufferSize];
			m_pred.objClsBuff   = new float[m_classBufferSize];
			// Lane Detection Buffers
			m_pred.laneBoxBuff   = new float[m_boxBufferSize];
			m_pred.laneConfBuff  = new float[m_confBufferSize];
			m_pred.laneClsBuff   = new float[m_classBufferSize];
			// Pose Detection Buffers
			m_pred.poseBoxBuff   = new float[m_boxBufferSize];
			m_pred.poseConfBuff  = new float[m_confBufferSize];
			m_pred.poseClsBuff   = new float[m_classBufferSize];
			m_pred.poseKptsBuff  = new float[m_kptsBufferSize];


			// Copy output tensors to prediction buffers
			std::memcpy(m_pred.objBoxBuff,   (float *)ea_tensor_data(m_outputTensors[0]), m_boxBufferSize * sizeof(float));
			std::memcpy(m_pred.objConfBuff,  (float *)ea_tensor_data(m_outputTensors[1]), m_confBufferSize * sizeof(float));
			std::memcpy(m_pred.objClsBuff,   (float *)ea_tensor_data(m_outputTensors[2]), m_classBufferSize * sizeof(float));

			std::memcpy(m_pred.laneBoxBuff,   (float *)ea_tensor_data(m_outputTensors[3]), m_boxBufferSize * sizeof(float));
			std::memcpy(m_pred.laneConfBuff,  (float *)ea_tensor_data(m_outputTensors[4]), m_confBufferSize * sizeof(float));
			std::memcpy(m_pred.laneClsBuff,   (float *)ea_tensor_data(m_outputTensors[5]), m_classBufferSize * sizeof(float));

			std::memcpy(m_pred.poseBoxBuff,   (float *)ea_tensor_data(m_outputTensors[6]), m_boxBufferSize * sizeof(float));
			std::memcpy(m_pred.poseConfBuff,  (float *)ea_tensor_data(m_outputTensors[7]), m_confBufferSize * sizeof(float));
			std::memcpy(m_pred.poseClsBuff,   (float *)ea_tensor_data(m_outputTensors[8]), m_classBufferSize * sizeof(float));
			std::memcpy(m_pred.poseKptsBuff,  (float *)ea_tensor_data(m_outputTensors[9]), m_kptsBufferSize * sizeof(float));
			logger->info("memcpy m_pred.xxxBuff (YOLOV8) finished");

		// DROID-SLAM: run inference using Ambarella's eazyai library
		// fnet
		if (EA_SUCCESS != ea_net_forward(m_fnet_model, 1))
		{	
			// TODO, this is not correct
			// _releaseInputTensor();
			// _releaseOutputTensor();
			// _releaseTensorBuffers();
			// _releaseModel();
		}
		else{
			logger->info("----------------------------------------------------------------");
			logger->info("ea_net_forward m_fnet_model finished");
			// DROID-SLAM
			// Sync output tensors between VP and CPU
#if defined(CV28)
			rval = ea_tensor_sync_cache(m_fnet_outputTensors[0], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 0");
			}
#endif
			m_fnet_outputTensors[0]	= ea_net_output_by_index(m_fnet_model, 0);
			// Allocate memory for all output tensors
			m_slam_pred.gmapBuff   = new float[m_gmapBufferSize]; 
			// Copy output tensors to prediction buffers
			std::memcpy(m_slam_pred.gmapBuff,   (float *)ea_tensor_data(m_fnet_outputTensors[0]), m_gmapBufferSize * sizeof(float));

			logger->info("memcpy m_slam_pred.gmapBuff finished");
		}
		
		// cnet
		if (EA_SUCCESS != ea_net_forward(m_cnet_model, 1))
		{	
			// TODO, this is not correct
			// _releaseInputTensor();
			// _releaseOutputTensor();
			// _releaseTensorBuffers();
			// _releaseModel();
			logger->error("ea_net_forward m_cnet_model failed");
		}
		else // DROID-SLAM, Alister add 2025-11-24
		{	
			logger->info("----------------------------------------------------------------");
			logger->info("ea_net_forward m_cnet_model finished");
#if defined(CV28)
			rval = ea_tensor_sync_cache(m_cnet_outputTensors[0], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 0");
			}
			else{
				logger->info("ea_tensor_sync_cache cnet tensor 0 finished");
			}
			rval = ea_tensor_sync_cache(m_cnet_outputTensors[1], EA_VP, EA_CPU);
			if (rval != EA_SUCCESS)
			{
				logger->error("Failed to sync output tensor 1");
			}else{
				logger->info("ea_tensor_sync_cache cnet tensor 1 finished");
			}
#endif
			m_cnet_outputTensors[0]	= ea_net_output_by_index(m_cnet_model, 0);
			m_cnet_outputTensors[1]	= ea_net_output_by_index(m_cnet_model, 1);
			// Allocate memory for all output tensors
			m_slam_pred.netBuff   = new float[m_netBufferSize]; 
			m_slam_pred.inpBuff   = new float[m_inpBufferSize]; 
			// Copy output tensors to prediction buffers
			std::memcpy(m_slam_pred.netBuff,   (float *)ea_tensor_data(m_cnet_outputTensors[0]), m_netBufferSize * sizeof(float));
			std::memcpy(m_slam_pred.inpBuff,   (float *)ea_tensor_data(m_cnet_outputTensors[1]), m_inpBufferSize * sizeof(float));
			logger->info("memcpy m_slam_pred.netBuff & m_slam_pred.inpBuff finished");
		}

#if defined(SAVE_OUTPUT_TENSOR)
			_saveOutputTensor(frameIdx);
#endif
		}
	}

	// Save raw image to buffer
	m_pred.img = img;	
	std::pair<int, YOLOv8_Prediction> pair = std::make_pair(frameIdx, m_pred);
	m_predictionBuffer.push_back(pair);
	logger->info("m_predictionBuffer push back pair finished");

	// DROID-SLAM, Alister add 2025-11-24
	// m_slam_pred.img = m_frameData.image;
	// m_slam_pred.intrinsics = m_frameData.intr;
	// m_slam_pred.ht = m_frameData.h;
	// m_slam_pred.wd = m_frameData.w;
	// // DROID-SLAM initialize more paramter, Alister add 2025-11-24
	// // TODO, add intrinsic, tstamp, index, dsip, disp_sen, etc.

	std::pair<int, DROID_SLAM_Prediction> slam_pair = std::make_pair(frameIdx, m_slam_pred);
	m_slam_predictionBuffer.push_back(slam_pair);
	logger->info("m_slam_predictionBuffer push back slam_pair finished");

	pred_lock.unlock();
	m_bProcessed = true;





	logger->debug(" ============= Model Frame Index: {} =============", frameIdx);
	logger->debug(" ============= Out Buffer Size: {} =============", m_predictionBuffer.size());

	if (m_estimateTime)
	{
		time_2 = std::chrono::high_resolution_clock::now();
		logger->info("[Inference]: {} ms",
			std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_1).count() / (1000.0 * 1000));
	}

	notifyProcessingComplete();
#endif
	// ==================================

	time_2           = std::chrono::high_resolution_clock::now();
	auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_0);
	m_inferenceTime  = static_cast<float>(nanoseconds.count()) / 1e9f;

	if (m_estimateTime)
	{
		logger->info("[Total]: {} ms", m_inferenceTime);
		logger->info("-----------------------------------------");
	}

	logger->debug("End AI Model Part");
	logger->debug("========================================");
	m_bDone = true;

	return true;
}

void MotionFilter::runThread()
{
    m_threadInference = std::thread(&MotionFilter::_runInferenceFunc, this);
    return;
}

void MotionFilter::stopThread()
{
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_threadTerminated = true;
	}

    m_condition.notify_all(); // Wake up the thread if it's waiting
    if (m_threadInference.joinable())
    {
			m_threadInference.join();
    }
}

bool MotionFilter::_runInferenceFunc()
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("yolov8");
#else
    auto logger = spdlog::get("yolov8");
#endif

    while (!m_threadTerminated)
    {
			std::unique_lock<std::mutex> lock(m_mutex);
			m_condition.wait(lock,
				[this]() { return m_threadTerminated || (!m_inputFrameBuffer.empty() && m_threadStarted); });

			if (m_threadTerminated)
				break;

			// Read a frame from buffer
			auto    pair     = m_inputFrameBuffer.front();
			int     frameIdx = pair.first;
			ea_tensor_t* imgTensor = pair.second;
			m_inputFrameBuffer.pop_front();
			lock.unlock();

			// Perform AI Inference
			if (!_run(imgTensor, frameIdx))
			{
				logger->error("Failed in AI Inference");
				return true;
			}
    }
    return true;
}

void MotionFilter::notifyProcessingComplete()
{
    if (m_wakeFunc)
        m_wakeFunc();
}

//TODO:
// // =================================================================================================
// // Ambarella CV28 Historical Inference
// // =================================================================================================
// #if defined(CV28) || defined(CV28_SIMULATOR)
// bool YOLOv8::run_sequential(cv::Mat& imgFrame, YOLOv8_Prediction& pred)
// {
// #ifdef SPDLOG_USE_SYSLOG
// 	auto logger = spdlog::get("yolov8");
// #else
// 	auto logger = spdlog::get("yolov8");
// #endif

// 	auto time_0 = std::chrono::high_resolution_clock::now();
// 	auto time_1 = std::chrono::time_point<std::chrono::high_resolution_clock>{};
// 	auto time_2 = std::chrono::high_resolution_clock::now();
// 	auto time_3 = std::chrono::time_point<std::chrono::high_resolution_clock>{};
	
// 	int rval = EA_SUCCESS;

// 	// STEP0: Initialize and acquire image resource data
// 	ea_img_resource_data_t data;
// 	memset(&data, 0, sizeof(data));
// 	RVAL_OK(ea_img_resource_hold_data(m_img_resource, &data)); // TODO: remember to init m_img_resource

// 	// STEP 1: load image resource to input tensor
// 	if (!_loadInput(data))
// 	{
// 		logger->error("Load Input Data Failed");
// 		return false;
// 	}

// 	if (EA_SUCCESS != ea_net_forward(m_model, 1))
// 	{
// 		_releaseTensorBuffers();
// 		_releaseModel();

// 		m_bInferenced = false;
// 		m_bProcessed  = false;

// 		m_pred = YOLOv8_Prediction();
// 		pred   = m_pred;
// 		logger->warn("inference failed. something goes wrong.");

// 		return false;
// 	}
// 	else
// 	{
// 		m_bInferenced       = true;
// 		m_bProcessed        = false;

// 		if (m_estimateTime)
// 		{
// 			time_3 = std::chrono::high_resolution_clock::now();
// 		}

// 		// Sync output tensors between VP and CPU
// #if defined(CV28)
// 		rval = ea_tensor_sync_cache(m_outputTensors[0], EA_VP, EA_CP);
// 		RVAL_ASSERT(rval == EA_SUCCESS);
// 		rval = ea_tensor_sync_cache(m_outputTensors[1], EA_VP, EA_CP);
// 		RVAL_ASSERT(rval == EA_SUCCESS);
// 		rval = ea_tensor_sync_cache(m_outputTensors[2], EA_VP, EA_CP);
// 		RVAL_ASSERT(rval == EA_SUCCESS);
// #endif
// 		// Allocate memory for all output tensors
// 		// Object Detection Buffers
// 		m_pred.objBoxBuff   = new float[m_boxBufferSize];
// 		m_pred.objConfBuff  = new float[m_confBufferSize];
// 		m_pred.objClsBuff   = new float[m_classBufferSize];

// 		// Copy output tensors to prediction buffers
// 		m_pred.objBoxBuff 	= ea_net_output_by_index(m_model, 0);
// 		m_pred.objConfBuff 	= ea_net_output_by_index(m_model, 1);
// 		m_pred.objClsBuff 	= ea_net_output_by_index(m_model, 2);

// 		m_pred.img 	= imgFrame;

// 		pred         = m_pred;
// 		m_bProcessed = true;

// 		if (m_estimateTime)
// 		{
// 			time_2 = std::chrono::high_resolution_clock::now();
// 			logger->info("[Memcpy]: {} ms",
// 				std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_3).count() / (1000.0 * 1000));
// 			logger->info("[Inference]: {} ms",
// 				std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_1).count() / (1000.0 * 1000));
// 		}

//     // STEP3: Drop the image resource data after processing is complete
// 		rval = ea_img_resource_drop_data(m_img_resource, &data);
// 		RVAL_ASSERT(rval == EA_SUCCESS);

// 		logger->debug("inference finished and tensor copied");
// 		return true;
// 	}
// }
// #endif
// =================================================================================================

// =================================================================================================
// Load Inputs
// =================================================================================================
bool MotionFilter::_loadInput(ea_tensor_t* imgTensor)
{
#ifdef SPDLOG_USE_SYSLOG
	auto logger = spdlog::get("yolov8");
#else
	auto logger = spdlog::get("yolov8");
#endif

	auto time_0 = m_estimateTime ? std::chrono::high_resolution_clock::now()
																: std::chrono::time_point<std::chrono::high_resolution_clock>{};
	auto time_1 = std::chrono::time_point<std::chrono::high_resolution_clock>{};

	// Preprocessing
	if (_preProcessingMemory(imgTensor) != EA_SUCCESS)
	{
		logger->error("Data Preprocessing Failed");
		return false;
	}

	if (m_estimateTime)
	{
		time_1 = std::chrono::high_resolution_clock::now();
		logger->info("[Load Input]: {} ms",
			std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
	}

	return true;
}

void MotionFilter::updateInputFrame(ea_tensor_t* imgTensor, int frameIdx)
{
#ifdef SPDLOG_USE_SYSLOG
	auto logger = spdlog::get("yolov8");
#else
	auto logger = spdlog::get("yolov8");
#endif

	std::unique_lock<std::mutex> lock(m_mutex);

	m_inputFrameBuffer.emplace_back(frameIdx, imgTensor);
	m_threadStarted = true;
	m_bDone         = false;
	lock.unlock();

	m_condition.notify_one();
}

// =================================================================================================
// Pre Processing
// =================================================================================================
int MotionFilter::_preProcessingMemory(ea_tensor_t* imgTensor)
{
#ifdef SPDLOG_USE_SYSLOG
	auto logger = spdlog::get("yolov8");
#else
	auto logger = spdlog::get("yolov8");
#endif

	int rval = EA_SUCCESS;
	unsigned long start_time = 0;
	unsigned long preprocess_time = 0;

	start_time = ea_gettime_us();
	if (m_inputMode == DETECTION_MODE_FILE)
	{
#if defined(CV28)
		rval = ea_cvt_color_resize(imgTensor, m_inputTensor, EA_COLOR_BGR2RGB, EA_VP);
		// DROID-SLAM inputTensor, Alister add 2025-11-24
		rval = ea_cvt_color_resize(imgTensor, m_fnet_inputTensor, EA_COLOR_BGR2RGB, EA_VP);
		rval = ea_cvt_color_resize(imgTensor, m_cnet_inputTensor, EA_COLOR_BGR2RGB, EA_VP);
#elif defined(CV28_SIMULATOR)
		rval = ea_cvt_color_resize(imgTensor, m_inputTensor, EA_COLOR_BGR2RGB, EA_CPU);
		// DROID-SLAM inputTensor, Alister add 2025-11-24
		rval = ea_cvt_color_resize(imgTensor, m_fnet_inputTensor, EA_COLOR_BGR2RGB, EA_CPU);
		rval = ea_cvt_color_resize(imgTensor, m_cnet_inputTensor, EA_COLOR_BGR2RGB, EA_CPU);
#endif
	}
	else
	{
#if defined(CV28)
		rval = ea_cvt_color_resize(imgTensor, m_inputTensor, EA_COLOR_YUV2RGB_NV12, EA_VP);
		// DROID-SLAM inputTensor, Alister add 2025-11-24
		rval = ea_cvt_color_resize(imgTensor, m_fnet_inputTensor, EA_COLOR_YUV2RGB_NV12, EA_VP);
		rval = ea_cvt_color_resize(imgTensor, m_cnet_inputTensor, EA_COLOR_YUV2RGB_NV12, EA_VP);
#elif defined(CV28_SIMULATOR)
		rval = ea_cvt_color_resize(imgTensor, m_inputTensor, EA_COLOR_YUV2RGB_NV12, EA_CPU);
		// DROID-SLAM inputTensor, Alister add 2025-11-24
		rval = ea_cvt_color_resize(imgTensor, m_fnet_inputTensor, EA_COLOR_YUV2RGB_NV12, EA_CPU);
		rval = ea_cvt_color_resize(imgTensor, m_cnet_inputTensor, EA_COLOR_YUV2RGB_NV12, EA_CPU);
#endif
	}
	preprocess_time = ea_gettime_us() - start_time;
	logger->info("Preprocess time: {} us", preprocess_time);

	return rval;
}



// =================================================================================================
// Post Processing
// =================================================================================================
bool MotionFilter::getLastestPrediction(YOLOv8_Prediction& pred, int& frameIdx)
{
#ifdef SPDLOG_USE_SYSLOG
	auto logger = spdlog::get("yolov8");
#else
	auto logger = spdlog::get("yolov8");
#endif

	std::unique_lock<std::mutex> lock(m_mutex);
	const auto                   bufferSize = m_predictionBuffer.size();

	if (bufferSize == 0)
		return false;

	if (bufferSize > 0)
	{
		// pred = m_predictionBuffer.front();
		auto pair = m_predictionBuffer.front();
		frameIdx  = pair.first;
		pred      = pair.second;
		m_predictionBuffer.pop_front();
		return true;
	}

	lock.unlock();

	logger->warn("buffSize is negative = {}", m_predictionBuffer.size());
	return false;
}

// =================================================================================================
// Utility Functions
// =================================================================================================
void MotionFilter::getDebugProfiles(float& inferenceTime, int& inputBufferSize, int& outputBufferSize)
{
	std::unique_lock<std::mutex> lock(m_mutex);
	inputBufferSize  = m_inputFrameBuffer.size();
	outputBufferSize = m_predictionBuffer.size();
	inferenceTime    = m_inferenceTime;
	lock.unlock();
}

bool MotionFilter::isInputBufferEmpty() const
{
	std::lock_guard<std::mutex> lock(m_mutex);
	return m_inputFrameBuffer.empty();
}

bool MotionFilter::isPredictionBufferEmpty() const
{
	std::lock_guard<std::mutex> lock(m_mutex);
	return m_predictionBuffer.empty();
}

void MotionFilter::updateTensorPath(const std::string& path)
{
#ifdef SPDLOG_USE_SYSLOG
	auto logger = spdlog::get("yolov8");
#else
	auto logger = spdlog::get("yolov8");
#endif

	m_tensorPath = path; // Assuming m_tensorPath is a member variable to store the path
	logger->info("Updated tensor path to: {}", m_tensorPath);
}