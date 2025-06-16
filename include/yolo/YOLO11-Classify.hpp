#pragma once

/**
 * @file YOLO11-Classify.hpp
 * @brief Header file for the YOLO11Classify class, responsible for image classification
 * using an ONNX model with optimized performance for minimal latency.
 */

// Include necessary ONNX Runtime and OpenCV headers
#define NOMINMAX

#include <yolo/YOLO11-Base.hpp>
#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <unordered_map>
#include <thread>
#include <iomanip> // For std::fixed and std::setprecision
#include <sstream> // For std::ostringstream

// #define DEBUG_MODE // Enable debug mode for detailed logging

// Include debug and custom ScopedTimer tools for performance measurement
// Assuming these are in a common 'tools' directory relative to this header
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"

/**
 * @brief Struct to represent a classification result.
 */
struct ClassifyResult
{
    int classId{-1}; // Predicted class ID, initialized to -1 for easier error checking
    float conf{0.0f}; // Confidence score for the prediction
    std::string label{}; // Name of the predicted class

    ClassifyResult() = default;

    ClassifyResult(int id, float conf_, std::string name)
        : classId(id), conf(conf_), label(std::move(name))
    {
    }
};

// 手动实现 to_json
inline void to_json(nlohmann::json& j, const ClassifyResult& result)
{
    j = nlohmann::json{
        {"classId", result.classId},
        {"conf", result.conf},
        {"label", result.label},
    };
}

/**
 * @namespace ClsUtils
 * @brief Namespace containing utility functions for the YOLO11Classifier.
 */
namespace ClsUtils
{
    /**
     * @brief Draws the classification result on the image.
     */
    inline void drawClassifyResult(cv::Mat& image, const ClassifyResult& result,
                                   const cv::Point& position = cv::Point(10, 10),
                                   const cv::Scalar& textColor = cv::Scalar(0, 255, 0),
                                   double fontScaleMultiplier = 0.0008,
                                   const cv::Scalar& bgColor = cv::Scalar(0, 0, 0))
    {
        if (image.empty())
        {
            std::cerr << "ERROR: Empty image provided to drawClassificationResult." << std::endl;
            return;
        }
        if (result.classId == -1)
        {
            DEBUG_PRINT("Skipping drawing due to invalid classification result.");
            return;
        }

        std::ostringstream ss;
        ss << result.label << ": " << std::fixed << std::setprecision(2) << result.conf * 100 << "%";
        std::string text = ss.str();

        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = std::min(image.rows, image.cols) * fontScaleMultiplier;
        if (fontScale < 0.4) fontScale = 0.4;
        const int thickness = std::max(1, static_cast<int>(fontScale * 1.8));
        int baseline = 0;

        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
        baseline += thickness;

        cv::Point textPosition = position;
        if (textPosition.x < 0) textPosition.x = 0;
        if (textPosition.y < textSize.height) textPosition.y = textSize.height + 2;

        cv::Point backgroundTopLeft(textPosition.x, textPosition.y - textSize.height - baseline / 3);
        cv::Point backgroundBottomRight(textPosition.x + textSize.width, textPosition.y + baseline / 2);

        backgroundTopLeft.x = YOLOUtils::clamp(backgroundTopLeft.x, 0, image.cols - 1);
        backgroundTopLeft.y = YOLOUtils::clamp(backgroundTopLeft.y, 0, image.rows - 1);
        backgroundBottomRight.x = YOLOUtils::clamp(backgroundBottomRight.x, 0, image.cols - 1);
        backgroundBottomRight.y = YOLOUtils::clamp(backgroundBottomRight.y, 0, image.rows - 1);

        cv::rectangle(image, backgroundTopLeft, backgroundBottomRight, bgColor, cv::FILLED);
        cv::putText(image, text, cv::Point(textPosition.x, textPosition.y), fontFace, fontScale, textColor, thickness,
                    cv::LINE_AA);

        DEBUG_PRINT("Classification result drawn on image: " << text);
    }
}; // end namespace utils


/**
 * @brief YOLO11Classifier class handles loading the classification model,
 * preprocessing images, running inference, and postprocessing results.
 */
class YOLO11Classify : public YOLO11Model
{
public:
    /**
     * @brief Constructor to initialize the classifier with model and label paths.
     */
    explicit YOLO11Classify(const std::vector<char>& modelBuffer,
                            bool useGPU = false);

    std::string getTask() const override { return "classify"; }

    /**
     * @brief Runs classification on the provided image.
     */
    ClassifyResult classify(const cv::Mat& image);

    /**
     * @brief Draws the classification result on the image.
     */
    void drawResult(cv::Mat& image, const ClassifyResult& result,
                    const cv::Point& position = cv::Point(10, 10)) const
    {
        ClsUtils::drawClassifyResult(image, result, position);
    }

    cv::Size getInputShape() const { return inputImageShape; } // CORRECTED
    bool isModelInputShapeDynamic() const { return isDynamicInputShape; } // CORRECTED


private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    bool isDynamicInputShape{false};
    cv::Size inputImageShape;

    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings{};
    std::vector<const char*> inputNames{};
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings{};
    std::vector<const char*> outputNames{};

    size_t numInputNodes{0};
    size_t numOutputNodes{0};
    int numClasses{0};

    std::vector<std::string> classNames{};

    cv::Mat preprocess(const cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
    ClassifyResult postprocess(const std::vector<Ort::Value>& outputTensors);
};

// Implementation of YOLO11Classify constructor
inline YOLO11Classify::YOLO11Classify(const std::vector<char>& modelBuffer, bool useGPU)
{
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLO_CLASSIFY");
    sessionOptions = Ort::SessionOptions();

    sessionOptions.SetIntraOpNumThreads(std::min(4, static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    // Retrieve available execution providers (e.g., CPU, DML)
    // 获取可用的执行器 CPU DML
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    // 根据是否使用 GPU 和可用的执行器配置会话选项
    if (auto dmlAvailable = std::find(availableProviders.begin(), availableProviders.end(), "DmlExecutionProvider");
        useGPU && dmlAvailable != availableProviders.end())
    {
        std::cout << "Inference device: GPU" << std::endl;
        OrtApi const& ortApi = Ort::GetApi();
        OrtDmlApi const* ortDmlApi = nullptr;
        ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<void const**>(&ortDmlApi));
        ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, 0);
    }
    else
    {
        if (useGPU) { std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl; }
        std::cout << "Inference device: CPU" << std::endl;
    }

    // Load the ONNX model into the session
    session = Ort::Session(env, modelBuffer.data(), modelBuffer.size(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    // Retrieve input tensor shape information
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShapeVec = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    isDynamicInputShape = (inputTensorShapeVec.size() >= 4) && (inputTensorShapeVec[2] == -1 && inputTensorShapeVec[3]
        == -1); // Check for dynamic dimensions

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputTensorShapeVec = outputTensorInfo.GetShape();

    numInputNodes = session.GetInputCount();
    numOutputNodes = session.GetOutputCount();

    for (int inputLayer = 0; inputLayer < numInputNodes; ++inputLayer)
    {
        // Allocate and store input node names
        auto input_name = session.GetInputNameAllocated(inputLayer, allocator);
        inputNodeNameAllocatedStrings.push_back(std::move(input_name));
        inputNames.push_back(inputNodeNameAllocatedStrings.back().get());
    }

    for (int outputLayer = 0; outputLayer < numOutputNodes; ++outputLayer)
    {
        // Allocate and store output node names
        auto output_name = session.GetOutputNameAllocated(outputLayer, allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        outputNames.push_back(outputNodeNameAllocatedStrings.back().get());
    }

    // Set the expected input image shape based on the model's input tensor
    if (inputTensorShapeVec.size() >= 4)
    {
        inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]), static_cast<int>(inputTensorShapeVec[2]));
    }
    else { throw std::runtime_error("Invalid input tensor shape."); }

    if (!outputTensorShapeVec.empty())
    {
        if (outputTensorShapeVec.size() == 2 && outputTensorShapeVec[0] > 0)
        {
            numClasses = static_cast<int>(outputTensorShapeVec[1]);
        }
        else if (outputTensorShapeVec.size() == 1 && outputTensorShapeVec[0] > 0)
        {
            numClasses = static_cast<int>(outputTensorShapeVec[0]);
        }
        else
        {
            for (long long dim : outputTensorShapeVec)
            {
                if (dim > 1 && numClasses == 0) { numClasses = static_cast<int>(dim); }
            }
            if (numClasses == 0 && !outputTensorShapeVec.empty())
            {
                numClasses = static_cast<int>(outputTensorShapeVec.back());
            }
        }
    }

    if (numClasses > 0)
    {
        // CORRECTED SECTION for printing outputTensorShapeVec
        std::ostringstream oss_shape;
        oss_shape << "[";
        for (size_t i = 0; i < outputTensorShapeVec.size(); ++i)
        {
            oss_shape << outputTensorShapeVec[i];
            if (i < outputTensorShapeVec.size() - 1) { oss_shape << ", "; }
        }
        oss_shape << "]";
        DEBUG_PRINT("Model predicts " << numClasses << " classes based on output shape: " << oss_shape.str());
        // END CORRECTED SECTION
    }
    else
    {
        std::cerr << "Warning: Could not reliably determine number of classes from output shape: [";
        for (size_t i = 0; i < outputTensorShapeVec.size(); ++i)
        {
            // Directly print to cerr
            std::cerr << outputTensorShapeVec[i] << (i == outputTensorShapeVec.size() - 1 ? "" : ", ");
        }
        std::cerr << "]. Postprocessing might be incorrect or assume a default." << std::endl;
    }

    // Load class names and generate corresponding colors
    const auto model_metadata = session.GetModelMetadata();
    Ort::AllocatedStringPtr search = model_metadata.LookupCustomMetadataMapAllocated("names", allocator);
    if (search != nullptr) { classNames = YOLOUtils::parseClassNames(search.get()); }
    if (numClasses > 0 && !classNames.empty() && classNames.size() != static_cast<size_t>(numClasses))
    {
        std::cerr << "Warning: Number of classes from model (" << numClasses << ") (" << classNames.size() << ")." <<
            std::endl;
    }
    if (classNames.empty() && numClasses > 0)
    {
        std::cout <<
            "Warning: Class names file is empty or failed to load. Predictions will use numeric IDs if labels are not available."
            << std::endl;
    }

    std::cout << "YOLO11Classify initialized successfully." << std::endl;
}

// ... (preprocess, postprocess, and classify methods remain the same as previous correct version) ...
inline cv::Mat YOLO11Classify::preprocess(const cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    ScopedTimer timer("Preprocessing (Ultralytics-style)");

    cv::Mat resizedImage;
    // Resize and pad the image using letterBox utility
    YOLOUtils::letterBox(image, resizedImage, inputImageShape, cv::Scalar(114, 114, 114), isDynamicInputShape, false,
                         true, 32);

    // Update input tensor shape based on resized image dimensions
    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    // Convert image to float and normalize to [0, 1]
    resizedImage.convertTo(resizedImage, CV_32FC3, 1 / 255.0f);

    // Allocate memory for the image blob in CHW format
    blob = new float[resizedImage.cols * resizedImage.rows * resizedImage.channels()];

    // Split the image into separate channels and store in the blob
    std::vector<cv::Mat> chw(resizedImage.channels());
    for (int i = 0; i < resizedImage.channels(); ++i)
    {
        chw[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1,
                         blob + i * resizedImage.cols * resizedImage.rows);
    }
    cv::split(resizedImage, chw); // Split channels into the blob

    DEBUG_PRINT("Preprocessing completed")

    return resizedImage;
}

inline ClassifyResult YOLO11Classify::postprocess(const std::vector<Ort::Value>& outputTensors)
{
    ScopedTimer timer("Postprocessing");

    if (outputTensors.empty())
    {
        std::cerr << "Error: No output tensors for postprocessing." << std::endl;
        return {};
    }

    const float* rawOutput = outputTensors[0].GetTensorData<float>();
    if (!rawOutput)
    {
        std::cerr << "Error: rawOutput pointer is null." << std::endl;
        return {};
    }

    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t numScores = YOLOUtils::vectorProduct(outputShape);

    // Debug output shape
    std::ostringstream oss_shape;
    oss_shape << "Output tensor shape: [";
    for (size_t i = 0; i < outputShape.size(); ++i)
    {
        oss_shape << outputShape[i] << (i == outputShape.size() - 1 ? "" : ", ");
    }
    oss_shape << "]";
    DEBUG_PRINT(oss_shape.str());

    // Determine the effective number of classes
    int currentNumClasses = numClasses > 0 ? numClasses : static_cast<int>(classNames.size());
    if (currentNumClasses <= 0)
    {
        std::cerr << "Error: No valid number of classes determined." << std::endl;
        return {};
    }

    // Debug first few raw scores
    std::ostringstream oss_scores;
    oss_scores << "First few raw scores: ";
    for (size_t i = 0; i < std::min(size_t(5), numScores); ++i) { oss_scores << rawOutput[i] << " "; }
    DEBUG_PRINT(oss_scores.str());

    // Find maximum score and its corresponding class
    int bestClassId = -1;
    float maxScore = -std::numeric_limits<float>::infinity();
    std::vector<float> scores(currentNumClasses);

    // Handle different output shapes
    if (outputShape.size() == 2 && outputShape[0] == 1)
    {
        // Case 1: [1, num_classes] shape
        for (int i = 0; i < currentNumClasses && i < static_cast<int>(outputShape[1]); ++i)
        {
            scores[i] = rawOutput[i];
            if (scores[i] > maxScore)
            {
                maxScore = scores[i];
                bestClassId = i;
            }
        }
    }
    else if (outputShape.size() == 1 || (outputShape.size() == 2 && outputShape[0] > 1))
    {
        // Case 2: [num_classes] shape or [batch_size, num_classes] shape (take first batch)
        for (int i = 0; i < currentNumClasses && i < static_cast<int>(numScores); ++i)
        {
            scores[i] = rawOutput[i];
            if (scores[i] > maxScore)
            {
                maxScore = scores[i];
                bestClassId = i;
            }
        }
    }

    if (bestClassId == -1)
    {
        std::cerr << "Error: Could not determine best class ID." << std::endl;
        return {};
    }

    // Apply softmax to get probabilities
    float sumExp = 0.0f;
    std::vector<float> probabilities(currentNumClasses);

    // Compute softmax with numerical stability
    for (int i = 0; i < currentNumClasses; ++i)
    {
        probabilities[i] = std::exp(scores[i] - maxScore);
        sumExp += probabilities[i];
    }

    // Calculate final confidence
    float confidence = sumExp > 0 ? probabilities[bestClassId] / sumExp : 0.0f;

    // Get class name
    std::string className = "Unknown";
    if (bestClassId >= 0 && static_cast<size_t>(bestClassId) < classNames.size())
    {
        className = classNames[bestClassId];
    }
    else if (bestClassId >= 0) { className = "ClassID_" + std::to_string(bestClassId); }

    DEBUG_PRINT("Best class ID: " << bestClassId << ", Name: " << className << ", Confidence: " << confidence);
    return ClassifyResult(bestClassId, confidence, className);
}

inline ClassifyResult YOLO11Classify::classify(const cv::Mat& image)
{
    ScopedTimer timer("ClassifyTask");

    float* blobPtr = nullptr; // Pointer to hold preprocessed image data
    // Define the shape of the input tensor (batch size, channels, height, width)
    std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height, inputImageShape.width};

    // Preprocess the image and obtain a pointer to the blob
    cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);

    // Compute the total number of elements in the input tensor
    size_t inputTensorSize = YOLOUtils::vectorProduct(inputTensorShape);

    // Create a vector from the blob data for ONNX Runtime input
    std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);

    delete[] blobPtr; // Free the allocated memory for the blob

    // Create an Ort memory info object (can be cached if used repeatedly)
    static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor object using the preprocessed data
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorSize,
        inputTensorShape.data(),
        inputTensorShape.size()
    );

    // Run the inference session with the input tensor and retrieve output tensors
    std::vector<Ort::Value> outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        &inputTensor,
        numInputNodes,
        outputNames.data(),
        numOutputNodes
    );

    auto result = postprocess(outputTensors);
    return result;
}
