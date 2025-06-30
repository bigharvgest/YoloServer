#pragma once

/**
 * @file YOLO11-Detect.hpp
 * @brief Header file for the YOLO11Detector class, responsible for object detection
 *        using the YOLOv11 model with optimized performance for minimal latency.
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

// Include debug and custom ScopedTimer tools for performance measurement
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"


/**
 * @brief Struct representing a single detection result.
 */
struct BoxDetection
{
    YOLOUtils::RectBox box; /**< Bounding box of the detected object */
    float conf{}; /**< Confidence score of the detection */
    int classId{}; /**< Class ID of the detected object */
    std::string label;
};

// 手动实现 to_json
inline void to_json(nlohmann::json& j, const BoxDetection& box)
{
    j = nlohmann::json{
        {"conf", box.conf},
        {"classId", box.classId},
        {"label", box.label},
        {"box", box.box}
    };
}

/**
 * @namespace DetUtils
 * @brief 给矩形框用的工具函数
 */
namespace DetUtils
{
    /**
     * @brief Draws bounding boxes and labels on the image based on detections.
     *
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     * @param classNames Vector of class names corresponding to object IDs.
     * @param colors Vector of colors for each class.
     */
    inline void drawBoundingBox(cv::Mat& image, const std::vector<BoxDetection>& detections,
                                const std::vector<std::string>& classNames, const std::vector<cv::Scalar>& colors)
    {
        // Iterate through each detection to draw bounding boxes and labels
        for (const auto& detection : detections)
        {
            // Ensure the object ID is within valid range
            if (detection.classId < 0 || static_cast<size_t>(detection.classId) >= classNames.size()) continue;

            // Select color based on object ID for consistent coloring
            const cv::Scalar& color = colors[detection.classId % colors.size()];

            // Draw the bounding box rectangle
            cv::rectangle(image, cv::Point(detection.box.x, detection.box.y),
                          cv::Point(detection.box.x + detection.box.width, detection.box.y + detection.box.height),
                          color, 1, cv::LINE_AA);

            // Prepare label text with class name and confidence percentage
            std::string label = classNames[detection.classId] + ": " + std::to_string(
                static_cast<int>(detection.conf * 100)) + "%";

            // Define text properties for labels
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = std::min(image.rows, image.cols) * 0.0008;
            const int thickness = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));
            int baseline = 0;

            // Calculate text size for background rectangles
            cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);

            // Define positions for the label
            int labelY = std::max(detection.box.y, textSize.height + 5);
            cv::Point labelTopLeft(detection.box.x, labelY - textSize.height - 5);
            cv::Point labelBottomRight(detection.box.x + textSize.width + 5, labelY + baseline - 5);

            // Draw background rectangle for label
            cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

            // Put label text
            cv::putText(image, label, cv::Point(detection.box.x + 2, labelY - 2), fontFace, fontScale,
                        cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        }
    }

    /**
     * @brief Draws bounding boxes and semi-transparent masks on the image based on detections.
     *
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     * @param classNames Vector of class names corresponding to object IDs.
     * @param classColors Vector of colors for each class.
     * @param maskAlpha Alpha value for the mask transparency.
     */
    inline void drawBoundingBoxMask(cv::Mat& image, const std::vector<BoxDetection>& detections,
                                    const std::vector<std::string>& classNames,
                                    const std::vector<cv::Scalar>& classColors,
                                    float maskAlpha = 0.4f)
    {
        // Validate input image
        if (image.empty())
        {
            std::cerr << "ERROR: Empty image provided to drawBoundingBoxMask." << std::endl;
            return;
        }

        const int imgHeight = image.rows;
        const int imgWidth = image.cols;

        // Precompute dynamic font size and thickness based on image dimensions
        const double fontSize = std::min(imgHeight, imgWidth) * 0.0006;
        const int textThickness = std::max(1, static_cast<int>(std::min(imgHeight, imgWidth) * 0.001));

        // Create a mask image for blending (initialized to zero)
        cv::Mat maskImage(image.size(), image.type(), cv::Scalar::all(0));

        // Pre-filter detections to include only those above the confidence threshold and with valid class IDs
        std::vector<const BoxDetection*> filteredDetections;
        for (const auto& detection : detections)
        {
            if (detection.classId >= 0 && static_cast<size_t>(detection.classId) < classNames.size())
            {
                filteredDetections.emplace_back(&detection);
            }
        }

        // Draw filled rectangles on the mask image for the semi-transparent overlay
        for (const auto* detection : filteredDetections)
        {
            cv::Rect box(detection->box.x, detection->box.y, detection->box.width, detection->box.height);
            const cv::Scalar& color = classColors[detection->classId];
            cv::rectangle(maskImage, box, color, cv::FILLED);
        }

        // Blend the maskImage with the original image to apply the semi-transparent masks
        cv::addWeighted(maskImage, maskAlpha, image, 1.0f, 0, image);

        // Draw bounding boxes and labels on the original image
        for (const auto* detection : filteredDetections)
        {
            cv::Rect box(detection->box.x, detection->box.y, detection->box.width, detection->box.height);
            const cv::Scalar& color = classColors[detection->classId];
            cv::rectangle(image, box, color, 2, cv::LINE_AA);

            std::string label = classNames[detection->classId] + ": " + std::to_string(
                static_cast<int>(detection->conf * 100)) + "%";
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontSize, textThickness, &baseLine);

            int labelY = std::max(detection->box.y, labelSize.height + 5);
            cv::Point labelTopLeft(detection->box.x, labelY - labelSize.height - 5);
            cv::Point labelBottomRight(detection->box.x + labelSize.width + 5, labelY + baseLine - 5);

            // Draw background rectangle for label
            cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

            // Put label text
            cv::putText(image, label, cv::Point(detection->box.x + 2, labelY - 2), cv::FONT_HERSHEY_SIMPLEX, fontSize,
                        cv::Scalar(255, 255, 255), textThickness, cv::LINE_AA);
        }

        DEBUG_PRINT("Bounding boxes and masks drawn on image.");
    }

    inline void NMSBoxes(const std::vector<YOLOUtils::RectBox>& boundingBoxes, const std::vector<float>& scores,
                         float scoreThreshold, float nmsThreshold, std::vector<int>& indices)
    {
        indices.clear();

        const size_t numBoxes = boundingBoxes.size();
        if (numBoxes == 0)
        {
            DEBUG_PRINT("No bounding boxes to process in NMS");
            return;
        }

        // Step 1: Filter out boxes with scores below the threshold
        // and create a list of indices sorted by descending scores
        std::vector<int> sortedIndices;
        sortedIndices.reserve(numBoxes);
        for (size_t i = 0; i < numBoxes; ++i)
        {
            if (scores[i] >= scoreThreshold) { sortedIndices.push_back(static_cast<int>(i)); }
        }

        // If no boxes remain after thresholding
        if (sortedIndices.empty())
        {
            DEBUG_PRINT("No bounding boxes above score threshold");
            return;
        }

        // Sort the indices based on scores in descending order
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                  [&scores](int idx1, int idx2) { return scores[idx1] > scores[idx2]; });

        // Step 2: Precompute the areas of all boxes
        std::vector<float> areas(numBoxes, 0.0f);
        for (size_t i = 0; i < numBoxes; ++i) { areas[i] = boundingBoxes[i].width * boundingBoxes[i].height; }

        // Step 3: Suppression mask to mark boxes that are suppressed
        std::vector<bool> suppressed(numBoxes, false);

        // Step 4: Iterate through the sorted list and suppress boxes with high IoU
        for (size_t i = 0; i < sortedIndices.size(); ++i)
        {
            int currentIdx = sortedIndices[i];
            if (suppressed[currentIdx]) { continue; }

            // Select the current box as a valid detection
            indices.push_back(currentIdx);

            const YOLOUtils::RectBox& currentBox = boundingBoxes[currentIdx];
            const float x1_max = currentBox.x;
            const float y1_max = currentBox.y;
            const float x2_max = currentBox.x + currentBox.width;
            const float y2_max = currentBox.y + currentBox.height;
            const float area_current = areas[currentIdx];

            // Compare IoU of the current box with the rest
            for (size_t j = i + 1; j < sortedIndices.size(); ++j)
            {
                int compareIdx = sortedIndices[j];
                if (suppressed[compareIdx]) { continue; }

                const YOLOUtils::RectBox& compareBox = boundingBoxes[compareIdx];
                const float x1 = std::max(x1_max, static_cast<float>(compareBox.x));
                const float y1 = std::max(y1_max, static_cast<float>(compareBox.y));
                const float x2 = std::min(x2_max, static_cast<float>(compareBox.x + compareBox.width));
                const float y2 = std::min(y2_max, static_cast<float>(compareBox.y + compareBox.height));

                const float interWidth = x2 - x1;
                const float interHeight = y2 - y1;

                if (interWidth <= 0 || interHeight <= 0) { continue; }

                const float intersection = interWidth * interHeight;
                const float unionArea = area_current + areas[compareIdx] - intersection;
                const float iou = (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;

                if (iou > nmsThreshold) { suppressed[compareIdx] = true; }
            }
        }

        DEBUG_PRINT("NMS completed with " + std::to_string(indices.size()) + " indices remaining");
    }
}

/**
 * @brief YOLO11Detector class handles loading the YOLO model, preprocessing images, running inference, and postprocessing results.
 */
class YOLO11Detect : public YOLO11Model
{
public:
    /**
     * @brief Constructor to initialize the YOLO detector with model and label paths.
     *
     * @param modelBuffer 模型二进制
     * @param useGPU Whether to use GPU for inference (default is false).
     * @param device GPU device ID to use (default is 0).
     */
    explicit YOLO11Detect(const std::vector<char>& modelBuffer, bool useGPU = false, int device = 0);

    std::string getTask() const override { return "detect"; }

    /**
     * @brief Runs detection on the provided image.
     *
     * @param image Input image for detection.
     * @param confThreshold Confidence threshold to filter detections (default is 0.4).
     * @param iouThreshold IoU threshold for Non-Maximum Suppression (default is 0.45).
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<BoxDetection> detect(const cv::Mat& image, float confThreshold = 0.4f, float iouThreshold = 0.45f);

    /**
     * @brief Draws bounding boxes on the image based on detections.
     *
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     */
    void drawBoundingBox(cv::Mat& image, const std::vector<BoxDetection>& detections) const
    {
        DetUtils::drawBoundingBox(image, detections, classNames, classColors);
    }

    /**
     * @brief Draws bounding boxes and semi-transparent masks on the image based on detections.
     *
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     * @param maskAlpha Alpha value for mask transparency (default is 0.4).
     */
    void drawBoundingBoxMask(cv::Mat& image, const std::vector<BoxDetection>& detections, float maskAlpha = 0.4f) const
    {
        DetUtils::drawBoundingBoxMask(image, detections, classNames, classColors, maskAlpha);
    }

private:
    Ort::Env env{nullptr}; // ONNX Runtime environment
    Ort::SessionOptions sessionOptions{nullptr}; // Session options for ONNX Runtime
    Ort::Session session{nullptr}; // ONNX Runtime session for running inference
    // 是否动态图像大小
    bool isDynamicInputShape{false}; // Flag indicating if input shape is dynamic
    cv::Size inputImageShape; // Expected input image shape for the model
    // 是否是yolov5模型
    bool isYoloV5{false};

    // Vectors to hold allocated input and output node names
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings{};
    std::vector<const char*> inputNames{};
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings{};
    std::vector<const char*> outputNames{};

    size_t numInputNodes{0};
    size_t numOutputNodes{0};

    std::vector<std::string> classNames{}; // Vector of class names loaded from file
    std::vector<cv::Scalar> classColors{}; // Vector of colors for each class

    /**
     * @brief Preprocesses the input image for model inference.
     *
     * @param image Input image.
     * @param blob Reference to pointer where preprocessed data will be stored.
     * @param inputTensorShape Reference to vector representing input tensor shape.
     * @return cv::Mat Resized image after preprocessing.
     */
    cv::Mat preprocess(const cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);

    /**
     * @brief Postprocesses the model output to extract detections.
     *
     * @param originalImageSize Size of the original input image.
     * @param resizedImageShape Size of the image after preprocessing.
     * @param outputTensors Vector of output tensors from the model.
     * @param confThreshold Confidence threshold to filter detections.
     * @param iouThreshold IoU threshold for Non-Maximum Suppression.
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<BoxDetection> postprocess(const cv::Size& originalImageSize, const cv::Size& resizedImageShape,
                                          const std::vector<Ort::Value>& outputTensors,
                                          float confThreshold, float iouThreshold);

    /**
     * @brief Postprocesses the model output to extract detections for YOLOv5.
     *
     * @param originalImageSize Size of the original input image.
     * @param resizedImageShape Size of the image after preprocessing.
     * @param outputTensors Vector of output tensors from the model.
     * @param confThreshold Confidence threshold to filter detections.
     * @param iouThreshold IoU threshold for Non-Maximum Suppression.
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<BoxDetection> postprocessYoloV5(const cv::Size& originalImageSize, const cv::Size& resizedImageShape,
                                                const std::vector<Ort::Value>& outputTensors,
                                                float confThreshold, float iouThreshold);

    /**
     * @brief Extracts the best class information from a detection row.
     *
     * @param it Iterator pointing to the start of the detection row.
     * @param numClasses Number of classes.
     * @param bestConf Reference to store the best confidence score.
     * @param bestClassId Reference to store the best class ID.
     */
    void getBestClassInfoYoloV5(std::vector<float>::iterator it, const int& numClasses,
                                float& bestConf, int& bestClassId);
};

// Implementation of YOLO11Detector constructor
inline YOLO11Detect::YOLO11Detect(const std::vector<char>& modelBuffer, bool useGPU, int device)
{
    // 初始化 ONNX Runtime 环境，设置警告级别为警告
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLO_DETECT");
    // 创建 ONNX Runtime 会话选项
    sessionOptions = Ort::SessionOptions();

    // Set number of intra-op threads for parallelism
    // 设置线程数为 CPU 核心数的最小值或 6
    sessionOptions.SetIntraOpNumThreads(std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
    // 设置图优化级别为最大化性能
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
        ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, device);
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

    // get output tensor shape information
    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    std::vector<int64_t> outputTensorShapeVec = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    if (outputTensorShapeVec[1] == 25200) { isYoloV5 = true; }

    // Get the number of input and output nodes
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

    // Load class names and generate corresponding colors
    const auto model_metadata = session.GetModelMetadata();
    Ort::AllocatedStringPtr names = model_metadata.LookupCustomMetadataMapAllocated("names", allocator);
    if (names != nullptr) { classNames = YOLOUtils::parseClassNames(names.get()); }
    classColors = YOLOUtils::generateColors(classNames);

    std::cout << "Model loaded successfully with " << numInputNodes << " input nodes and " << numOutputNodes <<
        " output nodes." << std::endl;
}

// Preprocess function implementation
inline cv::Mat YOLO11Detect::preprocess(const cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    ScopedTimer timer("preprocessing");

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

// Postprocess function to convert raw model output into detections
inline std::vector<BoxDetection> YOLO11Detect::postprocess(
    const cv::Size& originalImageSize,
    const cv::Size& resizedImageShape,
    const std::vector<Ort::Value>& outputTensors,
    float confThreshold,
    float iouThreshold
)
{
    if (isYoloV5)
    {
        return postprocessYoloV5(originalImageSize, resizedImageShape, outputTensors, confThreshold, iouThreshold);
    }
    ScopedTimer timer("postprocessing"); // Measure postprocessing time

    std::vector<BoxDetection> detections;
    const float* rawOutput = outputTensors[0].GetTensorData<float>();
    // Extract raw output data from the first output tensor
    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    // Determine the number of features and detections
    const size_t num_features = outputShape[1];
    const size_t num_detections = outputShape[2];

    // Early exit if no detections
    if (num_detections == 0) { return detections; }

    // Calculate number of classes based on output shape
    const int numClasses = static_cast<int>(num_features) - 4;
    if (numClasses <= 0)
    {
        // Invalid number of classes
        return detections;
    }

    // Reserve memory for efficient appending
    std::vector<YOLOUtils::RectBox> boxes;
    boxes.reserve(num_detections);
    std::vector<float> confs;
    confs.reserve(num_detections);
    std::vector<int> classIds;
    classIds.reserve(num_detections);
    std::vector<YOLOUtils::RectBox> nms_boxes;
    nms_boxes.reserve(num_detections);

    // Constants for indexing
    const float* ptr = rawOutput;

    for (size_t d = 0; d < num_detections; ++d)
    {
        // Extract bounding box coordinates (center x, center y, width, height)
        float centerX = ptr[0 * num_detections + d];
        float centerY = ptr[1 * num_detections + d];
        float width = ptr[2 * num_detections + d];
        float height = ptr[3 * num_detections + d];

        // Find class with the highest confidence score
        int classId = -1;
        float maxScore = -FLT_MAX;
        for (int c = 0; c < numClasses; ++c)
        {
            const float score = ptr[d + (4 + c) * num_detections];
            if (score > maxScore)
            {
                maxScore = score;
                classId = c;
            }
        }

        // Proceed only if confidence exceeds threshold
        if (maxScore > confThreshold)
        {
            // Convert center coordinates to top-left (x1, y1)
            float left = centerX - width / 2.0f;
            float top = centerY - height / 2.0f;

            // Scale to original image size
            YOLOUtils::RectBox scaledBox = YOLOUtils::scaleCoords(
                resizedImageShape,
                YOLOUtils::RectBox(left, top, width, height),
                originalImageSize,
                true
            );

            // Round coordinates for integer pixel positions
            YOLOUtils::RectBox roundedBox;
            roundedBox.x = std::round(scaledBox.x);
            roundedBox.y = std::round(scaledBox.y);
            roundedBox.width = std::round(scaledBox.width);
            roundedBox.height = std::round(scaledBox.height);

            // Adjust NMS box coordinates to prevent overlap between classes
            YOLOUtils::RectBox nmsBox = roundedBox;
            nmsBox.x += classId * 7680; // Arbitrary offset to differentiate classes
            nmsBox.y += classId * 7680;

            // Add to respective containers
            nms_boxes.emplace_back(nmsBox);
            boxes.emplace_back(roundedBox);
            confs.emplace_back(maxScore);
            classIds.emplace_back(classId);
        }
    }

    // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
    std::vector<int> indices;
    DetUtils::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);

    // Collect filtered detections into the result vector
    detections.reserve(indices.size());
    for (const int idx : indices)
    {
        detections.emplace_back(BoxDetection{
            boxes[idx], // Bounding box
            confs[idx], // Confidence score
            classIds[idx], // Class ID
            classNames[classIds[idx]]
        });
    }

    DEBUG_PRINT("Postprocessing completed") // Debug log for completion

    return detections;
}

inline std::vector<BoxDetection> YOLO11Detect::postprocessYoloV5(const cv::Size& originalImageSize,
                                                                 const cv::Size& resizedImageShape,
                                                                 const std::vector<Ort::Value>& outputTensors,
                                                                 float confThreshold, float iouThreshold)
{
    ScopedTimer timer("postprocessing");


    std::vector<BoxDetection> detections;
    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    int numClasses = static_cast<int>(outputShape[2]) - 5;
    int elementsInBatch = static_cast<int>(outputShape[1] * outputShape[2]);

    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
    {
        float clsConf = it[4];

        if (clsConf > confThreshold)
        {
            int centerX = static_cast<int>(it[0]);
            int centerY = static_cast<int>(it[1]);
            int width = static_cast<int>(it[2]);
            int height = static_cast<int>(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            this->getBestClassInfoYoloV5(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            auto box = YOLOUtils::RectBox(left, top, width, height);
            auto scaledBox = YOLOUtils::scaleCoords(resizedImageShape, box, originalImageSize, true);

            BoxDetection detection;
            detection.box = scaledBox;
            detection.conf = confidence;
            detection.classId = classId;
            detection.label = classNames[classId];

            detections.push_back(detection);
        }
    }

    // Extract BoundingBoxes and corresponding scores for NMS
    std::vector<YOLOUtils::RectBox> boundingBoxes;
    std::vector<float> scores;
    for (const auto& detection : detections)
    {
        boundingBoxes.push_back(detection.box);
        scores.push_back(detection.conf);
    }

    std::vector<int> nmsIndices;
    DetUtils::NMSBoxes(boundingBoxes, scores, confThreshold, iouThreshold, nmsIndices);

    std::vector<BoxDetection> finalDetections;
    for (int idx : nmsIndices) { finalDetections.push_back(detections[idx]); }

    return finalDetections;
}

inline void YOLO11Detect::getBestClassInfoYoloV5(std::vector<float>::iterator it, const int& numClasses,
                                                 float& bestConf, int& bestClassId)
{
    bestClassId = 5;
    bestConf = 0.0f;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }
}


// Detect function implementation
inline std::vector<BoxDetection> YOLO11Detect::detect(const cv::Mat& image, float confThreshold, float iouThreshold)
{
    // 使用ScopedTimer测量整个检测过程的耗时
    ScopedTimer timer("DetectTask");

    // 预处理阶段
    float* blobPtr = nullptr; // 指向预处理后图像数据的指针
    // 定义输入张量形状 [batch_size, channels, height, width]
    std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height, inputImageShape.width};

    // 图像预处理：调整大小、归一化、转换为CHW格式
    cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);

    // 计算输入张量的总元素数量
    size_t inputTensorSize = YOLOUtils::vectorProduct(inputTensorShape);

    // 将blob数据转换为vector格式供ONNX Runtime使用
    std::vector inputTensorValues(blobPtr, blobPtr + inputTensorSize);

    delete[] blobPtr; // 释放blob内存

    // 创建ONNX Runtime内存信息对象（可缓存复用）
    static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 使用预处理数据创建输入张量对象
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorSize,
        inputTensorShape.data(),
        inputTensorShape.size()
    );

    // 运行ONNX Runtime会话进行推理，获取输出张量
    std::vector<Ort::Value> outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        &inputTensor,
        numInputNodes,
        outputNames.data(),
        numOutputNodes
    );

    // 根据输入张量形状确定调整后的图像尺寸
    cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]), static_cast<int>(inputTensorShape[2]));

    // 对输出张量进行后处理，获取检测结果
    auto detections = postprocess(image.size(), resizedImageShape, outputTensors, confThreshold, iouThreshold);
    return detections; // 返回检测结果向量
}
