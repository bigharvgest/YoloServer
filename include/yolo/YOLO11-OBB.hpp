#pragma once

/**
 * @file YOLO11-OBB.hpp
 * @brief Header file for the YOLO11OBBDetector class, responsible for object detection
 *        using the YOLOv11 OBB model with optimized performance for minimal latency.
 */

// Include necessary ONNX Runtime and OpenCV headers
#define NOMINMAX

#include <yolo/YOLO11-Base.hpp>
#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

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
#include <cmath>


// Include debug and custom ScopedTimer tools for performance measurement
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"


/**
 * @brief Struct to represent an Oriented bounding box (OBB) in xywhr format.
 */
struct OBBRectBox
{
    float x; // x-coordinate of the center
    float y; // y-coordinate of the center
    float width; // width of the box
    float height; // height of the box
    float angle; // rotation angle in radians

    OBBRectBox() : x(0), y(0), width(0), height(0), angle(0)
    {
    }

    OBBRectBox(float x_, float y_, float width_, float height_, float angle_)
        : x(x_), y(y_), width(width_), height(height_), angle(angle_)
    {
    }
};

// 手动实现 to_json
inline void to_json(nlohmann::json& j, const OBBRectBox& box)
{
    j = nlohmann::json{
        {"x", box.x},
        {"y", box.y},
        {"width", box.width},
        {"height", box.height},
        {"angle", box.angle},
    };
}

/**
 * @brief Struct to represent a detection with an oriented bounding box.
 */
struct OBBDetection
{
    OBBRectBox box; // Oriented bounding box in xywhr format
    float conf{}; // Confidence score
    int classId{}; // Class ID
    std::string label{}; // Class label

    OBBDetection() = default;

    OBBDetection(const OBBRectBox& box_, float conf_, int classId_, const std::string& label_)
        : box(box_), conf(conf_), classId(classId_), label(label_)
    {
    }
};

// 手动实现 to_json
inline void to_json(nlohmann::json& j, const OBBDetection& obb)
{
    j = nlohmann::json{
        {"box", obb.box},
        {"conf", obb.conf},
        {"classId", obb.classId},
        {"label", obb.label},
    };
}


/**
 * @namespace OBB_NMS
 * @brief Namespace containing NMS functions for the OBB format.
 */
namespace OBB_NMS
{
    static constexpr float EPS = 1e-7f;

    /**
     * @brief Computes covariance matrix components for a single OBB.
     * @param box Input oriented bounding box.
     * @param out1 First component (a).
     * @param out2 Second component (b).
     * @param out3 Third component (c).
     */
    inline void getCovarianceComponents(const OBBRectBox& box, float& out1, float& out2, float& out3)
    {
        float a = (box.width * box.width) / 12.0f;
        float b = (box.height * box.height) / 12.0f;
        float angle = box.angle;

        float cos_theta = std::cos(angle);
        float sin_theta = std::sin(angle);
        float cos_sq = cos_theta * cos_theta;
        float sin_sq = sin_theta * sin_theta;

        out1 = a * cos_sq + b * sin_sq;
        out2 = a * sin_sq + b * cos_sq;
        out3 = (a - b) * cos_theta * sin_theta;
    }

    /**
     * @brief Computes IoU matrix between two sets of OBBs.
     * @param obb1 First set of OBBs.
     * @param obb2 Second set of OBBs.
     * @param eps Small constant for numerical stability.
     * @return 2D vector of IoU values.
     */
    inline std::vector<std::vector<float>> batchProbiou(const std::vector<OBBRectBox>& obb1,
                                                        const std::vector<OBBRectBox>& obb2, float eps = EPS)
    {
        size_t N = obb1.size();
        size_t M = obb2.size();
        std::vector<std::vector<float>> iou_matrix(N, std::vector<float>(M, 0.0f));

        for (size_t i = 0; i < N; ++i)
        {
            const OBBRectBox& box1 = obb1[i];
            float x1 = box1.x, y1 = box1.y;
            float a1, b1, c1;
            getCovarianceComponents(box1, a1, b1, c1);

            for (size_t j = 0; j < M; ++j)
            {
                const OBBRectBox& box2 = obb2[j];
                float x2 = box2.x, y2 = box2.y;
                float a2, b2, c2;
                getCovarianceComponents(box2, a2, b2, c2);

                // Compute denominator
                float denom = (a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2) + eps;

                // Terms t1 and t2
                float dx = x1 - x2;
                float dy = y1 - y2;
                float t1 = ((a1 + a2) * dy * dy + (b1 + b2) * dx * dx) * 0.25f / denom;
                float t2 = ((c1 + c2) * (x2 - x1) * dy) * 0.5f / denom;

                // Term t3
                float term1 = a1 * b1 - c1 * c1;
                term1 = std::max(term1, 0.0f);
                float term2 = a2 * b2 - c2 * c2;
                term2 = std::max(term2, 0.0f);
                float sqrt_term = std::sqrt(term1 * term2);

                float numerator = (a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2);
                float denominator_t3 = 4.0f * sqrt_term + eps;
                float t3 = 0.5f * std::log(numerator / denominator_t3 + eps);

                // Compute final IoU
                float bd = t1 + t2 + t3;
                bd = std::clamp(bd, eps, 100.0f);
                float hd = std::sqrt(1.0f - std::exp(-bd) + eps);
                iou_matrix[i][j] = 1.0f - hd;
            }
        }
        return iou_matrix;
    }

    /**
     * @brief Performs rotated NMS on sorted OBBs.
     * @param sorted_boxes Boxes sorted by confidence.
     * @param iou_thres Threshold for IoU suppression.
     * @return Indices of boxes to keep.
     */
    inline std::vector<int> nmsRotatedImpl(const std::vector<OBBRectBox>& sorted_boxes, float iou_thres)
    {
        auto ious = batchProbiou(sorted_boxes, sorted_boxes);
        std::vector<int> keep;
        const int n = sorted_boxes.size();

        for (int j = 0; j < n; ++j)
        {
            bool keep_j = true;
            for (int i = 0; i < j; ++i)
            {
                if (ious[i][j] >= iou_thres)
                {
                    keep_j = false;
                    break;
                }
            }
            if (keep_j) keep.push_back(j);
        }
        return keep;
    }

    /**
     * @brief Main NMS function for rotated boxes.
     * @param boxes Input boxes.
     * @param scores Confidence scores.
     * @param threshold IoU threshold.
     * @return Indices of boxes to keep.
     */
    inline std::vector<int> nmsRotated(const std::vector<OBBRectBox>& boxes, const std::vector<float>& scores,
                                       float threshold = 0.75f)
    {
        // Sort indices based on scores
        std::vector<int> indices(boxes.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });

        // Create sorted boxes list
        std::vector<OBBRectBox> sorted_boxes;
        for (int idx : indices) { sorted_boxes.push_back(boxes[idx]); }

        // Perform NMS
        std::vector<int> keep = nmsRotatedImpl(sorted_boxes, threshold);

        // Map back to original indices
        std::vector<int> keep_indices;
        for (int k : keep) { keep_indices.push_back(indices[k]); }
        return keep_indices;
    }

    /**
     * @brief Applies NMS to detections and returns filtered results.
     * @param input_detections Input detections.
     * @param conf_thres Confidence threshold.
     * @param iou_thres IoU threshold.
     * @param max_det Maximum detections to keep.
     * @return Filtered detections after NMS.
     */
    inline std::vector<OBBDetection> nonMaxSuppression(
        const std::vector<OBBDetection>& input_detections,
        float conf_thres = 0.25f,
        float iou_thres = 0.75f,
        int max_det = 1000)
    {
        // Filter by confidence
        std::vector<OBBDetection> candidates;
        for (const auto& det : input_detections) { if (det.conf > conf_thres) { candidates.push_back(det); } }
        if (candidates.empty()) return {};

        // Extract boxes and scores
        std::vector<OBBRectBox> boxes;
        std::vector<float> scores;
        for (const auto& det : candidates)
        {
            boxes.push_back(det.box);
            scores.push_back(det.conf);
        }

        // Run NMS
        std::vector<int> keep_indices = nmsRotated(boxes, scores, iou_thres);

        // Collect results
        std::vector<OBBDetection> results;
        for (int idx : keep_indices)
        {
            if (results.size() >= max_det) break;
            results.push_back(candidates[idx]);
        }
        return results;
    }
}


/**
 * @namespace OBBUtils
 * @brief Namespace containing utility functions for the YOLO11OBBDetector.
 */
namespace OBBUtils
{
    /**
    * @brief Draws oriented bounding boxes with rotation and labels on the image based on detections
    *
    * @param image Image on which to draw.
    * @param detections Vector of detections.
    * @param classNames Vector of class names corresponding to object IDs.
    * @param colors Vector of colors for each class.
    */
    inline void drawBoundingBox(cv::Mat& image, const std::vector<OBBDetection>& detections,
                                const std::vector<std::string>& classNames, const std::vector<cv::Scalar>& colors)
    {
        for (const auto& detection : detections)
        {
            if (detection.classId < 0 || static_cast<size_t>(detection.classId) >= classNames.size()) continue;

            // Convert angle from radians to degrees for OpenCV
            float angle_deg = detection.box.angle * 180.0f / CV_PI;

            cv::RotatedRect rect(cv::Point2f(detection.box.x, detection.box.y),
                                 cv::Size2f(detection.box.width, detection.box.height),
                                 angle_deg);

            // Convert rotated rectangle to polygon points
            cv::Mat points_mat;
            cv::boxPoints(rect, points_mat);
            points_mat.convertTo(points_mat, CV_32SC1);

            int thickness = 1;
            int baseline = 0;
            float fontScale = 0.6;
            // Draw bounding box
            cv::Scalar color = colors[detection.classId % colors.size()];
            cv::polylines(image, points_mat, true, color, thickness, cv::LINE_AA);

            // Prepare label
            std::string label = classNames[detection.classId] + ": " + cv::format("%.1f%%", detection.conf * 100);
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, fontScale, thickness, &baseline);

            // Calculate label position using bounding rect of rotated rectangle
            cv::Rect brect = rect.boundingRect();
            int x = brect.x;
            int y = brect.y - labelSize.height - baseline;

            // Adjust label position if it goes off-screen
            if (y < 0)
            {
                y = brect.y + brect.height;
                if (y + labelSize.height > image.rows) { y = image.rows - labelSize.height; }
            }

            x = std::max(0, std::min(x, image.cols - labelSize.width));

            // Draw label background (darker version of box color)
            cv::Scalar labelBgColor = color * 0.6;
            cv::rectangle(image, cv::Rect(x, y, labelSize.width, labelSize.height + baseline),
                          labelBgColor, cv::FILLED);

            // Draw label text
            cv::putText(image, label, cv::Point(x, y + labelSize.height),
                        cv::FONT_HERSHEY_DUPLEX, fontScale, cv::Scalar::all(255),
                        thickness, cv::LINE_AA);
        }
    }
}

/**
 * @brief YOLO11-OBB-Detector class handles loading the YOLO model, preprocessing images, running inference, and postprocessing results.
 */
class YOLO11OBB : public YOLO11Model
{
public:
    /**
     * @brief Constructor to initialize the YOLO detector with model and label paths.
     *
     * @param modelBuffer 模型二进制
     * @param useGPU Whether to use GPU for inference (default is false).
     * @param device GPU device ID to use (default is 0).
     */
    explicit YOLO11OBB(const std::vector<char>& modelBuffer, bool useGPU = false, int device = 0);

    std::string getTask() const override { return "obb"; }

    /**
     * @brief Runs detection on the provided image.
     *
     * @param image Input image for detection.
     * @param confThreshold Confidence threshold to filter detections (default is 0.4).
     * @param iouThreshold IoU threshold for Non-Maximum Suppression (default is 0.45).
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<OBBDetection> detect(const cv::Mat& image, float confThreshold = 0.25f, float iouThreshold = 0.25);

    /**
     * @brief Draws bounding boxes on the image based on detections.
     *
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     */
    void drawBoundingBox(cv::Mat& image, const std::vector<OBBDetection>& detections) const
    {
        OBBUtils::drawBoundingBox(image, detections, classNames, classColors);
    }

private:
    Ort::Env env{nullptr}; // ONNX Runtime environment
    Ort::SessionOptions sessionOptions{nullptr}; // Session options for ONNX Runtime
    Ort::Session session{nullptr}; // ONNX Runtime session for running inference
    bool isDynamicInputShape{}; // Flag indicating if input shape is dynamic
    cv::Size inputImageShape; // Expected input image shape for the model

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
     * @brief Postprocesses the model output to extract detections with oriented bounding boxes.
     *
     * @param originalImageSize Size of the original input image.
     * @param resizedImageShape Size of the image after preprocessing.
     * @param outputTensors Vector of output tensors from the model.
     * @param confThreshold Confidence threshold to filter detections.
     * @param iouThreshold IoU threshold for Non-Maximum Suppression (using ProbIoU for rotated boxes).
     * @return std::vector<Detection> Vector of detections with oriented bounding boxes.
     */
    std::vector<OBBDetection> postprocess(const cv::Size& originalImageSize,
                                          const cv::Size& resizedImageShape,
                                          const std::vector<Ort::Value>& outputTensors,
                                          float confThreshold, float iouThreshold,
                                          int topk = 500); // Default argument here
};

// Implementation of YOLO11OBBDetector constructor
inline YOLO11OBB::YOLO11OBB(const std::vector<char>& modelBuffer, bool useGPU, int device)
{
    // Initialize ONNX Runtime environment with warning level
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLO_OBB");
    sessionOptions = Ort::SessionOptions();

    // Set number of intra-op threads for parallelism
    sessionOptions.SetIntraOpNumThreads(std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
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

    // Allocate and store input node names
    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

    // Allocate and store output node names
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

    // Set the expected input image shape based on the model's input tensor
    if (inputTensorShapeVec.size() >= 4)
    {
        inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]), static_cast<int>(inputTensorShapeVec[2]));
    }
    else { throw std::runtime_error("Invalid input tensor shape."); }

    // Get the number of input and output nodes
    numInputNodes = session.GetInputCount();
    numOutputNodes = session.GetOutputCount();

    // Load class names and generate corresponding colors
    Ort::AllocatorWithDefaultOptions ort_alloc;
    const auto model_metadata = session.GetModelMetadata();
    Ort::AllocatedStringPtr search = model_metadata.LookupCustomMetadataMapAllocated("names", ort_alloc);
    if (search != nullptr) { classNames = YOLOUtils::parseClassNames(search.get()); }
    classColors = YOLOUtils::generateColors(classNames);

    std::cout << "Model loaded successfully with " << numInputNodes << " input nodes and " << numOutputNodes <<
        " output nodes." << std::endl;
}

// Preprocess function implementation
inline cv::Mat YOLO11OBB::preprocess(const cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
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

inline std::vector<OBBDetection> YOLO11OBB::postprocess(
    const cv::Size& originalImageSize,
    const cv::Size& resizedImageShape,
    const std::vector<Ort::Value>& outputTensors,
    float confThreshold,
    float iouThreshold,
    int topk)
{
    ScopedTimer timer("postprocessing");
    std::vector<OBBDetection> detections;

    // Get raw output data and shape (assumed [1, num_features, num_detections])
    const float* rawOutput = outputTensors[0].GetTensorData<float>();
    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int num_features = static_cast<int>(outputShape[1]);
    int num_detections = static_cast<int>(outputShape[2]);
    if (num_detections == 0) { return detections; }

    // Determine number of labels/classes (layout: [x, y, w, h, scores..., angle])
    int num_labels = num_features - 5;
    if (num_labels <= 0) { return detections; }

    // Compute letterbox parameters.
    float inp_w = static_cast<float>(resizedImageShape.width);
    float inp_h = static_cast<float>(resizedImageShape.height);
    float orig_w = static_cast<float>(originalImageSize.width);
    float orig_h = static_cast<float>(originalImageSize.height);
    float r = std::min(inp_h / orig_h, inp_w / orig_w);
    int padw = std::round(orig_w * r);
    int padh = std::round(orig_h * r);
    float dw = (inp_w - padw) / 2.0f;
    float dh = (inp_h - padh) / 2.0f;
    float ratio = 1.0f / r;

    // Wrap raw output data into a cv::Mat and transpose it.
    // After transposition, each row corresponds to one detection with layout:
    // [x, y, w, h, score_0, score_1, …, score_(num_labels-1), angle]
    cv::Mat output = cv::Mat(num_features, num_detections, CV_32F, const_cast<float*>(rawOutput));
    output = output.t(); // Now shape: [num_detections, num_features]

    // Extract detections without clamping.
    std::vector<OBBRectBox> obbs;
    std::vector<float> scores;
    std::vector<int> labels;
    for (int i = 0; i < num_detections; ++i)
    {
        float* row_ptr = output.ptr<float>(i);
        // Extract raw bbox parameters in letterbox coordinate space.
        float x = row_ptr[0];
        float y = row_ptr[1];
        float w = row_ptr[2];
        float h = row_ptr[3];

        // Extract class scores and determine the best class.
        float* scores_ptr = row_ptr + 4;
        float maxScore = -FLT_MAX;
        int classId = -1;
        for (int j = 0; j < num_labels; j++)
        {
            float score = scores_ptr[j];
            if (score > maxScore)
            {
                maxScore = score;
                classId = j;
            }
        }

        // Angle is stored right after the scores.
        float angle = row_ptr[4 + num_labels];

        if (maxScore > confThreshold)
        {
            // Correct the box coordinates with letterbox offsets and scaling.
            float cx = (x - dw) * ratio;
            float cy = (y - dh) * ratio;
            float bw = w * ratio;
            float bh = h * ratio;

            OBBRectBox obb(cx, cy, bw, bh, angle);
            obbs.push_back(obb);
            scores.push_back(maxScore);
            labels.push_back(classId);
        }
    }

    // Combine detections into a vector<Detection> for NMS.
    std::vector<OBBDetection> detectionsForNMS;
    for (size_t i = 0; i < obbs.size(); i++)
    {
        detectionsForNMS.emplace_back(OBBDetection{obbs[i], scores[i], labels[i], classNames[labels[i]]});
    }

    for (auto& det : detectionsForNMS)
    {
        det.box.x = std::min(std::max(det.box.x, 0.f), orig_w);
        det.box.y = std::min(std::max(det.box.y, 0.f), orig_h);
        det.box.width = std::min(std::max(det.box.width, 0.f), orig_w);
        det.box.height = std::min(std::max(det.box.height, 0.f), orig_h);
    }

    // Perform rotated NMS.
    std::vector<OBBDetection> post_nms_detections = OBB_NMS::nonMaxSuppression(
        detectionsForNMS, confThreshold, iouThreshold, topk);


    DEBUG_PRINT("Postprocessing completed");
    return post_nms_detections;
}


// Detect function implementation
inline std::vector<OBBDetection> YOLO11OBB::detect(const cv::Mat& image, float confThreshold, float iouThreshold)
{
    ScopedTimer timer("OBBTask");

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

    // Determine the resized image shape based on input tensor shape
    cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]), static_cast<int>(inputTensorShape[2]));

    std::vector<OBBDetection> detections = postprocess(image.size(), resizedImageShape, outputTensors, confThreshold,
                                                       iouThreshold, 100);

    return detections; // Return the vector of detections
}
