#pragma once

#define NOMINMAX

#include <yolo/YOLO11-Base.hpp>

#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Include debug and custom ScopedTimer tools for performance measurement
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"


struct Segmentation
{
    YOLOUtils::RectBox box;
    float conf{0.f};
    int classId{0};
    std::string label{};
    cv::Mat mask; // Single-channel (8UC1) mask in full resolution
};

inline void to_json(nlohmann::json& j, const Segmentation& segmentation)
{
    j = nlohmann::json{
        {"box", segmentation.box},
        {"conf", segmentation.conf},
        {"classId", segmentation.classId},
        {"label", segmentation.label}
    };
}

// ============================================================================
// Utility Namespace
// ============================================================================
namespace SegUtils
{
    inline cv::Mat sigmoid(const cv::Mat& src)
    {
        cv::Mat dst;
        cv::exp(-src, dst);
        dst = 1.0 / (1.0 + dst);
        return dst;
    }

    inline void NMSBoxes(const std::vector<YOLOUtils::RectBox>& boxes,
                         const std::vector<float>& scores,
                         float scoreThreshold,
                         float nmsThreshold,
                         std::vector<int>& indices)
    {
        indices.clear();
        if (boxes.empty()) { return; }

        std::vector<int> order;
        order.reserve(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) { if (scores[i] >= scoreThreshold) { order.push_back((int)i); } }
        if (order.empty()) return;

        std::sort(order.begin(), order.end(),
                  [&scores](int a, int b) { return scores[a] > scores[b]; });

        std::vector<float> areas(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) { areas[i] = (float)(boxes[i].width * boxes[i].height); }

        std::vector<bool> suppressed(boxes.size(), false);
        for (size_t i = 0; i < order.size(); ++i)
        {
            int idx = order[i];
            if (suppressed[idx]) continue;

            indices.push_back(idx);

            for (size_t j = i + 1; j < order.size(); ++j)
            {
                int idx2 = order[j];
                if (suppressed[idx2]) continue;

                const YOLOUtils::RectBox& a = boxes[idx];
                const YOLOUtils::RectBox& b = boxes[idx2];
                int interX1 = std::max(a.x, b.x);
                int interY1 = std::max(a.y, b.y);
                int interX2 = std::min(a.x + a.width, b.x + b.width);
                int interY2 = std::min(a.y + a.height, b.y + b.height);

                int w = interX2 - interX1;
                int h = interY2 - interY1;
                if (w > 0 && h > 0)
                {
                    float interArea = (float)(w * h);
                    float unionArea = areas[idx] + areas[idx2] - interArea;
                    float iou = (unionArea > 0.f) ? (interArea / unionArea) : 0.f;
                    if (iou > nmsThreshold) { suppressed[idx2] = true; }
                }
            }
        }
    }
} // namespace utils

// ============================================================================
// YOLOv11SegDetector Class
// ============================================================================
class YOLO11Segment : public YOLO11Model
{
public:
    explicit YOLO11Segment(const std::string& modelPath,
                           bool useGPU = false);

    std::string getTask() const override { return "segment"; }

    // Main API
    std::vector<Segmentation> segment(const cv::Mat& image,
                                      float confThreshold = 0.4f,
                                      float iouThreshold = 0.45f);

    // Draw results
    void drawSegmentationsAndBoxes(cv::Mat& image,
                                   const std::vector<Segmentation>& results,
                                   float maskAlpha = 0.5f) const;

    void drawSegmentations(cv::Mat& image,
                           const std::vector<Segmentation>& results,
                           float maskAlpha = 0.5f) const;
    // Accessors
    const std::vector<std::string>& getClassNames() const { return classNames; }
    const std::vector<cv::Scalar>& getClassColors() const { return classColors; }

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

    std::vector<std::string> classNames{};
    std::vector<cv::Scalar> classColors{};

    // Helpers
    cv::Mat preprocess(const cv::Mat& image,
                       float*& blobPtr,
                       std::vector<int64_t>& inputTensorShape);

    std::vector<Segmentation> postprocess(const cv::Size& origSize,
                                          const cv::Size& letterboxSize,
                                          const std::vector<Ort::Value>& outputs,
                                          float confThreshold,
                                          float iouThreshold);
};

inline YOLO11Segment::YOLO11Segment(const std::string& modelPath,
                                    bool useGPU)
{
    ScopedTimer timer("YOLOv11SegDetector Constructor");

    // 初始化 ONNX Runtime 环境，设置警告级别为警告
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLO_DETECT");
    // 创建 ONNX Runtime 会话选项
    sessionOptions = Ort::SessionOptions();

    sessionOptions.SetIntraOpNumThreads(std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

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

#ifdef _WIN32
    std::wstring w_modelPath = YOLOUtils::utf8_to_wstring(modelPath);
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    numInputNodes = session.GetInputCount();
    numOutputNodes = session.GetOutputCount();

    Ort::AllocatorWithDefaultOptions allocator;

    // Input
    {
        auto inNameAlloc = session.GetInputNameAllocated(0, allocator);
        inputNodeNameAllocatedStrings.emplace_back(std::move(inNameAlloc));
        inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

        auto inTypeInfo = session.GetInputTypeInfo(0);
        auto inShape = inTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

        if (inShape.size() == 4)
        {
            if (inShape[2] == -1 || inShape[3] == -1)
            {
                isDynamicInputShape = true;
                inputImageShape = cv::Size(640, 640); // Fallback if dynamic
            }
            else { inputImageShape = cv::Size(static_cast<int>(inShape[3]), static_cast<int>(inShape[2])); }
        }
        else { throw std::runtime_error("Model input is not 4D! Expect [N, C, H, W]."); }
    }

    // Outputs
    if (numOutputNodes != 2) { throw std::runtime_error("Expected exactly 2 output nodes: output0 and output1."); }

    for (size_t i = 0; i < numOutputNodes; ++i)
    {
        auto outNameAlloc = session.GetOutputNameAllocated(i, allocator);
        outputNodeNameAllocatedStrings.emplace_back(std::move(outNameAlloc));
        outputNames.push_back(outputNodeNameAllocatedStrings.back().get());
    }

    const auto model_metadata = session.GetModelMetadata();
    Ort::AllocatedStringPtr search = model_metadata.LookupCustomMetadataMapAllocated("names", allocator);
    if (search != nullptr) { classNames = YOLOUtils::parseClassNames(search.get()); }
    classColors = YOLOUtils::generateColors(classNames);

    std::cout << "[INFO] YOLO11Seg loaded: " << modelPath << std::endl
        << "      Input shape: " << inputImageShape
        << (isDynamicInputShape ? " (dynamic)" : "") << std::endl
        << "      #Outputs   : " << numOutputNodes << std::endl
        << "      #Classes   : " << classNames.size() << std::endl;
}

inline cv::Mat YOLO11Segment::preprocess(const cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
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

std::vector<Segmentation> YOLO11Segment::postprocess(
    const cv::Size& origSize,
    const cv::Size& letterboxSize,
    const std::vector<Ort::Value>& outputs,
    float confThreshold,
    float iouThreshold)
{
    ScopedTimer timer("PostprocessSeg");

    std::vector<Segmentation> results;

    // Validate outputs size
    if (outputs.size() < 2)
    {
        throw std::runtime_error("Insufficient outputs from the model. Expected at least 2 outputs.");
    }

    // Extract outputs
    const float* output0_ptr = outputs[0].GetTensorData<float>();
    const float* output1_ptr = outputs[1].GetTensorData<float>();

    // Get shapes
    auto shape0 = outputs[0].GetTensorTypeAndShapeInfo().GetShape(); // [1, 116, num_detections]
    auto shape1 = outputs[1].GetTensorTypeAndShapeInfo().GetShape(); // [1, 32, maskH, maskW]

    if (shape1.size() != 4 || shape1[0] != 1 || shape1[1] != 32) throw std::runtime_error(
        "Unexpected output1 shape. Expected [1, 32, maskH, maskW].");

    const size_t num_features = shape0[1]; // e.g 80 class + 4 bbox parms + 32 seg masks = 116
    const size_t num_detections = shape0[2];

    // Early exit if no detections
    if (num_detections == 0) { return results; }

    const int numClasses = static_cast<int>(num_features - 4 - 32); // Corrected number of classes

    // Validate numClasses
    if (numClasses <= 0) { throw std::runtime_error("Invalid number of classes."); }

    const int numBoxes = static_cast<int>(num_detections);
    const int maskH = static_cast<int>(shape1[2]);
    const int maskW = static_cast<int>(shape1[3]);

    // Constants from model architecture
    constexpr int BOX_OFFSET = 0;
    constexpr int CLASS_CONF_OFFSET = 4;
    const int MASK_COEFF_OFFSET = numClasses + CLASS_CONF_OFFSET;

    // 1. Process prototype masks
    // Store all prototype masks in a vector for easy access
    std::vector<cv::Mat> prototypeMasks;
    prototypeMasks.reserve(32);
    for (int m = 0; m < 32; ++m)
    {
        // Each mask is maskH x maskW
        cv::Mat proto(maskH, maskW, CV_32F, const_cast<float*>(output1_ptr + m * maskH * maskW));
        prototypeMasks.emplace_back(proto.clone()); // Clone to ensure data integrity
    }

    // 2. Process detections
    std::vector<YOLOUtils::RectBox> boxes;
    boxes.reserve(numBoxes);
    std::vector<float> confidences;
    confidences.reserve(numBoxes);
    std::vector<int> classIds;
    classIds.reserve(numBoxes);
    std::vector<std::vector<float>> maskCoefficientsList;
    maskCoefficientsList.reserve(numBoxes);

    for (int i = 0; i < numBoxes; ++i)
    {
        // Extract box coordinates
        float xc = output0_ptr[BOX_OFFSET * numBoxes + i];
        float yc = output0_ptr[(BOX_OFFSET + 1) * numBoxes + i];
        float w = output0_ptr[(BOX_OFFSET + 2) * numBoxes + i];
        float h = output0_ptr[(BOX_OFFSET + 3) * numBoxes + i];

        // Convert to xyxy format
        YOLOUtils::RectBox box{
            static_cast<int>(std::round(xc - w / 2.0f)),
            static_cast<int>(std::round(yc - h / 2.0f)),
            static_cast<int>(std::round(w)),
            static_cast<int>(std::round(h))
        };

        // Get class confidence
        float maxConf = 0.0f;
        int classId = -1;
        for (int c = 0; c < numClasses; ++c)
        {
            float conf = output0_ptr[(CLASS_CONF_OFFSET + c) * numBoxes + i];
            if (conf > maxConf)
            {
                maxConf = conf;
                classId = c;
            }
        }

        if (maxConf < confThreshold) continue;

        // Store detection
        boxes.push_back(box);
        confidences.push_back(maxConf);
        classIds.push_back(classId);

        // Store mask coefficients
        std::vector<float> maskCoeffs(32);
        for (int m = 0; m < 32; ++m) { maskCoeffs[m] = output0_ptr[(MASK_COEFF_OFFSET + m) * numBoxes + i]; }
        maskCoefficientsList.emplace_back(std::move(maskCoeffs));
    }

    // Early exit if no boxes after confidence threshold
    if (boxes.empty()) { return results; }

    // 3. Apply NMS
    std::vector<int> nmsIndices;
    SegUtils::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, nmsIndices);

    if (nmsIndices.empty()) { return results; }

    // 4. Prepare final results
    results.reserve(nmsIndices.size());

    // Calculate letterbox parameters
    const float gain = std::min(static_cast<float>(letterboxSize.height) / origSize.height,
                                static_cast<float>(letterboxSize.width) / origSize.width);
    const int scaledW = static_cast<int>(origSize.width * gain);
    const int scaledH = static_cast<int>(origSize.height * gain);
    const float padW = (letterboxSize.width - scaledW) / 2.0f;
    const float padH = (letterboxSize.height - scaledH) / 2.0f;

    // Precompute mask scaling factors
    const float maskScaleX = static_cast<float>(maskW) / letterboxSize.width;
    const float maskScaleY = static_cast<float>(maskH) / letterboxSize.height;

    for (const int idx : nmsIndices)
    {
        Segmentation seg;
        seg.box = boxes[idx];
        seg.conf = confidences[idx];
        seg.classId = classIds[idx];
        seg.label = classNames[classIds[idx]];

        // 5. Scale box to original image
        seg.box = YOLOUtils::scaleCoords(letterboxSize, seg.box, origSize, true);

        // 6. Process mask
        const auto& maskCoeffs = maskCoefficientsList[idx];

        // Linear combination of prototype masks
        cv::Mat finalMask = cv::Mat::zeros(maskH, maskW, CV_32F);
        for (int m = 0; m < 32; ++m) { finalMask += maskCoeffs[m] * prototypeMasks[m]; }

        // Apply sigmoid activation
        finalMask = SegUtils::sigmoid(finalMask);

        // Crop mask to letterbox area with a slight padding to avoid border issues
        int x1 = static_cast<int>(std::round((padW - 0.1f) * maskScaleX));
        int y1 = static_cast<int>(std::round((padH - 0.1f) * maskScaleY));
        int x2 = static_cast<int>(std::round((letterboxSize.width - padW + 0.1f) * maskScaleX));
        int y2 = static_cast<int>(std::round((letterboxSize.height - padH + 0.1f) * maskScaleY));

        // Ensure coordinates are within mask bounds
        x1 = std::max(0, std::min(x1, maskW - 1));
        y1 = std::max(0, std::min(y1, maskH - 1));
        x2 = std::max(x1, std::min(x2, maskW));
        y2 = std::max(y1, std::min(y2, maskH));

        // Handle cases where cropping might result in zero area
        if (x2 <= x1 || y2 <= y1)
        {
            // Skip this mask as cropping is invalid
            continue;
        }

        cv::Rect cropRect(x1, y1, x2 - x1, y2 - y1);
        cv::Mat croppedMask = finalMask(cropRect).clone(); // Clone to ensure data integrity

        // Resize to original dimensions
        cv::Mat resizedMask;
        cv::resize(croppedMask, resizedMask, origSize, 0, 0, cv::INTER_LINEAR);

        // Threshold and convert to binary
        cv::Mat binaryMask;
        cv::threshold(resizedMask, binaryMask, 0.5, 255.0, cv::THRESH_BINARY);
        binaryMask.convertTo(binaryMask, CV_8U);

        // Crop to bounding box
        cv::Mat finalBinaryMask = cv::Mat::zeros(origSize, CV_8U);
        cv::Rect roi(seg.box.x, seg.box.y, seg.box.width, seg.box.height);
        roi &= cv::Rect(0, 0, binaryMask.cols, binaryMask.rows); // Ensure ROI is within mask
        if (roi.area() > 0) { binaryMask(roi).copyTo(finalBinaryMask(roi)); }

        seg.mask = finalBinaryMask;
        results.push_back(seg);
    }

    return results;
}

inline void YOLO11Segment::drawSegmentationsAndBoxes(cv::Mat& image,
                                                     const std::vector<Segmentation>& results,
                                                     float maskAlpha) const
{
    for (const auto& [box, conf, classId,label, mask] : results)
    {
        cv::Scalar color = classColors[classId % classColors.size()];

        // -----------------------------
        // 1. Draw Bounding Box
        // -----------------------------
        cv::rectangle(image,
                      cv::Point(box.x, box.y),
                      cv::Point(box.x + box.width, box.y + box.height),
                      color, 1);

        // -----------------------------
        // 2. Draw Label
        // -----------------------------
        std::string drawLabel = label + " " + std::to_string(static_cast<int>(conf * 100)) + "%";
        int baseLine = 0;
        double fontScale = 0.5;
        int thickness = 1;
        cv::Size labelSize = cv::getTextSize(drawLabel, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);
        int top = std::max(box.y, labelSize.height + 5);
        cv::rectangle(image,
                      cv::Point(box.x, top - labelSize.height - 5),
                      cv::Point(box.x + labelSize.width + 5, top),
                      color, cv::FILLED);
        cv::putText(image, drawLabel,
                    cv::Point(box.x + 2, top - 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    cv::Scalar(255, 255, 255),
                    thickness);

        // -----------------------------
        // 3. Apply Segmentation Mask
        // -----------------------------
        if (!mask.empty())
        {
            // Ensure the mask is single-channel
            cv::Mat mask_gray;
            if (mask.channels() == 3) { cv::cvtColor(mask, mask_gray, cv::COLOR_BGR2GRAY); }
            else { mask_gray = mask.clone(); }

            // Threshold the mask to binary (object: 255, background: 0)
            cv::Mat mask_binary;
            cv::threshold(mask_gray, mask_binary, 127, 255, cv::THRESH_BINARY);

            // Create a colored version of the mask
            cv::Mat colored_mask;
            cv::cvtColor(mask_binary, colored_mask, cv::COLOR_GRAY2BGR);
            colored_mask.setTo(color, mask_binary); // Apply color where mask is present

            // Blend the colored mask with the original image
            cv::addWeighted(image, 1.0, colored_mask, maskAlpha, 0, image);
        }
    }
}


inline void YOLO11Segment::drawSegmentations(cv::Mat& image,
                                             const std::vector<Segmentation>& results,
                                             float maskAlpha) const
{
    for (const auto& seg : results)
    {
        cv::Scalar color = classColors[seg.classId % classColors.size()];

        // -----------------------------
        // Draw Segmentation Mask Only
        // -----------------------------
        if (!seg.mask.empty())
        {
            // Ensure the mask is single-channel
            cv::Mat mask_gray;
            if (seg.mask.channels() == 3) { cv::cvtColor(seg.mask, mask_gray, cv::COLOR_BGR2GRAY); }
            else { mask_gray = seg.mask.clone(); }

            // Threshold the mask to binary (object: 255, background: 0)
            cv::Mat mask_binary;
            cv::threshold(mask_gray, mask_binary, 127, 255, cv::THRESH_BINARY);

            // Create a colored version of the mask
            cv::Mat colored_mask;
            cv::cvtColor(mask_binary, colored_mask, cv::COLOR_GRAY2BGR);
            colored_mask.setTo(color, mask_binary); // Apply color where mask is present

            // Blend the colored mask with the original image
            cv::addWeighted(image, 1.0, colored_mask, maskAlpha, 0, image);
        }
    }
}

inline std::vector<Segmentation> YOLO11Segment::segment(const cv::Mat& image,
                                                        float confThreshold,
                                                        float iouThreshold)
{
    ScopedTimer timer("SegmentTask");

    float* blobPtr = nullptr;
    std::vector<int64_t> inputShape = {1, 3, inputImageShape.height, inputImageShape.width};
    cv::Mat letterboxImg = preprocess(image, blobPtr, inputShape);

    size_t inputSize = YOLOUtils::vectorProduct(inputShape);
    std::vector inputVals(blobPtr, blobPtr + inputSize);
    delete[] blobPtr;

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        inputVals.data(),
        inputSize,
        inputShape.data(),
        inputShape.size()
    );

    std::vector<Ort::Value> outputs = session.Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        &inputTensor,
        numInputNodes,
        outputNames.data(),
        numOutputNodes);

    cv::Size letterboxSize(static_cast<int>(inputShape[3]), static_cast<int>(inputShape[2]));
    return postprocess(image.size(), letterboxSize, outputs, confThreshold, iouThreshold);
}
