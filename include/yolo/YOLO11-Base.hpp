#pragma once

#define NOMINMAX

#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class YOLO11Model
{
public:
    virtual ~YOLO11Model() = default;
    virtual std::string getTask() const = 0; // 纯虚函数，强制子类实现
};

namespace YOLOUtils
{
    /**
     * @brief 表示图像中一个矩形框的结构体
     */
    struct RectBox
    {
        int x; /**< 左上角X坐标 */
        int y; /**< 左上角Y坐标 */
        int width; /**< 矩形框宽度 */
        int height; /**< 矩形框高度 */

        /**
         * @brief 默认构造函数，所有成员初始化为0
         */
        RectBox() : x(0), y(0), width(0), height(0)
        {
        }

        /**
         * @brief 带参数构造函数，初始化矩形框坐标和尺寸
         *
         * @param x_ 左上角X坐标
         * @param y_ 左上角Y坐标
         * @param width_ 矩形框宽度
         * @param height_ 矩形框高度
         */
        RectBox(int x_, int y_, int width_, int height_)
            : x(x_), y(y_), width(width_), height(height_)
        {
        }
    };

    // 手动实现 to_json
    inline void to_json(nlohmann::json& j, const RectBox& rect)
    {
        j = nlohmann::json{
            {"x", rect.x},
            {"y", rect.y},
            {"width", rect.width},
            {"height", rect.height}
        };
    }

    // UTF-8 转 UTF-16
    inline std::wstring utf8_to_wstring(const std::string& utf8Str)
    {
        if (utf8Str.empty()) return std::wstring();
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), (int)utf8Str.size(), NULL, 0);
        std::wstring wstrTo(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), (int)utf8Str.size(), &wstrTo[0], size_needed);
        return wstrTo;
    }

    /**
     * @brief clamp函数的健壮实现，将数值限制在指定范围[low, high]内
     *
     * @tparam T 要限制的数值类型（int、float等）
     * @param value 待限制的数值
     * @param low 下界
     * @param high 上界
     * @return const T& 限制后的数值
     *
     * @note 如果low > high，会自动交换上下界
     */
    template <typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type
    inline clamp(const T& value, const T& low, const T& high)
    {
        // 保证范围合法，如有必要交换上下界
        T validLow = low < high ? low : high;
        T validHigh = low < high ? high : low;

        // 限制value在[validLow, validHigh]之间
        if (value < validLow) return validLow;
        if (value > validHigh) return validHigh;
        return value;
    }

    /**
    * 解析模型内部的classNames
    * @param text 模型内部的classNames字符串
    * @return 类别名称字符串数组
    */
    inline std::vector<std::string> parseClassNames(const std::string& text)
    {
        auto trim = [](const std::string& s, const std::string& chars = " \t\n\r") -> std::string
        {
            size_t start = s.find_first_not_of(chars);
            if (start == std::string::npos) return "";
            size_t end = s.find_last_not_of(chars);
            return s.substr(start, end - start + 1);
        };

        auto trimChar = [](const std::string& s, char ch) -> std::string
        {
            size_t start = 0;
            while (start < s.size() && s[start] == ch) ++start;
            size_t end = s.size();
            while (end > start && s[end - 1] == ch) --end;
            return s.substr(start, end - start);
        };

        std::string s = trim(text, "{}");
        std::vector<std::string> result;

        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            item = trim(item);
            size_t pos = item.find(':');
            if (pos == std::string::npos) continue;

            std::string right = trim(item.substr(pos + 1));
            right = trimChar(right, '\'');
            result.push_back(right);
        }

        return result;
    }

    /**
      * @brief 计算向量中所有元素的乘积
      *
      * @param vector 整型向量
      * @return size_t 所有元素的乘积
      */
    inline size_t vectorProduct(const std::vector<int64_t>& vector)
    {
        return std::accumulate(vector.begin(), vector.end(), 1ull, std::multiplies<size_t>());
    }

    /**
     * @brief 对图像进行信封缩放（letterbox），保持宽高比
     *
     * @param image 输入图像
     * @param outImage 输出缩放并填充后的图像
     * @param newShape 目标输出尺寸
     * @param color 填充颜色（默认灰色）
     * @param auto_ 是否自动调整padding为stride的倍数
     * @param scaleFill 是否强制缩放填满目标尺寸（不保持比例）
     * @param scaleUp 是否允许放大图像
     * @param stride 填充对齐步长
     */
    inline void letterBox(const cv::Mat& image, cv::Mat& outImage,
                          const cv::Size& newShape,
                          const cv::Scalar& color = cv::Scalar(114, 114, 114),
                          bool auto_ = true,
                          bool scaleFill = false,
                          bool scaleUp = true,
                          int stride = 32)
    {
        // 计算缩放比例
        float ratio = std::min(static_cast<float>(newShape.height) / image.rows,
                               static_cast<float>(newShape.width) / image.cols);

        // 不允许放大时，限制最大比例为1
        if (!scaleUp) { ratio = std::min(ratio, 1.0f); }

        // 计算缩放后的新尺寸
        int newUnpadW = static_cast<int>(std::round(image.cols * ratio));
        int newUnpadH = static_cast<int>(std::round(image.rows * ratio));

        // 计算需要填充的像素
        int dw = newShape.width - newUnpadW;
        int dh = newShape.height - newUnpadH;

        if (auto_)
        {
            // 自动调整padding为stride的倍数
            dw = (dw % stride) / 2;
            dh = (dh % stride) / 2;
        }
        else if (scaleFill)
        {
            // 强制缩放填满目标尺寸（不保持比例）
            newUnpadW = newShape.width;
            newUnpadH = newShape.height;
            ratio = std::min(static_cast<float>(newShape.width) / image.cols,
                             static_cast<float>(newShape.height) / image.rows);
            dw = 0;
            dh = 0;
        }
        else
        {
            // 均匀分配padding到四周
            int padLeft = dw / 2;
            int padRight = dw - padLeft;
            int padTop = dh / 2;
            int padBottom = dh - padTop;

            // 尺寸变化时缩放图像
            if (image.cols != newUnpadW || image.rows != newUnpadH)
            {
                cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
            }
            else { outImage = image; }

            // 填充到目标尺寸
            cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
            return;
        }

        // 尺寸变化时缩放图像
        if (image.cols != newUnpadW || image.rows != newUnpadH)
        {
            cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
        }
        else { outImage = image; }

        // 均匀分配padding到四周
        int padLeft = dw / 2;
        int padRight = dw - padLeft;
        int padTop = dh / 2;
        int padBottom = dh - padTop;

        // 填充到目标尺寸
        cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
    }

    /**
     * @brief 为每个类别生成一种颜色
     *
     * @param classNames 类别名称数组
     * @param seed 随机种子，保证颜色可复现
     * @return std::vector<cv::Scalar> 颜色数组
     */
    inline std::vector<cv::Scalar> generateColors(const std::vector<std::string>& classNames, int seed = 42)
    {
        // 静态缓存，避免重复生成
        static std::unordered_map<size_t, std::vector<cv::Scalar>> colorCache;

        // 根据类别名称生成哈希key
        size_t hashKey = 0;
        for (const auto& name : classNames)
        {
            hashKey ^= std::hash<std::string>{}(name) + 0x9e3779b9 + (hashKey << 6) + (hashKey >> 2);
        }

        // 已缓存则直接返回
        auto it = colorCache.find(hashKey);
        if (it != colorCache.end()) { return it->second; }

        // 为每个类别生成随机颜色
        std::vector<cv::Scalar> colors;
        colors.reserve(classNames.size());

        std::mt19937 rng(seed); // 随机数生成器
        std::uniform_int_distribution<int> uni(0, 255); // 颜色分布

        for (size_t i = 0; i < classNames.size(); ++i)
        {
            colors.emplace_back(cv::Scalar(uni(rng), uni(rng), uni(rng))); // 随机BGR颜色
        }

        // 缓存结果
        colorCache.emplace(hashKey, colors);

        return colorCache[hashKey];
    }

    /**
     * @brief 将检测框坐标缩放回原图尺寸
     *
     * @param imageShape 推理时输入图像的尺寸
     * @param coords 检测框坐标
     * @param imageOriginalShape 原始图像尺寸
     * @param p_Clip 是否裁剪到图像边界
     * @return RectBox 缩放后的检测框
     */
    inline RectBox scaleCoords(const cv::Size& imageShape, RectBox coords,
                               const cv::Size& imageOriginalShape, bool p_Clip)
    {
        RectBox result;
        float gain = std::min(static_cast<float>(imageShape.height) / static_cast<float>(imageOriginalShape.height),
                              static_cast<float>(imageShape.width) / static_cast<float>(imageOriginalShape.width));

        int padX = static_cast<int>(std::round((imageShape.width - imageOriginalShape.width * gain) / 2.0f));
        int padY = static_cast<int>(std::round((imageShape.height - imageOriginalShape.height * gain) / 2.0f));

        result.x = static_cast<int>(std::round((coords.x - padX) / gain));
        result.y = static_cast<int>(std::round((coords.y - padY) / gain));
        result.width = static_cast<int>(std::round(coords.width / gain));
        result.height = static_cast<int>(std::round(coords.height / gain));

        if (p_Clip)
        {
            result.x = clamp(result.x, 0, imageOriginalShape.width);
            result.y = clamp(result.y, 0, imageOriginalShape.height);
            result.width = clamp(result.width, 0, imageOriginalShape.width - result.x);
            result.height = clamp(result.height, 0, imageOriginalShape.height - result.y);
        }
        return result;
    }
}
