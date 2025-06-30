//
// Created by BySlin on 2025/6/10.
//

#include <unordered_map>
#include <queue>

#include "cryptopp/cryptlib.h"
#include "mss/mss.hpp"
#include "yolo/YOLO11-Detect.hpp"
#include "yolo/YOLO11-OBB.hpp"
#include "yolo/YOLO11-Segment.hpp"
#include "yolo/YOLO11-Pose.hpp"
#include "yolo/YOLO11-Classify.hpp"

class YOLOServer
{
private:
    MSS mss;
    std::string successResult;
    std::string failResult;
    std::unordered_map<std::string, std::shared_ptr<YOLO11Model>> modelMap;

public:
    explicit YOLOServer()
    {
        nlohmann::json fail_result_;
        fail_result_["results"] = nlohmann::json::array();
        fail_result_["count"] = 0;
        fail_result_["success"] = false;
        failResult = fail_result_.dump();

        nlohmann::json success_result_;
        success_result_["success"] = true;
        success_result_["results"] = nlohmann::json::array();
        success_result_["count"] = 0;
        successResult = success_result_.dump();
    }

    std::string getFailResult()
    {
        return failResult;
    }

    // 加载模型
    std::string load_model(const std::string& id, const std::string& task, const std::vector<char>& modelBuffer,
                           const bool isGpu, const int device)
    {
        if (task == "detect")
        {
            modelMap[id] = std::make_shared<YOLO11Detect>(modelBuffer, isGpu, device);
            return successResult;
        }
        if (task == "obb")
        {
            modelMap[id] = std::make_shared<YOLO11OBB>(modelBuffer, isGpu, device);
            return successResult;
        }
        if (task == "segment")
        {
            modelMap[id] = std::make_shared<YOLO11Segment>(modelBuffer, isGpu, device);
            return successResult;
        }
        if (task == "pose")
        {
            modelMap[id] = std::make_shared<YOLO11Pose>(modelBuffer, isGpu, device);
            return successResult;
        }
        if (task == "classify")
        {
            modelMap[id] = std::make_shared<YOLO11Classify>(modelBuffer, isGpu, device);
            return successResult;
        }
        return failResult;
    }

    // 卸载模型
    std::string unload_model(const std::string& id)
    {
        if (const auto it = modelMap.find(id); it != modelMap.end()) { modelMap.erase(it); }
        return successResult;
    }

    // 启动屏幕截图
    std::string start_capture(const int monitor_index, const int x,
                              const int y, const int width, const int height)
    {
        mss.start_capture(monitor_index, x, y, width, height);
        return successResult;
    }

    // 停止屏幕截图
    std::string stop_capture()
    {
        mss.stop_capture();
        return successResult;
    }

    // 获取屏幕截图
    cv::Mat wait_and_get_frame() { return mss.wait_and_get_frame(); }

    // 检测图像
    std::string detect(const std::string& id, const cv::Mat& image, const float confThreshold, const float iouThreshold)
    {
        if (auto it = modelMap.find(id); it != modelMap.end())
        {
            auto model = it->second;
            if (auto task = model->getTask(); task == "detect")
            {
                auto yolo11Detect = std::dynamic_pointer_cast<YOLO11Detect>(model);
                auto results = yolo11Detect->detect(image, confThreshold, iouThreshold);
                nlohmann::json j;
                j["results"] = results;
                j["count"] = results.size();
                j["success"] = true;
                return j.dump();
            }
            else if (task == "obb")
            {
                auto yolo11OBB = std::dynamic_pointer_cast<YOLO11OBB>(model);
                auto results = yolo11OBB->detect(image, confThreshold, iouThreshold);
                nlohmann::json j;
                j["results"] = results;
                j["count"] = results.size();
                j["success"] = true;
                return j.dump();
            }
            else if (task == "segment")
            {
                auto yolo11Segment = std::dynamic_pointer_cast<YOLO11Segment>(model);
                auto results = yolo11Segment->segment(image, confThreshold, iouThreshold);
                nlohmann::json j;
                j["results"] = results;
                j["count"] = results.size();
                j["success"] = true;
                return j.dump();
            }
            else if (task == "pose")
            {
                auto yolo11Pose = std::dynamic_pointer_cast<YOLO11Pose>(model);
                auto results = yolo11Pose->detect(image, confThreshold, iouThreshold);
                nlohmann::json j;
                j["results"] = results;
                j["count"] = results.size();
                j["success"] = true;
                return j.dump();
            }
            else if (task == "classify")
            {
                auto yolo11Class = std::dynamic_pointer_cast<YOLO11Classify>(model);
                auto results = yolo11Class->classify(image);
                nlohmann::json j;
                j["results"] = std::vector{results};
                j["count"] = 1;
                j["success"] = true;
                return j.dump();
            }
        }
        return failResult;
    }

    // 检查屏幕
    std::string detect_screen(const std::string& id, const float confThreshold, const float iouThreshold)
    {
        const auto mat = mss.wait_and_get_frame();
        if (mat.empty())
        {
            return failResult;
        }
        return detect(id, mat, confThreshold, iouThreshold);
    }
};

// 封装带超时的读取4字节函数
uint32_t ReadUint32FromPipeWithTimeout(const HANDLE pipe, const DWORD timeoutMs)
{
    uint8_t buf[4] = {};
    DWORD bytesRead = 0;
    OVERLAPPED overlapped = {};
    overlapped.hEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    if (!overlapped.hEvent)
    {
        throw std::runtime_error(u8"创建事件失败");
    }

    const BOOL readResult = ReadFile(pipe, buf, 4, nullptr, &overlapped);
    if (!readResult)
    {
        const DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING)
        {
            CloseHandle(overlapped.hEvent);
            if (err == ERROR_BROKEN_PIPE)
            {
                throw std::runtime_error(u8"客户端已断开连接");
            }
            throw std::runtime_error(u8"ReadFile失败，错误码：" + std::to_string(err));
        }
    }

    // 等待读取完成或超时
    const DWORD waitRes = WaitForSingleObject(overlapped.hEvent, timeoutMs);
    if (waitRes == WAIT_TIMEOUT)
    {
        CancelIo(pipe); // 取消IO请求
        CloseHandle(overlapped.hEvent);
        // 超时返回0
        return 0;
    }
    else if (waitRes != WAIT_OBJECT_0)
    {
        CloseHandle(overlapped.hEvent);
        throw std::runtime_error(u8"等待事件失败");
    }

    // 获取实际读取字节数
    if (!GetOverlappedResult(pipe, &overlapped, &bytesRead, FALSE))
    {
        const DWORD err = GetLastError();
        CloseHandle(overlapped.hEvent);
        if (err == ERROR_BROKEN_PIPE)
            throw std::runtime_error(u8"客户端已断开连接");
        throw std::runtime_error(u8"GetOverlappedResult失败，错误码：" + std::to_string(err));
    }

    CloseHandle(overlapped.hEvent);

    if (bytesRead != 4)
    {
        throw std::runtime_error(u8"读取字节不足4字节，可能管道断开");
    }

    return static_cast<uint32_t>(buf[3]) << 24 |
        static_cast<uint32_t>(buf[2]) << 16 |
        static_cast<uint32_t>(buf[1]) << 8 |
        static_cast<uint32_t>(buf[0]);
}

// 读取指定长度数据到缓冲区
std::vector<char> ReadDataFromPipe(const HANDLE pipe, const size_t length)
{
    std::vector<char> buffer(length);
    size_t totalRead = 0;
    DWORD bytesRead = 0;
    while (totalRead < length)
    {
        if (!ReadFile(pipe, buffer.data() + totalRead, static_cast<DWORD>(length - totalRead), &bytesRead, nullptr))
        {
            throw std::runtime_error(u8"读取数据失败");
        }
        if (bytesRead == 0) { throw std::runtime_error(u8"管道读取结束"); }
        totalRead += bytesRead;
    }
    return buffer;
}

// 返回值：true表示写入成功且写入字节数与len一致，false表示失败或写入不完整
bool WriteToPipe(const HANDLE pipe, const char* data, const DWORD len)
{
    DWORD totalWritten = 0;
    while (totalWritten < len)
    {
        DWORD bytesWritten = 0;
        if (const BOOL result = WriteFile(pipe, data + totalWritten, len - totalWritten, &bytesWritten, nullptr); !
            result)
        {
            const DWORD err = GetLastError();
            std::cerr << u8"向管道写入失败，错误码：" << err << std::endl;
            return false;
        }
        if (bytesWritten == 0)
        {
            std::cerr << u8"向管道写入失败，写入字节数为0" << std::endl;
            return false;
        }
        totalWritten += bytesWritten;
    }

    if (!FlushFileBuffers(pipe))
    {
        std::cerr << u8"刷新管道缓冲区失败，错误码：" << GetLastError() << std::endl;
        // 这里一般不影响写入结果，不返回false
    }

    return true;
}

// C++ string版重载，方便直接传std::string
bool WriteToPipe(HANDLE pipe, const std::string& data)
{
    return WriteToPipe(pipe, data.data(), static_cast<DWORD>(data.size()));
}

// 处理客户端请求
void ProcessClient(HANDLE pipe, YOLOServer& server)
{
    while (true)
    {
        try
        {
            uint32_t jsonLength = ReadUint32FromPipeWithTimeout(pipe, 2000);
            if (jsonLength == 0)
            {
                continue;
            }

            DEBUG_PRINT(u8"收到json长度: " << jsonLength)

            std::vector<char> jsonBuf = ReadDataFromPipe(pipe, jsonLength);
            std::string jsonStr(jsonBuf.begin(), jsonBuf.end());
            auto json = nlohmann::json::parse(jsonStr);

            size_t binaryLength = 0;
            if (json.contains("binaryLength")) { binaryLength = json["binaryLength"].get<size_t>(); }

            std::vector<char> binaryData;
            if (binaryLength > 0)
            {
                binaryData = ReadDataFromPipe(pipe, binaryLength);
                std::cout << u8"读取到二进制数据长度: " << binaryLength << std::endl;
            }

            std::string result = server.getFailResult();
            if (json.contains("action"))
            {
                std::string action = json["action"];
                // 加载模型
                if (action == "load_model")
                {
                    std::string id = json.value("id", "default");
                    std::string modelPath = json.value("modelPath", "");
                    std::string password = json.value("password", "");
                    std::string task = json.value("task", "detect");
                    bool isGpu = json.value("isGpu", false);
                    int device = json.value("device", 0);

                    if (std::wstring w_model_path = YOLOUtils::utf8_to_wstring(modelPath); !std::filesystem::exists(
                        w_model_path))
                    {
                        std::cerr << u8"模型路径不存在: " << modelPath << std::endl;
                    }
                    else
                    {
                        std::vector<char> modelBuffer;
                        if (YOLOUtils::ReadModelFile(w_model_path, modelBuffer))
                        {
                            if (!password.empty())
                            {
                                modelBuffer = YOLOUtils::Decrypt(modelBuffer, password);
                            }

                            std::cout << u8"加载模型: id=" << id << ", path=" << modelPath << ", task=" << task << ", gpu="
                                <<
                                isGpu << std::endl;
                            // 加载模型逻辑
                            result = server.load_model(id, task, modelBuffer, isGpu, device);
                        }
                    }
                }
                // 卸载模型
                else if (action == "unload_model")
                {
                    std::string id = json.value("id", "default");
                    std::cout << u8"卸载模型: id=" << id << std::endl;
                    // 卸载模型逻辑
                    result = server.unload_model(id);
                }
                // 开始捕获屏幕
                else if (action == "start_capture")
                {
                    int monitor_index = json.value("monitor_index", 0);
                    int x = json.value("x", 0);
                    int y = json.value("y", 0);
                    int width = json.value("width", 0);
                    int height = json.value("height", 0);
                    std::cout << u8"开始捕获屏幕" << std::endl;
                    // 开始捕获屏幕逻辑
                    result = server.start_capture(monitor_index, x, y, width, height);
                }
                else if (action == "stop_capture")
                {
                    std::cout << u8"停止捕获屏幕" << std::endl;
                    // 停止捕获屏幕逻辑
                    result = server.stop_capture();
                }
                else if (action == "detect_img")
                {
                    std::string id = json.value("id", "default");
                    float confThreshold = json.value("confThreshold", 0.4f);
                    float iouThreshold = json.value("iouThreshold", 0.45f);

                    if (binaryData.empty()) { std::cerr << u8"没有收到图像二进制数据" << std::endl; }
                    else
                    {
                        if (cv::Mat img = cv::imdecode(binaryData, cv::IMREAD_COLOR); img.empty())
                        {
                            std::cerr << u8"图像解码失败" << std::endl;
                        }
                        else
                        {
                            std::cout << u8"进行检测 id=" << id << std::endl;
                            // 检测逻辑
                            result = server.detect(id, img, confThreshold, iouThreshold);
                        }
                    }
                }
                else if (action == "detect_screen")
                {
                    std::string id = json.value("id", "default");
                    float confThreshold = json.value("confThreshold", 0.4f);
                    float iouThreshold = json.value("iouThreshold", 0.45f);
                    result = server.detect_screen(id, confThreshold, iouThreshold);
                }
            }

            if (!result.empty())
            {
                if (!WriteToPipe(pipe, result))
                {
                    std::cerr << u8"写入命名管道失败" << std::endl;
                }
                else
                {
                    DEBUG_PRINT(u8"成功写入数据: " << result)
                }
            }
        }
        catch (const std::exception& ex)
        {
            if (DWORD lastError = GetLastError(); lastError == ERROR_BROKEN_PIPE)
            {
                // 客户端断开，退出循环关闭管道
                std::cerr << u8"客户端已断开连接" << std::endl;
                break;
            }

            std::cerr << u8"处理客户端异常: " << ex.what() << std::endl;
            try
            {
                // 将异常信息封装成 JSON 结构字符串发回客户端
                nlohmann::json errorJson = {
                    {"success", false},
                    {"results", nlohmann::json::array()},
                    {"count", 0},
                    {"message", ex.what()}
                };
                std::string errorStr = errorJson.dump();
                if (!WriteToPipe(pipe, errorStr))
                {
                    std::cerr << u8"异常处理时写入管道失败" << std::endl;
                }
                else
                {
                    DEBUG_PRINT(u8"异常信息已发送给客户端")
                }
            }
            catch (const std::exception& innerEx)
            {
                std::cerr << u8"处理异常时再次发生异常: " << innerEx.what() << std::endl;
            }
        }
    }


    DisconnectNamedPipe(pipe);
    CloseHandle(pipe);
    std::cout << u8"客户端线程结束" << std::endl;
}

int main()
{
    // 设置控制台宽字符模式
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    // 创建一个命名互斥体，名字要全局唯一
    HANDLE mutex = CreateMutex(nullptr, TRUE, L"YoloServerMutex");
    if (mutex == nullptr)
    {
        std::cerr << u8"CreateMutex failed, error: " << GetLastError() << std::endl;
        return 1;
    }
    // 如果已经有实例在运行，GetLastError == ERROR_ALREADY_EXISTS
    if (GetLastError() == ERROR_ALREADY_EXISTS)
    {
        std::cout << u8"程序已经在运行，退出..." << std::endl;
        CloseHandle(mutex);
        return 0;
    }

    YOLOServer server;

    const auto pipeName = L"\\\\.\\pipe\\YoloServerPipe";
    const HANDLE hPipe = CreateNamedPipeW(
        pipeName,
        PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
        1, // 限制客户端数量为1
        4096, 4096,
        0,
        nullptr);

    if (hPipe == INVALID_HANDLE_VALUE)
    {
        std::cerr << u8"创建命名管道失败: " << GetLastError() << std::endl;
        return 1;
    }

    std::cout << u8"等待客户端连接..." << std::endl;
    if (const BOOL connected = ConnectNamedPipe(hPipe, nullptr) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED); !
        connected)
    {
        std::cerr << u8"连接客户端失败: " << GetLastError() << std::endl;
        CloseHandle(hPipe);
        CloseHandle(mutex);
        return 1;
    }

    std::cout << u8"客户端已连接，开始处理" << std::endl;
    ProcessClient(hPipe, server);
    std::cout << u8"客户端断开，程序退出" << std::endl;

    CloseHandle(mutex);
    return 0;
}
