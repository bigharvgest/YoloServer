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
    std::string defaultResult;
    std::string successResult;
    std::string failResult;
    std::unordered_map<std::string, std::shared_ptr<YOLO11Model>> modelMap;

    // 屏幕检测线程相关成员
    std::mutex screenDetectMutex;
    std::string latestScreenDetectResult;
    std::atomic<bool> screenDetectRunning{false};
    std::thread screenDetectThread;

public:
    explicit YOLOServer()
    {
        auto monitors = mss.get_monitors();
        nlohmann::json null_result;
        null_result["results"] = nlohmann::json::array();
        null_result["count"] = 0;
        null_result["success"] = false;
        defaultResult = null_result.dump();
        latestScreenDetectResult = null_result.dump();

        nlohmann::json success_result;
        success_result["success"] = true;
        successResult = success_result.dump();

        nlohmann::json fail_result;
        fail_result["success"] = false;
        failResult = fail_result.dump();
    }

    // 启动屏幕采集和检测线程
    void start_screen_detect_thread(const std::string& modelId,
                                    float confThreshold = 0.4f,
                                    float iouThreshold = 0.45f)
    {
        if (screenDetectRunning.load())
        {
            return; // 已经运行
        }

        screenDetectRunning.store(true);
        screenDetectThread = std::thread([this, modelId, confThreshold, iouThreshold]
        {
            while (screenDetectRunning.load())
            {
                cv::Mat frame = wait_and_get_frame();
                if (frame.empty())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }

                std::string detectResult = detect(modelId, frame, confThreshold, iouThreshold);

                {
                    std::lock_guard lock(screenDetectMutex);
                    latestScreenDetectResult = detectResult;
                }
            }

            {
                std::lock_guard lock(screenDetectMutex);
                latestScreenDetectResult = defaultResult;
            }
        });
        screenDetectThread.detach();
    }

    // 停止线程
    void stop_screen_detect_thread()
    {
        if (screenDetectRunning.load())
        {
            screenDetectRunning.store(false);
        }
    }

    // 获取当前屏幕检测结果
    std::string get_latest_screen_detect_result()
    {
        std::lock_guard lock(screenDetectMutex);
        return latestScreenDetectResult;
    }

    std::string load_model(const std::string& id, const std::string& task, const std::vector<char>& modelBuffer,
                           const bool isGpu)
    {
        if (task == "detect")
        {
            modelMap[id] = std::make_shared<YOLO11Detect>(modelBuffer, isGpu);
            return successResult;
        }
        if (task == "obb")
        {
            modelMap[id] = std::make_shared<YOLO11OBB>(modelBuffer, isGpu);
            return successResult;
        }
        if (task == "segment")
        {
            modelMap[id] = std::make_shared<YOLO11Segment>(modelBuffer, isGpu);
            return successResult;
        }
        if (task == "pose")
        {
            modelMap[id] = std::make_shared<YOLO11Pose>(modelBuffer, isGpu);
            return successResult;
        }
        if (task == "classify")
        {
            modelMap[id] = std::make_shared<YOLO11Classify>(modelBuffer, isGpu);
            return successResult;
        }
        return failResult;
    }

    std::string unload_model(const std::string& id)
    {
        if (const auto it = modelMap.find(id); it != modelMap.end()) { modelMap.erase(it); }
        return successResult;
    }

    std::string start_capture(const std::string& id, const float confThreshold, const float iouThreshold)
    {
        start_screen_detect_thread(id, confThreshold, iouThreshold);
        mss.start_capture();
        return successResult;
    }

    std::string stop_capture()
    {
        stop_screen_detect_thread();
        mss.stop_capture();
        return successResult;
    }

    cv::Mat wait_and_get_frame() { return mss.wait_and_get_frame(); }

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
        return defaultResult;
    }
};

// 读取4字节
uint32_t ReadUint32FromPipe(const HANDLE pipe)
{
    uint8_t buf[4];
    DWORD bytesRead = 0;
    if (!ReadFile(pipe, buf, 4, &bytesRead, nullptr) || bytesRead != 4)
    {
        return 0;
    }
    return static_cast<uint32_t>(buf[3]) << 24 | static_cast<uint32_t>(buf[2]) << 16 | static_cast<uint32_t>(buf[1]) <<
        8 | static_cast<uint32_t>(buf[0]);
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
            throw std::runtime_error("读取数据失败");
        }
        if (bytesRead == 0) { throw std::runtime_error("管道读取结束"); }
        totalRead += bytesRead;
    }
    return buffer;
}

void ProcessClient(HANDLE pipe, YOLOServer& server)
{
    try
    {
        while (true)
        {
            uint32_t jsonLength = ReadUint32FromPipe(pipe);
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

            std::string result = R"({"success": false})";
            if (json.contains("action"))
            {
                std::string action = json["action"];
                if (action == "load_model")
                {
                    std::string id = json.value("id", "default");
                    std::string modelPath = json.value("modelPath", "");
                    std::string keyPath = json.value("keyPath", "");
                    std::string task = json.value("task", "detect");
                    bool isGpu = json.value("isGpu", false);

                    if (std::wstring w_model_path = YOLOUtils::utf8_to_wstring(modelPath); !std::filesystem::exists(
                        w_model_path))
                    {
                        std::cerr << u8"模型路径不存在: " << modelPath << std::endl;
                    }
                    else
                    {
                        // 读取密钥（如果有）
                        std::vector<uchar> keyBuffer;
                        if (!keyPath.empty())
                        {
                            std::wstring w_key_path = YOLOUtils::utf8_to_wstring(keyPath);
                            if (!std::filesystem::exists(w_key_path))
                            {
                                std::cerr << u8"密钥路径不存在: " << keyPath << std::endl;
                            }
                            else
                            {
                                if (!YOLOUtils::ReadKeyFile(w_key_path, keyBuffer))
                                {
                                    std::cerr << u8"读取密钥文件失败" << std::endl;
                                }
                            }
                        }

                        std::vector<char> modelBuffer;
                        if (YOLOUtils::ReadModelFile(w_model_path, modelBuffer))
                        {
                            if (!keyBuffer.empty())
                            {
                                modelBuffer = YOLOUtils::Decrypt(modelBuffer, keyBuffer);
                            }

                            std::cout << u8"加载模型: id=" << id << ", path=" << modelPath << ", task=" << task << ", gpu="
                                <<
                                isGpu << std::endl;
                            // 加载模型逻辑
                            result = server.load_model(id, task, modelBuffer, isGpu);
                        }
                    }
                }
                else if (action == "unload_model")
                {
                    std::string id = json.value("id", "default");
                    std::cout << u8"卸载模型: id=" << id << std::endl;
                    // 卸载模型逻辑
                    result = server.unload_model(id);
                }
                else if (action == "start_capture")
                {
                    std::string id = json.value("id", "default");
                    float confThreshold = json.value("confThreshold", 0.4f);
                    float iouThreshold = json.value("iouThreshold", 0.45f);
                    std::cout << u8"开始捕获屏幕" << std::endl;
                    // 开始捕获屏幕逻辑
                    result = server.start_capture(id, confThreshold, iouThreshold);
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
                    result = server.get_latest_screen_detect_result();
                }
            }

            if (!result.empty())
            {
                // 向客户端发送数据
                DWORD bytesWritten = 0;
                DWORD len = static_cast<DWORD>(result.size()); // 推荐用 size()

                BOOL writeResult = WriteFile(pipe, result.c_str(), len, &bytesWritten, nullptr);
                if (!writeResult)
                {
                    std::cerr << u8"写入命名管道失败, 异常: " << GetLastError() << std::endl;
                }
                else if (bytesWritten != len)
                {
                    std::cerr << u8"写入命名管道失败, 写入字节数: " << bytesWritten << ", 期望字节数: " << len << std::endl;
                }
                else
                {
                    DEBUG_PRINT(u8"成功写入数据: " << result)
                }
                FlushFileBuffers(pipe);
            }
        }
    }
    catch (const std::exception& ex) { std::cerr << u8"处理客户端异常: " << ex.what() << std::endl; }

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
        std::cerr << "CreateMutex failed, error: " << GetLastError() << std::endl;
        return 1;
    }
    // 如果已经有实例在运行，GetLastError == ERROR_ALREADY_EXISTS
    if (GetLastError() == ERROR_ALREADY_EXISTS)
    {
        std::cout << "程序已经在运行，退出..." << std::endl;
        CloseHandle(mutex);
        return 0;
    }

    YOLOServer server;

    while (true)
    {
        const auto pipeName = L"\\\\.\\pipe\\YoloServerPipe";
        // 创建管道实例，注意这里设置最大实例数为 PIPE_UNLIMITED_INSTANCES，支持多客户端
        HANDLE hPipe = CreateNamedPipeW(
            pipeName,
            PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES,
            4096, 4096,
            0,
            nullptr);

        if (hPipe == INVALID_HANDLE_VALUE)
        {
            std::cerr << u8"创建命名管道失败: " << GetLastError() << std::endl;
            break;
        }

        BOOL connected = ConnectNamedPipe(hPipe, nullptr) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
        if (connected)
        {
            std::cout << u8"有客户端连接进来，创建线程处理" << std::endl;
            std::thread clientThread(ProcessClient, hPipe, std::ref(server));
            clientThread.detach();
        }
        else
        {
            std::cerr << u8"等待客户端连接失败: " << GetLastError() << std::endl;
            CloseHandle(hPipe);
        }
    }

    CloseHandle(mutex);
    return 0;
}
