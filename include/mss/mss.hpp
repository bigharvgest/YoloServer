#pragma once

#define NOMINMAX

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d11.lib")

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "tools/ScopedTimer.hpp"

using Microsoft::WRL::ComPtr;

struct Monitor
{
    int left;
    int top;
    int width;
    int height;
};

inline void to_json(nlohmann::json& j, const Monitor& monitor)
{
    j = nlohmann::json{
        {"left", monitor.left},
        {"top", monitor.top},
        {"width", monitor.width},
        {"height", monitor.height}
    };
}

class MSS
{
private:
    std::vector<Monitor> _monitors;

    // D3D11 设备相关
    ComPtr<ID3D11Device> _d3dDevice;
    ComPtr<ID3D11DeviceContext> _d3dContext;

    // DXGI Factory, Adapter, Outputs
    ComPtr<IDXGIFactory1> _dxgiFactory;
    ComPtr<IDXGIAdapter> _dxgiAdapter;

    struct OutputDuplication
    {
        ComPtr<IDXGIOutput> output;
        ComPtr<IDXGIOutput1> output1;
        ComPtr<IDXGIOutputDuplication> duplication;
        Monitor monitor;

        ComPtr<ID3D11Texture2D> stagingTexture;
    };

    std::vector<OutputDuplication> _outputDuplications;

    // 持续截图相关
    std::thread _captureThread;
    std::atomic<bool> _running{false};
    std::mutex _frameMutex;
    std::condition_variable _frameCondVar;

    cv::Mat _latestFrame;
    Monitor _captureMonitor;

    int captureX{0};
    int captureY{0};
    int captureWidth{0};
    int captureHeight{0};
    bool captureFullScreen{true};

    void captureLoop()
    {
        constexpr std::chrono::nanoseconds frame_duration_ns(16666667); // 约16.66667ms
        auto next_frame_time = std::chrono::steady_clock::now();

        while (_running.load())
        {
            cv::Mat frame = captureFullScreen
                                ? grab(_captureMonitor)
                                : grab_region(_captureMonitor, captureX, captureY, captureWidth, captureHeight);
            if (!frame.empty())
            {
                {
                    std::lock_guard lock(_frameMutex);
                    _latestFrame = std::move(frame);
                }
                _frameCondVar.notify_one();
            }

            next_frame_time += frame_duration_ns;
            std::this_thread::sleep_until(next_frame_time);

            // 防止时间漂移，重置时间点
            if (std::chrono::steady_clock::now() > next_frame_time + frame_duration_ns)
            {
                next_frame_time = std::chrono::steady_clock::now();
            }
        }
    }

public:
    MSS()
    {
        init_d3d();
        init_monitors_and_duplication();
    }

    ~MSS()
    {
        stop_capture();
        release_resources();
    }

    const std::vector<Monitor>& get_monitors() const { return _monitors; }

    // 启动持续截图线程，默认抓第0个显示器
    void start_capture(const int monitor_index = 0,
                       const int x = 0,
                       const int y = 0,
                       const int width = 0,
                       const int height = 0)
    {
        if (_running.load())
        {
            std::cerr << "[MSS] Capture already running\n";
            return;
        }
        if (monitor_index < 0 || monitor_index >= static_cast<int>(_monitors.size()))
        {
            std::cerr << "[MSS] Invalid monitor index\n";
            return;
        }
        captureX = x;
        captureY = y;
        captureWidth = width;
        captureHeight = height;

        // 判断是否截取全屏
        if (captureX == 0 && captureY == 0 && captureWidth == 0 && captureHeight == 0)
        {
            captureFullScreen = true;
        }
        else
        {
            captureFullScreen = false;
        }

        _captureMonitor = _monitors[monitor_index];
        _running.store(true);
        _captureThread = std::thread(&MSS::captureLoop, this);
    }

    // 停止截图线程
    void stop_capture()
    {
        if (!_running.load()) return;
        _running.store(false);
        if (_captureThread.joinable()) { _captureThread.join(); }
    }

    // 等待并获取最新截图帧，timeout_ms <=0 表示无限等待
    cv::Mat wait_and_get_frame(int timeout_ms = 100)
    {
        ScopedTimer timer("wait_and_get_frame");
        std::unique_lock lock(_frameMutex);
        if (timeout_ms <= 0) { _frameCondVar.wait(lock, [this] { return !_latestFrame.empty(); }); }
        else
        {
            _frameCondVar.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                   [this] { return !_latestFrame.empty(); });
        }
        return _latestFrame;
    }

    // 一次性抓取指定显示器全屏截图（无后台线程）
    cv::Mat grab(const Monitor& monitor)
    {
        // ScopedTimer timer("grabTask");
        const auto it = std::find_if(_outputDuplications.begin(), _outputDuplications.end(),
                                     [&monitor](const OutputDuplication& od)
                                     {
                                         return od.monitor.left == monitor.left &&
                                             od.monitor.top == monitor.top &&
                                             od.monitor.width == monitor.width &&
                                             od.monitor.height == monitor.height;
                                     });
        if (it == _outputDuplications.end())
        {
            std::cerr << "[MSS] grab: monitor duplication not found." << std::endl;
            return cv::Mat();
        }

        const cv::Mat fullImage = capture_output(*it);
        if (fullImage.empty())
        {
            std::cerr << "[MSS] grab: capture output failed." << std::endl;
            return cv::Mat();
        }

        cv::Mat bgrImage;
        cv::cvtColor(fullImage, bgrImage, cv::COLOR_BGRA2BGR);
        return bgrImage;
    }

    // 抓取指定显示器指定区域截图
    cv::Mat grab_region(const Monitor& monitor, const int x, const int y, const int width, const int height)
    {
        if (x < 0 || y < 0 || x + width > monitor.width || y + height > monitor.height)
        {
            std::cerr << "[MSS] grab_region: invalid region parameters." << std::endl;
            return cv::Mat();
        }

        const cv::Mat full = grab(monitor);
        if (full.empty())
        {
            std::cerr << "[MSS] grab_region: grab full screen failed." << std::endl;
            return cv::Mat();
        }

        const cv::Rect roi(x, y, width, height);
        return full(roi).clone();
    }

protected:
    void init_d3d()
    {
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

        D3D_FEATURE_LEVEL featureLevel;
        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            flags,
            nullptr, 0,
            D3D11_SDK_VERSION,
            &_d3dDevice,
            &featureLevel,
            &_d3dContext);

        if (FAILED(hr))
        {
            std::cerr << "[MSS] Failed to create D3D11 device. HRESULT=0x" << std::hex << hr << std::endl;
            return;
        }

        hr = CreateDXGIFactory1(IID_PPV_ARGS(&_dxgiFactory));
        if (FAILED(hr))
        {
            std::cerr << "[MSS] Failed to create DXGI factory. HRESULT=0x" << std::hex << hr << std::endl;
            return;
        }

        hr = _dxgiFactory->EnumAdapters(0, &_dxgiAdapter);
        if (FAILED(hr))
        {
            std::cerr << "[MSS] Failed to enumerate adapters. HRESULT=0x" << std::hex << hr << std::endl;
            return;
        }
    }

    void init_monitors_and_duplication()
    {
        _monitors.clear();
        _outputDuplications.clear();

        UINT i = 0;
        while (true)
        {
            ComPtr<IDXGIOutput> output;
            HRESULT hr = _dxgiAdapter->EnumOutputs(i++, &output);
            if (hr == DXGI_ERROR_NOT_FOUND) break;
            if (FAILED(hr))
            {
                std::cerr << "[MSS] Failed to enumerate outputs. HRESULT=0x" << std::hex << hr << std::endl;
                continue;
            }

            DXGI_OUTPUT_DESC desc;
            output->GetDesc(&desc);

            Monitor mon{};
            mon.left = desc.DesktopCoordinates.left;
            mon.top = desc.DesktopCoordinates.top;
            mon.width = desc.DesktopCoordinates.right - desc.DesktopCoordinates.left;
            mon.height = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;
            _monitors.push_back(mon);

            ComPtr<IDXGIOutput1> output1;
            hr = output.As(&output1);
            if (FAILED(hr))
            {
                std::cerr << "[MSS] Failed to get IDXGIOutput1. HRESULT=0x" << std::hex << hr << std::endl;
                continue;
            }

            ComPtr<IDXGIOutputDuplication> duplication;
            hr = output1->DuplicateOutput(_d3dDevice.Get(), &duplication);
            if (FAILED(hr))
            {
                std::cerr << "[MSS] Failed to duplicate output. HRESULT=0x" << std::hex << hr << std::endl;
                continue;
            }

            _outputDuplications.push_back({std::move(output), std::move(output1), std::move(duplication), mon});
        }
    }

    void release_resources()
    {
        for (auto& dup : _outputDuplications) { if (dup.duplication) { dup.duplication->ReleaseFrame(); } }
        _outputDuplications.clear();
        _dxgiAdapter.Reset();
        _dxgiFactory.Reset();
        _d3dContext.Reset();
        _d3dDevice.Reset();
    }

    bool ensure_staging_texture(OutputDuplication& od) const
    {
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = od.monitor.width;
        desc.Height = od.monitor.height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

        if (od.stagingTexture)
        {
            D3D11_TEXTURE2D_DESC existingDesc;
            od.stagingTexture->GetDesc(&existingDesc);
            if (existingDesc.Width == desc.Width &&
                existingDesc.Height == desc.Height &&
                existingDesc.Format == desc.Format) { return true; }
        }

        HRESULT hr = _d3dDevice->CreateTexture2D(&desc, nullptr, &od.stagingTexture);
        if (FAILED(hr))
        {
            std::cerr << "[MSS] Failed to create staging texture. HRESULT=0x" << std::hex << hr << std::endl;
            od.stagingTexture.Reset();
            return false;
        }
        return true;
    }

    cv::Mat capture_output(OutputDuplication& od) const
    {
        if (!ensure_staging_texture(od)) { return cv::Mat(); }

        constexpr int maxRetries = 1000;
        int retryCount = 0;

        while (retryCount < maxRetries)
        {
            DXGI_OUTDUPL_FRAME_INFO frameInfo;
            ComPtr<IDXGIResource> desktopResource;

            HRESULT hr = od.duplication->AcquireNextFrame(30, &frameInfo, &desktopResource);
            if (hr == DXGI_ERROR_WAIT_TIMEOUT)
            {
                Sleep(1);
                retryCount++;
                continue;
            }
            if (FAILED(hr))
            {
                std::cerr << "[MSS] AcquireNextFrame failed: 0x" << std::hex << hr << std::endl;
                Sleep(1);
                retryCount++;
                continue;
            }

            retryCount = 0;

            ComPtr<ID3D11Texture2D> acquiredDesktopImage;
            hr = desktopResource.As(&acquiredDesktopImage);
            desktopResource.Reset();

            if (FAILED(hr))
            {
                od.duplication->ReleaseFrame();
                std::cerr << "[MSS] Failed to query ID3D11Texture2D from IDXGIResource." << std::endl;
                continue;
            }

            _d3dContext->CopyResource(od.stagingTexture.Get(), acquiredDesktopImage.Get());

            D3D11_MAPPED_SUBRESOURCE mapped;
            hr = _d3dContext->Map(od.stagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
            if (FAILED(hr))
            {
                od.duplication->ReleaseFrame();
                std::cerr << "[MSS] Failed to map staging texture." << std::endl;
                continue;
            }

            const cv::Mat image(od.monitor.height, od.monitor.width, CV_8UC4, mapped.pData, mapped.RowPitch);
            cv::Mat ret = image.clone();

            _d3dContext->Unmap(od.stagingTexture.Get(), 0);
            od.duplication->ReleaseFrame();

            return ret;
        }

        std::cerr << "[MSS] capture_output: exceeded max retries, returning empty image." << std::endl;
        return cv::Mat();
    }
};
