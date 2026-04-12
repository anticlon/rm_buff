#include "buff_tracker/cudas.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda/std/limits>

namevspace buff_tracker {
    namespace cudas {
        /**
        * @brief 基于 HSV 空间的双通道掩码核函数，基于 RGB 的双目标提取
        *计算 HSV 且双路并发：核函数接收拆分好的 R、G、B 三个单通道图像，在 GPU 线程中手动计算每个像素的 H、S、V 值。
        *一次遍历，双倍产出：在算出 HSV 后，立刻同时判断该像素是否满足“扇叶（Arrow）”和“装甲板（Armor）”
        *两套不同的颜色阈值。如果满足，则在对应的 dst_arrow 或 dst_armor 图的同一位置写入 255，否则写入 0。
        *这避免了常规流程中“原图转HSV -> 阈值分割图1 -> 阈值分割图2”带来的三次显存读写瓶颈。
        */
        __global__ void rgb2mask_dual_hsv_kernel(
            const cv::cuda::PtrStepSz<uint8_t> r, 
            const cv::cuda::PtrStepSz<uint8_t> g,
            const cv::cuda::PtrStepSz<uint8_t> b,/*cv::cuda::PtrStepSz<uint8_t>
                                                   OpenCV GPU 矩阵包装类
                                                   可以直接用 ptr(y,x) 访问第 y 行第 x 列像素
                                                   核函数里必须用这个传参*/
            cv::cuda::PtrStepSz<uint8_t> dst_arrow,
            cv::cuda::PtrStepSz<uint8_t> dst_armor,
            // Arrow HSV 范围
            int h_low_a, int h_high_a, int s_low_a, int s_high_a, int v_low_a, int v_high_a,
            // Armor HSV 范围
            int h_low_m, int h_high_m, int s_low_m, int s_high_m, int v_low_m, int v_high_m)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            if (x >= r.cols || y >= r.rows) return;

            // 1) 读取 RGB,r（y,x)是高宽为y和x的数组
            uint8_t rv = r(y, x), gv = g(y, x), bv = b(y, x);

            // 2) 计算 max/min/delta
            uint8_t maxv = max(max(rv, gv), bv);
            uint8_t minv = min(min(rv, gv), bv);
            uint8_t dv   = maxv - minv;

            // 3) 计算 H（0–360）、S（0–255）、V（0–255）
            int H;
            if (dv == 0) {
                H = 0;
            } else if (maxv == rv) {
                H = ((int)(gv - bv) * 60) / dv;
                if (H < 0) H += 360;
            } else if (maxv == gv) {
                H = ((int)(bv - rv) * 60) / dv + 120;
            } else {
                H = ((int)(rv - gv) * 60) / dv + 240;
            }
            uint8_t S = (maxv == 0 ? 0 : (dv * 255 / maxv));
            uint8_t V = maxv;

            // 4) 判断 HSV 是否在各自范围内
            bool ok_a = (h_low_a <= H && H <= h_high_a)
                    && (s_low_a <= S && S <= s_high_a)
                    && (v_low_a <= V && V <= v_high_a);
            bool ok_m = (h_low_m <= H && H <= h_high_m)
                    && (s_low_m <= S && S <= s_high_m)
                    && (v_low_m <= V && V <= v_high_m);

            dst_arrow(y, x) = ok_a ? 255 : 0;
            dst_armor(y, x) = ok_m ? 255 : 0;
        }

        /**
        * @brief Host 侧封装：通过 HSV 空间提取 Arrow / Armor 掩码
        */
        void getColorFromRGB_hsv(
            const cv::cuda::GpuMat& r,   const cv::cuda::GpuMat& g,   const cv::cuda::GpuMat& b,
            cv::cuda::GpuMat& dst_arrow, cv::cuda::GpuMat& dst_armor,
            int h_low_a, int h_high_a, int s_low_a, int s_high_a, int v_low_a, int v_high_a,
            int h_low_m, int h_high_m, int s_low_m, int s_high_m, int v_low_m, int v_high_m,
            cv::cuda::Stream& stream)
        {
            CV_Assert(r.type()==CV_8UC1 && g.type()==CV_8UC1 && b.type()==CV_8UC1);
            CV_Assert(r.size()==g.size() && g.size()==b.size());
            CV_Assert(dst_arrow.type()==CV_8UC1 && dst_armor.type()==CV_8UC1);
            CV_Assert(dst_arrow.size()==dst_armor.size());
            //检查输入是不是单通道 8 位图像
            //检查所有图像尺寸一样大
            //不满足就直接报错，防止核函数崩溃

            dim3 block(32, 32);
            dim3 grid((r.cols + block.x - 1) / block.x,
                    (r.rows + block.y - 1) / block.y);
            auto cuStream = cv::cuda::StreamAccessor::getStream(stream);
            //把 ROS 的 cuda 流 → 转换成 CUDA 原生流
            rgb2mask_dual_hsv_kernel<<<grid, block, 0, cuStream>>>(
                r, g, b, dst_arrow, dst_armor,
                h_low_a, h_high_a, s_low_a, s_high_a, v_low_a, v_high_a,
                h_low_m, h_high_m, s_low_m, s_high_m, v_low_m, v_high_m
            );
            //rgb2mask_dual_hsv_kernel将rgb通道转换成hsv格式，判断arrow和armor的范围并写入掩码图中
        }

    //     /** 
    //     * @brief Bayer‑RG 单通道版 HSV 分割核
    //     */
    //     __global__ void bayer2mask_dual_hsv_kernel(
    //         const cv::cuda::PtrStepSz<uint8_t> src,
    //         cv::cuda::PtrStepSz<uint8_t> dst_arrow,
    //         cv::cuda::PtrStepSz<uint8_t> dst_armor,
    //         int h_low_a, int h_high_a, int s_low_a, int s_high_a, int v_low_a, int v_high_a,
    //         int h_low_m, int h_high_m, int s_low_m, int s_high_m, int v_low_m, int v_high_m)
    //     {
    //         int x = threadIdx.x + blockIdx.x * blockDim.x;
    //         int y = threadIdx.y + blockIdx.y * blockDim.y;
    //         if (x >= dst_arrow.cols || y >= dst_arrow.rows) return;

    //         // 1）从 2×2 Bayer‑RG 块里取出 R/G/B
    //         int i = x*2, j = y*2;
    //         uint8_t r = src(j+1, i+1);
    //         uint8_t g = (uint8_t)((src(j, i+1) + src(j+1, i)) / 2);
    //         uint8_t b = src(j, i);

    //         // 2）HSV 计算（同 RGB 版）
    //         uint8_t maxv = max(max(r,g), b);
    //         uint8_t minv = min(min(r,g), b);
    //         uint8_t dv   = maxv - minv;

    //         int H;
    //         if (dv == 0) {
    //             H = 0;
    //         } else if (maxv == r) {
    //             H = ((int)(g - b) * 60) / dv;
    //             if (H < 0) H += 360;
    //         } else if (maxv == g) {
    //             H = ((int)(b - r) * 60) / dv + 120;
    //         } else {
    //             H = ((int)(r - g) * 60) / dv + 240;
    //         }
    //         uint8_t S = (maxv==0 ? 0 : dv * 255 / maxv);
    //         uint8_t V = maxv;

    //         // 3）各自范围内就 255，否则 0
    //         bool ok_a = (h_low_a <= H && H <= h_high_a)
    //                 && (s_low_a <= S && S <= s_high_a)
    //                 && (v_low_a <= V && V <= v_high_a);

    //         bool ok_m = (h_low_m <= H && H <= h_high_m)
    //                 && (s_low_m <= S && S <= s_high_m)
    //                 && (v_low_m <= V && V <= v_high_m);

    //         dst_arrow(y, x) = ok_a ? 255 : 0;
    //         dst_armor(y, x) = ok_m ? 255 : 0;
    //     }

    //     /**
    //     * @brief Host 侧封装：Bayer‑RG 版 HSV 分割
    //     */
    //     void getColorFromBayerRG_hsv(
    //         const cv::cuda::GpuMat& src,
    //         cv::cuda::GpuMat& dst_arrow,
    //         cv::cuda::GpuMat& dst_armor,
    //         int h_low_a, int h_high_a, int s_low_a, int s_high_a, int v_low_a, int v_high_a,
    //         int h_low_m, int h_high_m, int s_low_m, int s_high_m, int v_low_m, int v_high_m,
    //         cv::cuda::Stream& stream)
    //     {
    //         CV_Assert(src.type() == CV_8UC1);
    //         CV_Assert(dst_arrow.type() == CV_8UC1 && dst_armor.type() == CV_8UC1);
    //         CV_Assert(dst_arrow.size() == dst_armor.size());

    //         dim3 block(32,32);
    //         dim3 grid((dst_arrow.cols + block.x-1)/block.x,
    //                 (dst_arrow.rows + block.y-1)/block.y);
    //         auto cuSt = cv::cuda::StreamAccessor::getStream(stream);
    //         bayer2mask_dual_hsv_kernel<<<grid,block,0,cuSt>>>(
    //             src, dst_arrow, dst_armor,
    //             h_low_a, h_high_a, s_low_a, s_high_a, v_low_a, v_high_a,
    //             h_low_m, h_high_m, s_low_m, s_high_m, v_low_m, v_high_m
    //         );
    //     }
    // }
}

// 1. getColorFromRGB_hsv (RGB 版本)
// 输入 (Inputs):

// r, g, b: 三张大小相等的 CV_8UC1 单通道 GpuMat 图像（原图分离得到）。

// 12 个整型阈值：分别对应 Arrow (扇叶) 和 Armor (装甲板) 的 H, S, V 上下限。

// stream: 用于异步执行的 CUDA 队列。

// 输出 (Outputs):

// dst_arrow: 扇叶的二值化掩码图（CV_8UC1，在设定 HSV 范围内为 255，否则为 0）。尺寸与输入图像一致。

// dst_armor: 装甲板的二值化掩码图。尺寸与输入图像一致。

// 2. getColorFromBayerRG_hsv (Bayer 版本)
// 输入 (Inputs):

// src: 一张 CV_8UC1 的 GpuMat 原图，且数据格式必须严格为 Bayer-RG。

// 12 个整型阈值（同上）。

// stream（同上）。

// 输出 (Outputs):

// dst_arrow 和 dst_armor: 两张二值化掩码图（CV_8UC1）。

// 关键注意：这两张输出掩码图的长宽必须提前分配为输入 src 长宽的一半。因为核函数是以 2x2 为步长在原图上采样的。