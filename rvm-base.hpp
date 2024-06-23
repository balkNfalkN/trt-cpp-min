#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>
#include <assert.h>

#include "logger.hpp"

// TODO Put in RVM headers later
// 0 : src : dims=1x288x512x3 , dtype=5 IN
// 9 : fgr : dims=1x288x512x4 , dtype=5 OUT
//
// 1 : r1i : dims=1x16x144x256 , dtype=1 IN
// 2 : r2i : dims=1x32x72x128 , dtype=1 IN
// 3 : r3i : dims=1x64x36x64 , dtype=1 IN
// 4 : r4i : dims=1x128x18x32 , dtype=1 IN
//
// 5 : r4o : dims=1x128x18x32 , dtype=1 OUT
// 6 : r3o : dims=1x64x36x64 , dtype=1 OUT
// 7 : r2o : dims=1x32x72x128 , dtype=1 OUT
// 8 : r1o : dims=1x16x144x256 , dtype=1 OUT
class RVMBase
{
  public:
    RVMBase( nvinfer1::IExecutionContext* pTrtExecutionContext
	   , size_t picWidth, size_t picHeight, Logger& logger );
    virtual ~RVMBase();

  protected:
    nvinfer1::IExecutionContext* m_pTrtExecutionContext;
    Logger& m_logger;
    cudaStream_t m_cudaStream;

    const size_t m_picWidth;
    const size_t m_picHeight;
    const size_t m_picSizeRGB = m_picWidth*m_picHeight*3*sizeof(uint8_t);
    const size_t m_picSizeRGBA = m_picWidth*m_picHeight*4*sizeof(uint8_t);
    size_t m_sizeR1;
    size_t m_sizeR2;
    size_t m_sizeR3;
    size_t m_sizeR4;

    enum BindingIndices
    {
      IDX_SRC = 0,
      IDX_R1I = 1,
      IDX_R2I = 2,
      IDX_R3I = 3,
      IDX_R4I = 4,
      IDX_R1O = 8,
      IDX_R2O = 7,
      IDX_R3O = 6,
      IDX_R4O = 5,
      IDX_FGR = 9,
      IDX_NUM = 10,
    };
    void* m_cuBufs[IDX_NUM];

  protected:
    bool InitBuffers();
    bool FreeBuffers();
    bool RunInference();
    void SwapRecurrents();

  protected:
    inline void* GetCuBufSrc()
    {
      return m_cuBufs[IDX_SRC];
    }
    inline void* GetCuBufFgr()
    {
      return m_cuBufs[IDX_SRC];
    }
};

