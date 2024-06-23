
#include "rvm-base.hpp"

RVMBase::RVMBase( nvinfer1::IExecutionContext* pTrtExecutionContext, size_t picSizeSrc, size_t picSizeFgr, Logger& logger )
  : m_pTrtExecutionContext(pTrtExecutionContext)
  , m_logger(logger)
{
  for( int i = 0; i < IDX_NUM; i++ )
  {
    m_cuBufs[i] = nullptr;
  }

  // Create stream
  //
  if( cudaStreamCreate( &m_cudaStream ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to create cudaStream. Exiting." );
    assert( false );
  }
  m_logger.log( nvinfer1::ILogger::Severity::kINFO, "Successfully created cudaStream." );

  // Initialize CUDA buffers.
  //
  bool bRet = InitBuffers( picSizeSrc, picSizeFgr );
  assert( bRet );
}

RVMBase::~RVMBase()
{
  bool bRet = FreeBuffers();
  assert( bRet );
}

bool RVMBase::InitBuffers( size_t picSizeSrc, size_t picSizeFgr )
{
  // Allocate device memory buffers for bindings
  //
  if( cudaMalloc( &m_cuBufs[IDX_SRC], picSizeSrc ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA memory. Exiting." );
    assert( false );
  }

  if( cudaMalloc( &m_cuBufs[IDX_FGR], picSizeFgr ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA memory. Exiting." );
    assert( false );
  }

  if( cudaMalloc( &m_cuBufs[IDX_R1I], m_sizeR1*sizeof(uint16_t) ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA memory. Exiting." );
    assert( false );
  }
  if( cudaMalloc( &m_cuBufs[IDX_R1O], m_sizeR1*sizeof(uint16_t) ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA memory. Exiting." );
    assert( false );
  }

  if( cudaMalloc( &m_cuBufs[IDX_R2I], m_sizeR1*sizeof(uint16_t) ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA memory. Exiting." );
    assert( false );
  }
  if( cudaMalloc( &m_cuBufs[IDX_R2O], m_sizeR1*sizeof(uint16_t) ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA memory. Exiting." );
    assert( false );
  }

  if( cudaMalloc( &m_cuBufs[IDX_R3I], m_sizeR1*sizeof(uint16_t) ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA memory. Exiting." );
    assert( false );
  }
  if( cudaMalloc( &m_cuBufs[IDX_R3O], m_sizeR1*sizeof(uint16_t) ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA memory. Exiting." );
    assert( false );
  }

  if( cudaMalloc( &m_cuBufs[IDX_R4I], m_sizeR1*sizeof(uint16_t) ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA memory. Exiting." );
    assert( false );
  }
  if( cudaMalloc( &m_cuBufs[IDX_R4O], m_sizeR1*sizeof(uint16_t) ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA memory. Exiting." );
    assert( false );
  }

  // Initialize recurrents to 0
  //
  if( cudaMemset( m_cuBufs[IDX_R1I], 0, m_sizeR1) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to initialize CUDA memory. Exiting." );
    assert( false );
    return false;
  }
  if( cudaMemset( m_cuBufs[IDX_R2I], 0, m_sizeR2) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to initialize CUDA memory. Exiting." );
    assert( false );
    return false;
  }
  if( cudaMemset( m_cuBufs[IDX_R3I], 0, m_sizeR3) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to initialize CUDA memory. Exiting." );
    assert( false );
    return false;
  }
  if( cudaMemset( m_cuBufs[IDX_R4I], 0, m_sizeR4) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to initialize CUDA memory. Exiting." );
    assert( false );
    return false;
  }

  m_logger.log( nvinfer1::ILogger::Severity::kINFO, "Successfully allocated device memory for bindings." );

  return true;
}

bool RVMBase::FreeBuffers()
{
  if( m_cuBufs[IDX_SRC] && cudaFree( m_cuBufs[IDX_SRC] ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA memory. Exiting." );
    return false;
  }
  if( m_cuBufs[IDX_FGR] && cudaFree( m_cuBufs[IDX_FGR] ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA memory. Exiting." );
    return false;
  }
  if( m_cuBufs[IDX_R1I] && cudaFree( m_cuBufs[IDX_R1I] ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA memory. Exiting." );
    return false;
  }
  if( m_cuBufs[IDX_R1O] && cudaFree( m_cuBufs[IDX_R1O] ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA memory. Exiting." );
    return false;
  }
  if( m_cuBufs[IDX_R2I] && cudaFree( m_cuBufs[IDX_R2I] ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA memory. Exiting." );
    return false;
  }
  if( m_cuBufs[IDX_R2O] && cudaFree( m_cuBufs[IDX_R2O] ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA memory. Exiting." );
    return false;
  }
  if( m_cuBufs[IDX_R3I] && cudaFree( m_cuBufs[IDX_R3I] ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA memory. Exiting." );
    return false;
  }
  if( m_cuBufs[IDX_R3O] && cudaFree( m_cuBufs[IDX_R3O] ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA memory. Exiting." );
    return false;
  }
  if( m_cuBufs[IDX_R4I] && cudaFree( m_cuBufs[IDX_R4I] ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA memory. Exiting." );
    return false;
  }
  if( m_cuBufs[IDX_R4O] && cudaFree( m_cuBufs[IDX_R4O] ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA memory. Exiting." );
    return false;
  }
  m_logger.log( nvinfer1::ILogger::Severity::kINFO, "Successfully freed CUDA memory for bindings." );

  return true;
}

bool RVMBase::RunInference()
{
  if( !m_pTrtExecutionContext->enqueueV2( m_cuBufs, m_cudaStream, nullptr ) )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "TRT enqueueV2() failed. Exiting." );
    return false;
  }

  return true;
}

void RVMBase::SwapRecurrents()
{
  void* tmpR1 = m_cuBufs[IDX_R1I];
  void* tmpR2 = m_cuBufs[IDX_R2I];
  void* tmpR3 = m_cuBufs[IDX_R3I];
  void* tmpR4 = m_cuBufs[IDX_R4I];

  m_cuBufs[IDX_R1I] = m_cuBufs[IDX_R1O];
  m_cuBufs[IDX_R2I] = m_cuBufs[IDX_R2O];
  m_cuBufs[IDX_R3I] = m_cuBufs[IDX_R3O];
  m_cuBufs[IDX_R4I] = m_cuBufs[IDX_R4O];

  m_cuBufs[IDX_R1O] = tmpR1;
  m_cuBufs[IDX_R2O] = tmpR2;
  m_cuBufs[IDX_R3O] = tmpR3;
  m_cuBufs[IDX_R4O] = tmpR4;
}
