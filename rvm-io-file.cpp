
#include "rvm-io-file.h"

MattingIOFile::MattingIOFile( size_t picSizeSrc, size_t picSizeFgr, const std::vector<std::string>& args, Logger& logger )
  : m_logger( logger )
  , m_bufStageSrc( nullptr )
  , m_bufStageFgr( nullptr )
  , m_picSizeSrc( picSizeSrc )
  , m_picSizeFgr( picSizeFgr )
  , m_inFilepaths( args )
{
  m_itInFilepath = m_inFilepaths.begin();
}

bool MattingIOFile::KeepRunning()
{
  return m_itInFilepath != m_inFilepaths.end();
}

MattingIOFile::~MattingIOFile()
{
  FreeStagingBuffers();
}

bool MattingIOFile::InitStagingBuffers()
{
  // Allocate host memory for staging input/output
  //
  if( cudaMallocHost( &m_bufStageSrc, m_picSizeSrc ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA host memory. Exiting." );
    return false;
  }

  if( cudaMallocHost( &m_bufStageFgr, m_picSizeFgr ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA host memory. Exiting." );
    return false;
  }

  m_logger.log( nvinfer1::ILogger::Severity::kINFO, "Successfully allocated host memory for staging input/output." );

  return true;
}

bool MattingIOFile::FreeStagingBuffers()
{
  if( m_bufStageSrc && cudaFreeHost( m_bufStageSrc ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA host memory. Exiting." );
    return false;
  }
  if( m_bufStageFgr && cudaFreeHost( m_bufStageFgr ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to free CUDA host memory. Exiting." );
    return false;
  }
  m_logger.log( nvinfer1::ILogger::Severity::kINFO, "Successfully freed CUDA host memory for staging." );

  return true;
}
