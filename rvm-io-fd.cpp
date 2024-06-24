
#include "rvm-io-fd.h"

#include <assert.h>
#include <stdio.h>
#include <signal.h>

static volatile sig_atomic_t stop = 0;
void handle_sigint(int sig)
{
  stop = 1;
}

MattingIOFd::MattingIOFd( size_t picSizeSrc, size_t picSizeFgr
                        , const std::vector<std::string>& args, Logger& logger )
  : IOBaseHost( picSizeSrc, picSizeFgr, logger )
{
  if( !InitStagingBuffers() )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to initialize staging buffer. Exiting." );
    assert( false );
  }

  // Set up the SIGINT signal handler
  struct sigaction sa;
  sa.sa_handler = handle_sigint;
  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask);

  if( sigaction(SIGINT, &sa, NULL) == -1 )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to setup sigaction. Exiting." );
    assert( false );
  }
}

bool MattingIOFd::IsStillRunning()
{
  return !stop && !feof(stdin);
}

bool MattingIOFd::ConsumeNextInput( void* cuBufSrc, cudaStream_t cudaStream )
{
  // Read entire file
  //
  size_t bytesRead = fread( m_bufStageSrc, 1, m_picSizeSrc, stdin );
  if( bytesRead != m_picSizeSrc )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to read raw RGB file from stdin." );
    return false;
  }

  if( cudaMemcpyAsync( cuBufSrc, m_bufStageSrc, m_picSizeSrc, cudaMemcpyHostToDevice, cudaStream ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to do HtoD cudaMemcpyAsync().");
    return false;
  }

  return true;
}

bool MattingIOFd::ProduceNextOutput( void* cuBufFgr, cudaStream_t cudaStream )
{
  if( cudaMemcpyAsync( m_bufStageFgr, cuBufFgr, m_picSizeFgr, cudaMemcpyDeviceToHost, cudaStream ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to do DtoH cudaMemcpyAsync().");
    return false;
  }

  // Pretty sure I don't need that but keeping for now.
  if( cudaStreamSynchronize( cudaStream ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "cudaStreamSynchronize() failed. Exiting." );
    return false;
  }

  // write out to file
  //
  size_t bytesWritten = fwrite( m_bufStageFgr, 1, m_picSizeFgr, stdout );
  if( bytesWritten != m_picSizeFgr )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to write raw RGBA file to stdout." );
    return false;
  }

  return true;
}
