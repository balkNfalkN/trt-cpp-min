
#include "rvm-io-file.h"
#include <assert.h>

MattingIOFile::MattingIOFile( size_t picSizeSrc, size_t picSizeFgr
                            , const std::vector<std::string>& args, Logger& logger )
  : m_logger( logger )
  , m_bufStageSrc( nullptr )
  , m_bufStageFgr( nullptr )
  , m_picSizeSrc( picSizeSrc )
  , m_picSizeFgr( picSizeFgr )
  , m_inFilepaths( args )
{
  if( !InitStagingBuffers() )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to initialize staging buffer. Exiting." );
    assert( false );
  }
  m_itInFilepath = m_inFilepaths.begin();
}

bool MattingIOFile::IsStillRunning()
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

bool MattingIOFile::ConsumeNextInput( void* cuBufSrc, cudaStream_t cudaStream )
{
  if( m_itInFilepath == m_inFilepaths.end() )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kINTERNAL_ERROR
	        , "ConsumeNextInput past end. Exiting." );
    assert( false );
    return false;
  }

  FILE* fileRawFrame = fopen( m_itInFilepath->c_str(), "rb" );
  if( !fileRawFrame )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR
	        , (std::string("Bad filepath for raw RGB frame '") + *m_itInFilepath + "'. Exiting").c_str() );
    return false;
  }

  // Read entire file
  //
  size_t bytesRead = fread( m_bufStageSrc, 1, m_picSizeSrc, fileRawFrame );
  if( bytesRead != m_picSizeSrc )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, ( std::string("Failed to read raw RGB file '") + *m_itInFilepath + "'.").c_str() );
    fclose( fileRawFrame );
    return false;
  }

  fclose( fileRawFrame );

  if( cudaMemcpyAsync( cuBufSrc, m_bufStageSrc, m_picSizeSrc, cudaMemcpyHostToDevice, cudaStream ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to do HtoD cudaMemcpyAsync().");
    return false;
  }

  return true;
}

bool MattingIOFile::ProduceNextOutput( void* cuBufFgr, cudaStream_t cudaStream )
{
  if( m_itInFilepath == m_inFilepaths.end() )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kINTERNAL_ERROR
	        , "szInRawRGBFilepath agument is NULL. Exiting." );
    assert( false );
    return false;
  }

  std::string outFilepath = *m_itInFilepath + ".fgr";
  ++m_itInFilepath;

  if( cudaMemcpyAsync( m_bufStageFgr, cuBufFgr, m_picSizeFgr, cudaMemcpyDeviceToHost, cudaStream ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to do DtoH cudaMemcpyAsync().");
    return false;
  }

  FILE* fileRawFrame = fopen( outFilepath.c_str(), "wb" );
  if( !fileRawFrame )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR
	        , (std::string("Bad filepath for raw RGBA frame '") + outFilepath + "'. Exiting").c_str() );
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
  size_t bytesWritten = fwrite( m_bufStageFgr, 1, m_picSizeFgr, fileRawFrame );
  if( bytesWritten != m_picSizeFgr )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, ( std::string("Failed to write out raw RGBA file '") + outFilepath + "'.").c_str() );
    fclose( fileRawFrame );
    return false;
  }

  fclose( fileRawFrame );

  m_logger.log( nvinfer1::ILogger::Severity::kINFO
              , (std::string("Processed picture '") + outFilepath + "'").c_str() );

  return true;
}
