#include "rvm.hpp"

#define PIC_WIDTH 512
#define PIC_HEIGHT 288

bool LoadTRTEngineMatting( const std::string& strTrtEngineFilepath
                         , Logger& logger
			 , nvinfer1::IRuntime* pTrtRuntime
			 , nvinfer1::ICudaEngine** ppTrtCudaEngine
			 , nvinfer1::IExecutionContext** ppTrtExecutionContext )
{

  logger.log( nvinfer1::ILogger::Severity::kINFO, (std::string("Loading Matting TRT Engine '") + strTrtEngineFilepath + "'").c_str() );

  FILE* fileTrtEngine = fopen( strTrtEngineFilepath.c_str(), "rb" );
  if( !fileTrtEngine )
  {
    logger.log( nvinfer1::ILogger::Severity::kERROR, (std::string("Bad filepath for TRT engine '") + strTrtEngineFilepath + "'. Exiting").c_str() );
    return false;
  }

  // Get the file size
  fseek( fileTrtEngine, 0, SEEK_END );
  long fileSizeTrtEngine = ftell(fileTrtEngine);
  fseek( fileTrtEngine, 0, SEEK_SET );

  // Allocate memory to hold the file content
  unsigned char *bufferTrtEngine = (unsigned char *)malloc( fileSizeTrtEngine) ;
  if (!bufferTrtEngine)
  {
    logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate memory for TRT engine");
    fclose( fileTrtEngine );
    return false;
  }

  // Read the file content into the allocated memory
  size_t bytesRead = fread( bufferTrtEngine, 1, fileSizeTrtEngine, fileTrtEngine );
  if( bytesRead != fileSizeTrtEngine )
  {
    logger.log( nvinfer1::ILogger::Severity::kERROR, ("Failed to read the TRT engine file '" + strTrtEngineFilepath + "'.").c_str() );
    free( bufferTrtEngine );
    fclose( fileTrtEngine );
    return false;
  }

  fclose( fileTrtEngine );
  logger.log( nvinfer1::ILogger::Severity::kINFO, (std::string("File size: ") + std::to_string(fileSizeTrtEngine) + " bytes.").c_str() );

  // Deserialize
  //
  *ppTrtCudaEngine = pTrtRuntime->deserializeCudaEngine( bufferTrtEngine, fileSizeTrtEngine );
  free( bufferTrtEngine );
  if( !*ppTrtCudaEngine )
  {
    logger.log( nvinfer1::ILogger::Severity::kERROR, ("Invalid TRT engine file '" + strTrtEngineFilepath + "'.").c_str() );
    return false;
  }

  logger.log( nvinfer1::ILogger::Severity::kINFO, ("Successfully loaded Matting TRT Engine '" + strTrtEngineFilepath + "'.").c_str() );

  // Print out inputs/outpus
  //
  std::cerr << "=============\nInputs / Outputs (i.e. Bindings) :\n";
  int numBindings = (*ppTrtCudaEngine)->getNbBindings();
  for (int i = 0; i < numBindings; ++i)
  {
    nvinfer1::Dims dims = (*ppTrtCudaEngine)->getBindingDimensions(i);
    std::cerr << i << " : " << (*ppTrtCudaEngine)->getBindingName(i) << " : dims=";
    for (int j = 0; j < dims.nbDims; ++j)
    {
      std::cerr << dims.d[j];
      if (j < dims.nbDims - 1)
      {
        std::cerr << "x";
      }
    }
    std::cerr << " , dtype=" << (int) (*ppTrtCudaEngine)->getBindingDataType(i) << " ";
    std::cerr << ((*ppTrtCudaEngine)->bindingIsInput(i) ? "IN" : "OUT") << std::endl;
  }
  std::cerr << "=============\n\n";

  *ppTrtExecutionContext = (*ppTrtCudaEngine)->createExecutionContext();
  if( !*ppTrtExecutionContext )
  {
    logger.log( nvinfer1::ILogger::Severity::kERROR, ("Failed to create TRT execution context from engine file '" + strTrtEngineFilepath + "'.").c_str() );
    return false;
  }

  logger.log( nvinfer1::ILogger::Severity::kINFO, ("Successfully created TRT execution context from engine '" + strTrtEngineFilepath + "'.").c_str() );


  return true;
}

bool MattingRunner::ProcessPictures( const std::vector<std::string> &args )
{
  for( std::vector<std::string>::const_iterator itInFilepath = args.begin(); itInFilepath != args.end(); itInFilepath++ )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kINFO, (std::string("itInFilepath = ") + *itInFilepath).c_str() );
    if( !ConsumeNextInput( itInFilepath->c_str() ) )
    {
      m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to consume input. Exiting." );
      return false;
    }

    if( !RunInference() )
    {
      m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to run TRT inference. Exiting." );
      return false;
    }

    std::string outFilepath = *itInFilepath + ".fgr";
    if( !ProduceNextOutput( outFilepath.c_str() ) )
    {
      m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to produce output. Exiting." );
      return false;
    }

    m_logger.log( nvinfer1::ILogger::Severity::kINFO
	        , (std::string("Processed picture '") + outFilepath + "'").c_str() );

    SwapRecurrents();
  }
  return true;
}

// raw RGB file implementation:
//

bool MattingRunner::InitStagingBuffers()
{
  // Allocate host memory for staging input/output
  //
  if( cudaMallocHost( &m_bufStageSrc, m_picSizeRGB ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA host memory. Exiting." );
    assert( false );
    return false;
  }

  if( cudaMallocHost( &m_bufStageFgr, m_picSizeRGBA ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA host memory. Exiting." );
    assert( false );
    return false;
  }

  m_logger.log( nvinfer1::ILogger::Severity::kINFO, "Successfully allocated host memory for staging input/output." );

  return true;
}

bool MattingRunner::FreeStagingBuffers()
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

bool MattingRunner::ConsumeNextInput( const char* szInRawRGBFilepath )
{
  if( !szInRawRGBFilepath )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kINTERNAL_ERROR
	        , "szInRawRGBFilepath agument is NULL. Exiting." );
    assert( false );
    return false;
  }

  m_logger.log( nvinfer1::ILogger::Severity::kINFO, (std::string("szInRawRGBFilepath = ") + szInRawRGBFilepath).c_str() );

  FILE* fileRawFrame = fopen( szInRawRGBFilepath, "rb" );
  if( !fileRawFrame )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR
	        , (std::string("Bad filepath for raw RGB frame '") + szInRawRGBFilepath + "'. Exiting").c_str() );
    return false;
  }

  // Read entire file
  //
  size_t bytesRead = fread( m_bufStageSrc, 1, m_picSizeRGB, fileRawFrame );
  if( bytesRead != m_picSizeRGB )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, ( std::string("Failed to read raw RGB file '") + szInRawRGBFilepath + "'.").c_str() );
    fclose( fileRawFrame );
    return false;
  }

  fclose( fileRawFrame );

  if( cudaMemcpyAsync( m_cuBufs[IDX_SRC], m_bufStageSrc, m_picSizeRGB, cudaMemcpyHostToDevice, m_cudaStream ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to do HtoD cudaMemcpyAsync().");
    return false;
  }

  return true;
}

bool MattingRunner::ProduceNextOutput( const char* szInRawRGBAFilepath )
{
  if( !szInRawRGBAFilepath )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kINTERNAL_ERROR
	        , "szInRawRGBFilepath agument is NULL. Exiting." );
    assert( false );
    return false;
  }

  if( cudaMemcpyAsync( m_bufStageFgr, m_cuBufs[IDX_FGR], m_picSizeRGBA, cudaMemcpyDeviceToHost, m_cudaStream ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to do DtoH cudaMemcpyAsync().");
    return false;
  }

  FILE* fileRawFrame = fopen( szInRawRGBAFilepath, "wb" );
  if( !fileRawFrame )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR
	        , (std::string("Bad filepath for raw RGBA frame '") + szInRawRGBAFilepath + "'. Exiting").c_str() );
    return false;
  }

  // Pretty sure I don't need that but keeping for now.
  if( cudaStreamSynchronize( m_cudaStream ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "cudaStreamSynchronize() failed. Exiting." );
    return false;
  }

  // write out to file
  //
  size_t bytesWritten = fwrite( m_bufStageFgr, 1, m_picSizeRGBA, fileRawFrame );
  if( bytesWritten != m_picSizeRGBA )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, ( std::string("Failed to write out raw RGBA file '") + szInRawRGBAFilepath + "'.").c_str() );
    fclose( fileRawFrame );
    return false;
  }

  fclose( fileRawFrame );

  return true;
}

MattingRunner::~MattingRunner()
{
  bool bRet = FreeStagingBuffers();
  assert( bRet );
}

MattingRunner::MattingRunner( const std::vector<std::string>& args
                            , size_t picWidth, size_t picHeight
			    , nvinfer1::IExecutionContext* pTrtExecutionContext
			    , Logger& logger )
  : RVMBase( pTrtExecutionContext
           , picWidth
	   , picHeight
	   , logger )
  , m_bufStageSrc(nullptr)
  , m_bufStageFgr(nullptr)
{
  bool bRet = InitStagingBuffers();
  assert( bRet );
}

struct Arguments_ExampleRVM
{
  std::string strTrtEngineFilepath;
  std::vector<std::string> strArgs;

  bool parse( int argc, char* argv[], Logger& logger );
};

bool Arguments_ExampleRVM::parse( int argc, char* argv[], Logger& logger )
{
  if( argc < 2 )
  {
    logger.log( nvinfer1::ILogger::Severity::kERROR, "Must specify path to RVM TRT engine as first argument. Exiting." );
    return false;
  }

  strTrtEngineFilepath = argv[1];

  for( int i = 2; i < argc; i++ )
  {
    strArgs.push_back( argv[i] );
  }

  return true;
}

#define ERROR_TRT_RUNTIME_INIT_FAILED         1
#define ERROR_TRT_ENGINE_LOAD_FAILED          2

int main( int argc, char* argv[] )
{
  Logger logger;
  logger.log( nvinfer1::ILogger::Severity::kINFO, "C++ TensorRT RVM Inference example" );
  
  Arguments_ExampleRVM args;
  args.parse( argc, argv, logger );

  // Initialize TRT
  //
  nvinfer1::IRuntime* pTrtRuntime = nvinfer1::createInferRuntime( logger );
  if( !pTrtRuntime )
  {
    logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to initialize TRT Runtime");
    return ERROR_TRT_RUNTIME_INIT_FAILED;
  }
  logger.log( nvinfer1::ILogger::Severity::kINFO, "TRT Runtime Created." );

  // Initialize RVM TRT Engine
  //
  nvinfer1::ICudaEngine* pTrtCudaEngine = nullptr;
  nvinfer1::IExecutionContext* pTrtExecutionContext = nullptr;
  if( !LoadTRTEngineMatting( args.strTrtEngineFilepath, logger, pTrtRuntime, &pTrtCudaEngine, &pTrtExecutionContext ) )
  {
    return ERROR_TRT_ENGINE_LOAD_FAILED;
  }

  // Process Loop
  //
  MattingRunner rvmRunState( args.strArgs, PIC_WIDTH, PIC_HEIGHT, pTrtExecutionContext, logger);

  rvmRunState.ProcessPictures( args.strArgs );

  // Cleanup
  //
  pTrtExecutionContext->destroy();
  pTrtCudaEngine->destroy();
  pTrtRuntime->destroy();

  return 0;
}
