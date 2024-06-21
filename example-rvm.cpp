
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeBase.h>

#include <cstdint>
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <iostream>
#include <assert.h>

class Logger : public nvinfer1::ILogger
{
  public:
  void log( Severity severity, const nvinfer1::AsciiChar *msg ) noexcept override
  {
    std::string s;
    switch( severity ) 
    {
      case Severity::kINTERNAL_ERROR:
        s = "INTERNAL_ERROR";
        break;
      case Severity::kERROR:
        s = "ERROR";
        break;
      case Severity::kWARNING:
        s = "WARNING";
        break;
      case Severity::kINFO:
        s = "INFO";
        break;
      case Severity::kVERBOSE:
        s = "VERBOSE";
        break;
    }
    std::cerr << s << ": " << msg << std::endl;
  }
};

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
  std::cout << "=============\nInputs / Outputs (i.e. Bindings) :\n";
  int numBindings = (*ppTrtCudaEngine)->getNbBindings();
  for (int i = 0; i < numBindings; ++i)
  {
    nvinfer1::Dims dims = (*ppTrtCudaEngine)->getBindingDimensions(i);
    std::cout << i << " : " << (*ppTrtCudaEngine)->getBindingName(i) << " : dims=";
    for (int j = 0; j < dims.nbDims; ++j)
    {
      std::cout << dims.d[j];
      if (j < dims.nbDims - 1)
      {
        std::cout << "x";
      }
    }
    std::cout << " , dtype=" << (int) (*ppTrtCudaEngine)->getBindingDataType(i) << " ";
    std::cout << ((*ppTrtCudaEngine)->bindingIsInput(i) ? "IN" : "OUT") << std::endl;
  }
  std::cout << "=============\n\n";

  *ppTrtExecutionContext = (*ppTrtCudaEngine)->createExecutionContext();
  if( !*ppTrtExecutionContext )
  {
    logger.log( nvinfer1::ILogger::Severity::kERROR, ("Failed to create TRT execution context from engine file '" + strTrtEngineFilepath + "'.").c_str() );
    return false;
  }

  logger.log( nvinfer1::ILogger::Severity::kINFO, ("Successfully created TRT execution context from engine '" + strTrtEngineFilepath + "'.").c_str() );


  return true;
}

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
class RVMRunState
{
  public:
    RVMRunState( const std::vector<std::string>& args, nvinfer1::IExecutionContext* pTrtExecutionContext, Logger& logger );
    ~RVMRunState();

  private:
    nvinfer1::IExecutionContext* m_pTrtExecutionContext;
    Logger& m_logger;
    cudaStream_t m_cudaStream;

    const size_t m_picWidth = 512;
    const size_t m_picHeight = 288;
    const size_t m_picSize = m_picWidth*m_picHeight*3*sizeof(uint8_t);
    const size_t m_sizeR1 = 1*16*144*256*sizeof(uint16_t); 
    const size_t m_sizeR2 = 1*32*72*128*sizeof(uint16_t); 
    const size_t m_sizeR3 = 1*64*36*64*sizeof(uint16_t); 
    const size_t m_sizeR4 = 1*128*18*32*sizeof(uint16_t); 

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

    void* m_bufStageSrc;
    void* m_bufStageFgr;

  // Common
  //
  private:
    bool InitBuffers();
    bool FreeBuffers();
    bool RunInference();
    void SwapRecurrents();

  // Templated
  //
  private:
    bool ConsumeInput( const char* szInPic = nullptr );
    bool ProduceOutput( const char* szOutPic = nullptr );

  public:
    bool ProcessPictures( const std::vector<std::string>& args );
};

// raw RGB file implementation:
//
//

bool RVMRunState::ConsumeInput( const char* szInRawRGBFilepath )
{
  if( !szInRawRGBFilepath )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kINTERNAL_ERROR
	        , "szInRawRGBFilepath agument is NULL. Exiting." );
    assert( false );
    return false;
  }

  FILE* fileRawFrame = fopen( szInRawRGBFilepath, "rb" );
  if( !fileRawFrame )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR
	        , (std::string("Bad filepath for raw RGB frame '") + szInRawRGBFilepath + "'. Exiting").c_str() );
    return false;
  }

  // Read entire file
  //
  size_t bytesRead = fread( m_bufStageSrc, 1, m_picSize, fileRawFrame );
  if( bytesRead != m_picSize )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, ( std::string("Failed to read raw RGB file '") + szInRawRGBFilepath + "'.").c_str() );
    fclose( fileRawFrame );
    return false;
  }

  fclose( fileRawFrame );

  if( cudaMemcpyAsync( m_cuBufs[IDX_SRC], m_bufStageSrc, m_picSize, cudaMemcpyHostToDevice ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to do HtoD cudaMemcpyAsync().");
    return false;
  }

  return true;
}

bool RVMRunState::ProduceOutput( const char* szInRawRGBAFilepath )
{
  if( !szInRawRGBAFilepath )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kINTERNAL_ERROR
	        , "szInRawRGBFilepath agument is NULL. Exiting." );
    assert( false );
    return false;
  }

  if( cudaMemcpyAsync( m_bufStageFgr, m_cuBufs[IDX_SRC], m_picSize, cudaMemcpyDeviceToHost ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to do DtoH cudaMemcpyAsync().");
    return false;
  }

  FILE* fileRawFrame = fopen( szInRawRGBAFilepath, "wb" );
  if( !fileRawFrame )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR
	        , (std::string("Bad filepath for raw RGB frame '") + szInRawRGBAFilepath + "'. Exiting").c_str() );
    return false;
  }

  // write out to file
  //
  size_t bytesWritten = fwrite( m_bufStageFgr, 1, m_picSize, fileRawFrame );
  if( bytesWritten != m_picSize )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, ( std::string("Failed to write out raw RGBA file '") + szInRawRGBAFilepath + "'.").c_str() );
    fclose( fileRawFrame );
    return false;
  }

  fclose( fileRawFrame );

  return true;
}

void RVMRunState::SwapRecurrents()
{
  void* tmpR1 = m_cuBufs[IDX_R1I];
  void* tmpR2 = m_cuBufs[IDX_R2I];
  void* tmpR3 = m_cuBufs[IDX_R3I];
  void* tmpR4 = m_cuBufs[IDX_R1I];

  m_cuBufs[IDX_R1I] = m_cuBufs[IDX_R1O];
  m_cuBufs[IDX_R2I] = m_cuBufs[IDX_R2O];
  m_cuBufs[IDX_R3I] = m_cuBufs[IDX_R3O];
  m_cuBufs[IDX_R4I] = m_cuBufs[IDX_R4O];

  m_cuBufs[IDX_R1O] = tmpR1;
  m_cuBufs[IDX_R2O] = tmpR2;
  m_cuBufs[IDX_R3O] = tmpR3;
  m_cuBufs[IDX_R4O] = tmpR4;
}

bool RVMRunState::InitBuffers()
{
  // Allocate device memory buffers for bindings
  //
  if( cudaMalloc( &m_cuBufs[IDX_SRC], m_picSize ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA memory. Exiting." );
    assert( false );
  }

  if( cudaMalloc( &m_cuBufs[IDX_FGR], m_picSize ) != cudaError_t::cudaSuccess )
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

  m_logger.log( nvinfer1::ILogger::Severity::kINFO, "Successfully allocated device memory for bindings." );

  // Allocate host memory for staging input/output
  //
  if( cudaMallocHost( &m_bufStageSrc, m_picWidth*m_picHeight*3*sizeof(uint8_t) ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA host memory. Exiting." );
    assert( false );
  }

  if( cudaMallocHost( &m_bufStageFgr, m_picWidth*m_picHeight*4*sizeof(uint8_t) ) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to allocate CUDA host memory. Exiting." );
    assert( false );
  }

  m_logger.log( nvinfer1::ILogger::Severity::kINFO, "Successfully allocated host memory for staging input/output." );

  // Initialize recurrents to 0
  //
  if( cudaMemset( m_cuBufs[IDX_R1I], 0, m_sizeR1) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to initialize CUDA memory. Exiting." );
    assert( false );
  }
  if( cudaMemset( m_cuBufs[IDX_R2I], 0, m_sizeR2) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to initialize CUDA memory. Exiting." );
    assert( false );
  }
  if( cudaMemset( m_cuBufs[IDX_R3I], 0, m_sizeR3) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to initialize CUDA memory. Exiting." );
    assert( false );
  }
  if( cudaMemset( m_cuBufs[IDX_R4I], 0, m_sizeR4) != cudaError_t::cudaSuccess )
  {
    m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to initialize CUDA memory. Exiting." );
    assert( false );
  }

  return true;
}

bool RVMRunState::FreeBuffers()
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
  m_logger.log( nvinfer1::ILogger::Severity::kINFO, "Successfully freed CUDA host memory for bindings." );

  return true;
}

RVMRunState::~RVMRunState()
{
  bool bRet = FreeBuffers();
  assert( bRet );
}

RVMRunState::RVMRunState( const std::vector<std::string>& args, nvinfer1::IExecutionContext* pTrtExecutionContext, Logger& logger )
  : m_pTrtExecutionContext(pTrtExecutionContext)
  , m_logger(logger)
  , m_bufStageSrc(nullptr)
  , m_bufStageFgr(nullptr)
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
  bool bRet = InitBuffers();
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
  RVMRunState rvmRunState( args.strArgs, pTrtExecutionContext, logger);

  // Cleanup
  //
  pTrtExecutionContext->destroy();
  pTrtCudaEngine->destroy();
  pTrtRuntime->destroy();

  return 0;
}
