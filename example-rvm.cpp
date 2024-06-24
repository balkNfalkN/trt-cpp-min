#include "rvm.hpp"
#include "rvm-io-file.h"

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

#define Q(x) #x
#define QUOTE(x) Q(x)

#define MATTING_IO_IMPL_QUOTED QUOTE(MATTING_IO_IMPL)

int main( int argc, char* argv[] )
{
  Logger logger;
  logger.log( nvinfer1::ILogger::Severity::kINFO, (std::string("C++ TensorRT RVM Inference example - IO: ") + MATTING_IO_IMPL_QUOTED).c_str() );
  
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
  MattingRunner<MATTING_IO_IMPL> rvmRunState( args.strArgs, PIC_WIDTH, PIC_HEIGHT, pTrtExecutionContext, logger);

  rvmRunState.ProcessPictures( args.strArgs );

  // Cleanup
  //
  pTrtExecutionContext->destroy();
  pTrtCudaEngine->destroy();
  pTrtRuntime->destroy();

  return 0;
}
