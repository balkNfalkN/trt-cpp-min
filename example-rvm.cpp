
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeBase.h>

#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <iostream>

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
			 , nvinfer1::ICudaEngine** ppTrtCudaEngine )
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
  if( !ppTrtCudaEngine )
  {
    logger.log( nvinfer1::ILogger::Severity::kERROR, ("Invalid TRT engine file '" + strTrtEngineFilepath + "'.").c_str() );
    return false;
  }

  logger.log( nvinfer1::ILogger::Severity::kINFO, ("Successfully loaded Matting TRT Engine '" + strTrtEngineFilepath + "'.").c_str() );

  // Print out inputs/outpus
  //
  std::cout << "=============\nInputs / Outputs (i.e. Bindings) :\n";
  int n = (*ppTrtCudaEngine)->getNbBindings();
  for (int i = 0; i < n; ++i)
  {
    nvinfer1::Dims d = (*ppTrtCudaEngine)->getBindingDimensions(i);
    std::cout << i << " : " << (*ppTrtCudaEngine)->getBindingName(i) << " : dims=";
    for (int j = 0; j < d.nbDims; ++j)
    {
      std::cout << d.d[j];
      if (j < d.nbDims - 1)
      {
        std::cout << "x";
      }
    }
    std::cout << " , dtype=" << (int) (*ppTrtCudaEngine)->getBindingDataType(i) << " ";
    std::cout << ((*ppTrtCudaEngine)->bindingIsInput(i) ? "IN" : "OUT") << std::endl;
  }
  std::cout << "=============\n\n";

  return true;
}

#define ERROR_TRT_RUNTIME_INIT_FAILED         1
#define ERROR_TRT_ENGINE_LOAD_FAILED          2

struct Arguments_ExampleRVM
{
  std::string strTrtEngineFilepath;
  std::vector<std::string> strPictureFilepaths;

  bool parse( int argc, char* argv[], Logger& logger )
  {
    if( argc < 2 )
    {
      logger.log( nvinfer1::ILogger::Severity::kERROR, "Must specify path to RVM TRT engine as first argument. Exiting." );
      return false;
    }

    strTrtEngineFilepath = argv[1];

    for( int i = 2; i < argc; i++ )
    {
      strPictureFilepaths.push_back( argv[i] );
    }

    return true;
  }
};

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
  if( !LoadTRTEngineMatting( args.strTrtEngineFilepath, logger, pTrtRuntime, &pTrtCudaEngine ) )
  {
    return ERROR_TRT_ENGINE_LOAD_FAILED;
  }

  // Process Loop
  //

  // Cleanup
  //
  pTrtCudaEngine->destroy();
  pTrtRuntime->destroy();

  return 0;
}
