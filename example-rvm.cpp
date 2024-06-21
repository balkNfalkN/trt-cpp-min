
#include <NvInfer.h>
#include <NvInferRuntimeBase.h>

#include <cuda_runtime.h>

#include <string>
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


int main( int argc, char* argv[] )
{
  Logger logger;
  logger.log( nvinfer1::ILogger::Severity::kINFO, "C++ TensorRT RVM Inference example" );

  for( int i = 0; i < argc; i++ )
  {
    std::cout << "arg[" << i << "] = " << argv[i] << std::endl;
  }

  return 0;
}
