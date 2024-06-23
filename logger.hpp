#pragma once

#include <NvInferRuntimeBase.h>
#include <iostream>
#include <string>

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
