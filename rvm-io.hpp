#pragma once

#include "rvm-base.hpp"

class RVMRunner : public RVMBase
{
  public:
    RVMRunner( const std::vector<std::string>& args, nvinfer1::IExecutionContext* pTrtExecutionContext, Logger& logger );
    ~RVMRunner();

  private:

  // Templated
  //
  public:
    bool ProcessPictures( const std::vector<std::string>& args );

  private:
    bool InitStagingBuffers();
    bool FreeStagingBuffers();
    bool ConsumeInput( const char* szInPic = nullptr );
    bool ProduceOutput( const char* szOutPic = nullptr );

    void* m_bufStageSrc;
    void* m_bufStageFgr;
};
