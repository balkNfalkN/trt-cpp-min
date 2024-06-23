#pragma once

// TODO: RVM later to be abstracted away to any matting model.
#include "rvm-base.hpp"

class MattingRunner : public RVMBase
{
  public:
    MattingRunner( const std::vector<std::string>& args
	         , size_t picWidth, size_t picHeight
		 , nvinfer1::IExecutionContext* pTrtExecutionContext, Logger& logger );
    ~MattingRunner();

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
