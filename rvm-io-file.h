#pragma once

#include <stddef.h>

#include "logger.hpp"
#include "rvm-iobase-host.h"

#include <string>
#include <vector>

class MattingIOFile : public IOBaseHost
{
  protected:
    MattingIOFile( size_t picSizeSrc, size_t picSizeFgr
	         , const std::vector<std::string>& args, Logger& logger );
    virtual ~MattingIOFile();

  protected:
    bool ConsumeNextInput( void* cuBufSrc, cudaStream_t cudaStream );
    bool ProduceNextOutput( void* cuBufFgr, cudaStream_t cudaStream ) ;

    // Skeleton of process loop to come.
    bool IsStillRunning();

    std::vector<std::string> m_inFilepaths;
    std::vector<std::string>::const_iterator m_itInFilepath;

    Logger& m_logger;
};
