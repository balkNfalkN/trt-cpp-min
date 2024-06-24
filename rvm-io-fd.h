#pragma once

#include <stddef.h>

#include "logger.hpp"
#include "rvm-iobase-host.h"

#include <string>
#include <vector>

class MattingIOFd : public IOBaseHost
{
  protected:
    MattingIOFd( size_t picSizeSrc, size_t picSizeFgr
	       , const std::vector<std::string>& args, Logger& logger );
    virtual ~MattingIOFd(){}

  protected:
    bool ConsumeNextInput( void* cuBufSrc, cudaStream_t cudaStream );
    bool ProduceNextOutput( void* cuBufFgr, cudaStream_t cudaStream ) ;

    // Skeleton of process loop to come.
    bool IsStillRunning();
};

