#include "logger.hpp"

class IOBaseHost
{
  protected:
    IOBaseHost( size_t picSizeSrc, size_t picSizeFgr, Logger& logger );
    bool InitStagingBuffers( Logger& logger );
    bool FreeStagingBuffers( Logger& logger );

  protected:
    void* m_bufStageSrc;
    void* m_bufStageFgr;
    size_t m_picSizeSrc;
    size_t m_picSizeFgr;
};
