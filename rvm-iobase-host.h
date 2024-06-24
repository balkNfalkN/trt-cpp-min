#include "logger.hpp"

class IOBaseHost
{
  protected:
    IOBaseHost( size_t picSizeSrc, size_t picSizeFgr, Logger& logger );
    virtual ~IOBaseHost();
    bool InitStagingBuffers();
    bool FreeStagingBuffers();

  protected:
    void* m_bufStageSrc;
    void* m_bufStageFgr;
    size_t m_picSizeSrc;
    size_t m_picSizeFgr;

  protected:
    Logger& m_logger;
};
