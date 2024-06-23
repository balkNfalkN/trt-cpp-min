#include <stddef.h>

#include "logger.hpp"
#include <string>
#include <vector>

class MattingIOFile
{
  protected:
    MattingIOFile( size_t picSizeSrc, size_t picSizeFgr, const std::vector<std::string>& args, Logger& logger );
    virtual ~MattingIOFile();

  protected:
    bool InitStagingBuffers();

  protected:
    bool FreeStagingBuffers();
    inline bool ConsumeInput( const char* szInPic = nullptr );
    inline bool ProduceOutput( const char* szOutPic = nullptr );

    // Skeleton of process loop to come.
    bool KeepRunning();
    const char* CurIteration();

  private:
    void* m_bufStageSrc;
    void* m_bufStageFgr;
    size_t m_picSizeSrc;
    size_t m_picSizeFgr;

    std::vector<std::string> m_inFilepaths;
    std::vector<std::string>::const_iterator m_itInFilepath;

    Logger& m_logger;
};
