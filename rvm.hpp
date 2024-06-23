#pragma once

// TODO: RVM later to be abstracted away to any matting model.
#include "rvm-base.hpp"
#include <cstdint>

template <typename TInOut>
class MattingRunner : public RVMBase
		    , TInOut
{
  public:
    MattingRunner( const std::vector<std::string>& args
	         , size_t picWidth, size_t picHeight
		 , nvinfer1::IExecutionContext* pTrtExecutionContext, Logger& logger );
    ~MattingRunner(){}

  public:
    bool ProcessPictures( const std::vector<std::string>& args );

};

template <typename TInOut>
MattingRunner<TInOut>::MattingRunner( const std::vector<std::string>& args
                                    , size_t picWidth, size_t picHeight
			            , nvinfer1::IExecutionContext* pTrtExecutionContext
			            , Logger& logger )
  : RVMBase( pTrtExecutionContext
           , picWidth
	   , picHeight
	   , logger )
  , TInOut( picWidth * picHeight * 3 * sizeof(uint8_t) 
          , picWidth * picHeight * 4 * sizeof(uint8_t)
	  , args
	  , logger
          )
{
}

template <typename TInOut>
bool MattingRunner<TInOut>::ProcessPictures( const std::vector<std::string> &args )
{
  while( TInOut::IsStillRunning() )
  {
    if( !TInOut::ConsumeNextInput( GetCuBufSrc(), GetCuStream() ) )
    {
      m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to consume input. Exiting." );
      return false;
    }

    if( !RunInference() )
    {
      m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to run TRT inference. Exiting." );
      return false;
    }

    if( !TInOut::ProduceNextOutput( GetCuBufFgr(), GetCuStream() ) )
    {
      m_logger.log( nvinfer1::ILogger::Severity::kERROR, "Failed to produce output. Exiting." );
      return false;
    }

    SwapRecurrents();
  }
  return true;
}
