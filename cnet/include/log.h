/*****************************************************************************
 *                                  LOG
 *                           Log Utils for CNet.
 ****************************************************************************/

#ifndef CNET_LOG_H
#define CNET_LOG_H

#include <stdio.h>
#include <time.h>


#define LOG(LEVEL, FORMAT, ...) printf(\
        "%s-%s [%s]: " FORMAT,\
        __DATE__, __TIME__,\
        LEVEL,\
        __VA_ARGS__\
) 


#define log_info(FORMAT, ...)  LOG("INFO",    FORMAT, __VA_ARGS__)
#define log_debug(FORMAT, ...) LOG("DEBUG",   FORMAT, __VA_ARGS__)
#define log_warn(FORMAT, ...)  LOG("WARNING", FORMAT, __VA_ARGS__)
#define log_error(FORMAT, ...) LOG("ERROR",   FORMAT, __VA_ARGS__)

#endif /* CNET_LOG_H */
