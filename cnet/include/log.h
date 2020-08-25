/*****************************************************************************
 *                                  LOG
 *                           Log Utils for CNet.
 ****************************************************************************/

#ifndef CNET_LOG_H
#define CNET_LOG_H

#include <stdio.h>
#include <time.h>
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"


#define LOG(LEVEL, FMT, ...) printf(\
        "%s-%s [%s]: " FMT "\n",\
        __DATE__, __TIME__,\
        LEVEL, \
        ##__VA_ARGS__\
) 


#define log_info(FORMAT, ...)  LOG("INFO",    FORMAT, ##__VA_ARGS__)
#define log_debug(FORMAT, ...) LOG("DEBUG",   FORMAT, ##__VA_ARGS__)
#define log_warn(FORMAT, ...)  LOG("WARNING", FORMAT, ##__VA_ARGS__)
#define log_error(FORMAT, ...) LOG("ERROR",   FORMAT, ##__VA_ARGS__)

#endif /* CNET_LOG_H */
