cmake_minimum_required(VERSION 3.10)

project(dtaidistance LANGUAGES C)

option(DTAIDISTANCE_WITH_OPENMP
       "Enable the parallel distance-matrix implementation with OpenMP" ON)

set(DTAIDISTANCE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/DTAIDistanceC")

add_library(dtaidistance SHARED
    "${DTAIDISTANCE_SOURCE_DIR}/dd_dtw.c"
    "${DTAIDISTANCE_SOURCE_DIR}/dd_dtw_openmp.c"
    "${DTAIDISTANCE_SOURCE_DIR}/dd_ed.c"
    "${DTAIDISTANCE_SOURCE_DIR}/dd_globals.c"
    "${DTAIDISTANCE_SOURCE_DIR}/dd_loco.c"
)
add_library(dtaidistance::dtaidistance ALIAS dtaidistance)

target_compile_features(dtaidistance PUBLIC c_std_99)
target_include_directories(dtaidistance
    PUBLIC
        $<BUILD_INTERFACE:${DTAIDISTANCE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/dtaidistance>
)

if(UNIX)
    target_link_libraries(dtaidistance PRIVATE m)
endif()

if(DTAIDISTANCE_WITH_OPENMP)
    find_package(OpenMP QUIET COMPONENTS C)
    if(OpenMP_C_FOUND)
        target_link_libraries(dtaidistance PRIVATE OpenMP::OpenMP_C)
    else()
        message(WARNING
            "OpenMP was not found; parallel DTAIDistanceC functions will be disabled")
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS dtaidistance
    EXPORT dtaidistanceTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(FILES
    "${DTAIDISTANCE_SOURCE_DIR}/dd_dtw.h"
    "${DTAIDISTANCE_SOURCE_DIR}/dd_dtw_openmp.h"
    "${DTAIDISTANCE_SOURCE_DIR}/dd_ed.h"
    "${DTAIDISTANCE_SOURCE_DIR}/dd_globals.h"
    "${DTAIDISTANCE_SOURCE_DIR}/dd_loco.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/dtaidistance"
)
install(EXPORT dtaidistanceTargets
    FILE dtaidistanceTargets.cmake
    NAMESPACE dtaidistance::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/dtaidistance"
)
