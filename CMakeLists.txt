cmake_minimum_required(VERSION 3.16)
project(OBITradingStrategy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define sources and headers
set(SOURCES
    src/main.cpp
    src/parser.cpp
    src/orderbook.cpp
    src/strategy.cpp
    src/portfolio.cpp
)

set(HEADERS
    include/config.hpp
    include/parser.hpp
    include/orderbook.hpp
    include/strategy.hpp
    include/portfolio.hpp
)

# Include headers
include_directories(include)

# Define the executable
add_executable(obi_trading ${SOURCES} ${HEADERS})

# Optional: enable warnings and optimizations
target_compile_options(obi_trading PRIVATE -Wall -Wextra -O2)
