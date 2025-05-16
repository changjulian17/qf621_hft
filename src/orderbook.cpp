// orderbook.cpp
#include "orderbook.hpp"

double OrderBook::compute_obi() const
{
    int total_size = bid_size + ask_size;
    if (total_size == 0)
        return 0.0;
    return static_cast<double>(bid_size - ask_size) / total_size;
}
