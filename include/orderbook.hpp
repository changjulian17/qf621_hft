// orderbook.hpp
#pragma once

struct OrderBook {
    double bid_price;
    int bid_size;
    double ask_price;
    int ask_size;

    double compute_obi() const;
};
