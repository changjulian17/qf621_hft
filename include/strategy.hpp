// strategy.hpp
#pragma once
#include "orderbook.hpp"

enum class Signal { BUY, SELL, HOLD };

class Strategy {
public:
    Strategy();
    Signal generate_signal(const OrderBook& ob);
    void update_position(Signal signal);

private:
    int current_position;
};
