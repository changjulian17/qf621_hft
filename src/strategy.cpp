// strategy.cpp
#include "strategy.hpp"
#include "config.hpp"

Strategy::Strategy() : current_position(0) {}

Signal Strategy::generate_signal(const OrderBook &ob)
{
    double obi = ob.compute_obi();

    if (obi > config::OBI_THRESHOLD && current_position < config::MAX_POSITION)
    {
        return Signal::BUY;
    }
    else if (obi < -config::OBI_THRESHOLD && current_position > -config::MAX_POSITION)
    {
        return Signal::SELL;
    }
    else
    {
        return Signal::HOLD;
    }
}

void Strategy::update_position(Signal signal)
{
    if (signal == Signal::BUY)
    {
        current_position += config::MIN_ORDER_SIZE;
    }
    else if (signal == Signal::SELL)
    {
        current_position -= config::MIN_ORDER_SIZE;
    }
}
