// strategy.hpp
#pragma once
#include "orderbook.hpp"
#include <string>
#include <map>

enum class Signal
{
    BUY,
    SELL,
    HOLD
};

class Strategy
{
public:
    Strategy();
    Signal generate_signal(const OrderBook &ob, const std::string &timestamp);
    void update_position(Signal signal);
    void update_position_value(double current_price);
    bool should_close_position(double current_price) const;
    void update_cash(double amount);
    double get_total_value() const { return cash + position_value; }

private:
    double cash;
    int current_position;
    double position_value;
    double entry_price;
    double last_trade_price;
    double daily_starting_value;
    std::string current_date;
    std::map<std::string, double> daily_pnl;
    bool check_risk_limits(const OrderBook &ob) const;
    bool is_trading_time(const std::string &timestamp) const;
    void update_daily_tracking(const std::string &timestamp, double current_value);
    bool check_daily_loss_limit(double current_value) const;
    std::pair<int, int> parse_time(const std::string &timestamp) const;
    bool is_end_of_day(const std::string &timestamp) const;
    Signal get_eod_signal(const OrderBook &ob) const;
    bool should_flatten_position() const;
};
