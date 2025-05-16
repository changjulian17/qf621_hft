// strategy.cpp
#include "strategy.hpp"
#include "config.hpp"
#include <cmath>
#include <sstream>

Strategy::Strategy()
    : cash(100000.0) // Initial cash
      ,
      current_position(0), position_value(0.0), entry_price(0.0), last_trade_price(0.0), daily_starting_value(100000.0) // Initial portfolio value
      ,
      current_date("")
{
}

std::pair<int, int> Strategy::parse_time(const std::string &timestamp) const
{
    // Expected format: "YYYY-MM-DD HH:MM:SS"
    size_t space_pos = timestamp.find(' ');
    if (space_pos == std::string::npos)
        return {0, 0};

    std::string time = timestamp.substr(space_pos + 1);
    int hour = std::stoi(time.substr(0, 2));
    int minute = std::stoi(time.substr(3, 2));

    return {hour, minute};
}

bool Strategy::is_trading_time(const std::string &timestamp) const
{
    auto [hour, minute] = parse_time(timestamp);

    // Check if within trading hours
    if (hour < config::TRADING_START_HOUR ||
        (hour == config::TRADING_START_HOUR && minute < config::TRADING_START_MIN) ||
        hour > config::TRADING_END_HOUR ||
        (hour == config::TRADING_END_HOUR && minute > config::TRADING_END_MIN))
    {
        return false;
    }

    // Avoid market open period
    if (config::AVOID_MARKET_OPEN &&
        hour == config::TRADING_START_HOUR &&
        minute < config::TRADING_START_MIN + 30)
    {
        return false;
    }

    // Avoid market close period
    if (config::AVOID_MARKET_CLOSE &&
        hour == config::TRADING_END_HOUR &&
        minute > config::TRADING_END_MIN - 30)
    {
        return false;
    }

    // Avoid lunch hour
    if (config::AVOID_LUNCH_HOUR && hour == 12)
    {
        return false;
    }

    return true;
}

void Strategy::update_daily_tracking(const std::string &timestamp, double current_value)
{
    std::string date = timestamp.substr(0, 10); // Extract YYYY-MM-DD

    if (date != current_date)
    {
        // New trading day
        current_date = date;
        daily_starting_value = current_value;
        daily_pnl[date] = 0.0;
    }

    // Update daily P&L
    daily_pnl[date] = current_value - daily_starting_value;
}

bool Strategy::check_daily_loss_limit(double current_value) const
{
    if (current_date.empty())
        return true;

    double daily_loss_pct = (current_value - daily_starting_value) / daily_starting_value;
    return daily_loss_pct >= -config::MAX_DAILY_LOSS_PCT;
}

bool Strategy::check_risk_limits(const OrderBook &ob) const
{
    // Check spread
    double spread = ob.ask_price - ob.bid_price;
    double spread_pct = spread / ob.bid_price;
    if (spread_pct > config::MIN_SPREAD_THRESHOLD)
    {
        return false;
    }

    // Check position value limits
    double potential_position_value = std::abs(current_position + config::MIN_ORDER_SIZE) * ob.ask_price;
    if (potential_position_value > config::MAX_POSITION_VALUE)
    {
        return false;
    }

    return true;
}

void Strategy::update_cash(double amount)
{
    cash += amount;
}

Signal Strategy::generate_signal(const OrderBook &ob, const std::string &timestamp)
{
    if (!is_trading_time(timestamp))
    {
        return Signal::HOLD;
    }

    if (!check_risk_limits(ob))
    {
        return Signal::HOLD;
    }

    double mid_price = (ob.ask_price + ob.bid_price) / 2.0;
    double current_value = get_total_value();

    update_daily_tracking(timestamp, current_value);

    if (!check_daily_loss_limit(current_value))
    {
        // If we've hit our daily loss limit, close any open positions
        if (current_position > 0)
            return Signal::SELL;
        if (current_position < 0)
            return Signal::BUY;
        return Signal::HOLD;
    }

    // Update position value
    update_position_value(mid_price);

    // Check if we need to close position based on stop loss or take profit
    if (should_close_position(mid_price))
    {
        return (current_position > 0) ? Signal::SELL : Signal::BUY;
    }

    double obi = ob.compute_obi();

    if (obi > config::OBI_THRESHOLD && current_position < config::MAX_POSITION)
    {
        return Signal::BUY;
    }
    else if (obi < -config::OBI_THRESHOLD && current_position > -config::MAX_POSITION)
    {
        return Signal::SELL;
    }

    return Signal::HOLD;
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

void Strategy::update_position_value(double current_price)
{
    position_value = current_position * current_price;
    if (current_position != 0 && entry_price == 0.0)
    {
        entry_price = current_price;
    }
    last_trade_price = current_price;
}

bool Strategy::should_close_position(double current_price) const
{
    if (current_position == 0 || entry_price == 0.0)
    {
        return false;
    }

    double pnl_pct = (current_price - entry_price) / entry_price;

    if (current_position > 0)
    {
        // Long position
        return (pnl_pct <= -config::STOP_LOSS_PCT) || (pnl_pct >= config::TAKE_PROFIT_PCT);
    }
    else
    {
        // Short position
        return (pnl_pct >= config::STOP_LOSS_PCT) || (pnl_pct <= -config::TAKE_PROFIT_PCT);
    }
}
