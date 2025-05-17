// config.hpp
#pragma once
#include <string>

namespace config
{
    // Strategy parameters
    constexpr double OBI_THRESHOLD = 0.15;   // Reduced from 0.3 to be less aggressive
    constexpr double STOP_LOSS_PCT = 0.01;   // 1% stop loss
    constexpr double TAKE_PROFIT_PCT = 0.02; // 2% take profit
    constexpr int MAX_POSITION = 5;          // Reduced from 10 to limit risk
    constexpr int MIN_ORDER_SIZE = 1;
    constexpr double MAX_POSITION_VALUE = 50000.0; // Maximum position value in dollars
    constexpr double MIN_SPREAD_THRESHOLD = 0.01;  // Minimum spread to consider trading
    constexpr double TRANSACTION_COST = 0.0001;    // 1 basis point per trade
    constexpr double MAX_DAILY_LOSS_PCT = 0.02;    // 2% maximum daily loss

    // Time-based filters (24-hour format)
    constexpr int TRADING_START_HOUR = 9;
    constexpr int TRADING_START_MIN = 35; // Avoid first 5 minutes
    constexpr int TRADING_END_HOUR = 15;
    constexpr int TRADING_END_MIN = 55; // Stop 5 minutes before close

    // Avoid trading during typical high volatility periods
    constexpr bool AVOID_MARKET_OPEN = true;  // First 30 minutes
    constexpr bool AVOID_MARKET_CLOSE = true; // Last 30 minutes
    constexpr bool AVOID_LUNCH_HOUR = true;   // 12:00-13:00
}
