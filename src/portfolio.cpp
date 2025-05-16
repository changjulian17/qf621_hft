#include "portfolio.hpp"
#include <iostream>
#include <iomanip>

Portfolio::Portfolio(double starting_cash)
    : cash(starting_cash), position(0), average_entry_price(0.0), pnl_realized(0.0), pnl_unrealized(0.0) {}

void Portfolio::executeBuy(double price, int qty, const std::string &timestamp)
{
    cash -= price * qty;
    average_entry_price = (position * average_entry_price + qty * price) / (position + qty);
    position += qty;
    trade_log.push_back({timestamp, "BUY", qty, price});
}

void Portfolio::executeSell(double price, int qty, const std::string &timestamp)
{
    cash += price * qty;
    if (position > 0)
    {
        pnl_realized += (price - average_entry_price) * std::min(position, qty);
    }
    position -= qty;
    if (position == 0)
        average_entry_price = 0.0;
    trade_log.push_back({timestamp, "SELL", qty, price});
}

void Portfolio::markToMarket(double last_price)
{
    pnl_unrealized = position * (last_price - average_entry_price);
}

void Portfolio::report() const
{
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Cash: $" << cash << "\n"
              << "Position: " << position << " shares\n"
              << "Realized PnL: $" << pnl_realized << "\n"
              << "Unrealized PnL: $" << pnl_unrealized << "\n"
              << "Total Equity: $" << (cash + pnl_realized + pnl_unrealized) << "\n";
}
