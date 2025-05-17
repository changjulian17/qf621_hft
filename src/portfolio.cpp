#include "portfolio.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

Portfolio::Portfolio(double starting_cash)
    : cash(starting_cash), position(0), average_entry_price(0.0), pnl_realized(0.0), pnl_unrealized(0.0) {}

void Portfolio::executeBuy(double price, int qty, const std::string &timestamp)
{
    if (std::isnan(price) || price <= 0.0)
    {
        return; // Invalid price
    }

    double trade_value = price * qty;
    cash -= trade_value;

    // Update average entry price
    if (position >= 0)
    {
        // Adding to long position or starting new long position
        average_entry_price = (position * average_entry_price + trade_value) / (position + qty);
    }
    else
    {
        // Covering short position
        if (position + qty >= 0)
        {
            // If covering moves us to flat or long, realize P&L
            pnl_realized += (-position) * (average_entry_price - price);
            average_entry_price = (qty + position > 0) ? price : 0.0;
        }
    }

    position += qty;
    trade_log.push_back({timestamp, "BUY", qty, price});
}

void Portfolio::executeSell(double price, int qty, const std::string &timestamp)
{
    if (std::isnan(price) || price <= 0.0)
    {
        return; // Invalid price
    }

    double trade_value = price * qty;
    cash += trade_value;

    // Update average entry price
    if (position <= 0)
    {
        // Adding to short position or starting new short position
        average_entry_price = (position * average_entry_price - trade_value) / (position - qty);
    }
    else
    {
        // Closing long position
        if (position - qty <= 0)
        {
            // If closing moves us to flat or short, realize P&L
            pnl_realized += position * (price - average_entry_price);
            average_entry_price = (position - qty < 0) ? price : 0.0;
        }
    }

    position -= qty;
    trade_log.push_back({timestamp, "SELL", qty, price});
}

void Portfolio::markToMarket(double last_price)
{
    if (std::isnan(last_price) || last_price <= 0.0)
    {
        return; // Invalid price
    }

    if (position != 0 && average_entry_price > 0.0)
    {
        pnl_unrealized = position * (last_price - average_entry_price);
    }
    else
    {
        pnl_unrealized = 0.0;
    }
}

void Portfolio::report() const
{
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Cash: $" << cash << "\n"
              << "Position: " << position << " shares\n"
              << "Average Entry Price: $" << (average_entry_price > 0 ? average_entry_price : 0.0) << "\n"
              << "Realized PnL: $" << pnl_realized << "\n"
              << "Unrealized PnL: $" << pnl_unrealized << "\n"
              << "Total Equity: $" << (cash + pnl_realized + pnl_unrealized) << "\n";
}
