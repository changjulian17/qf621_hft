#pragma once
#include <vector>
#include <string>

struct Trade
{
    std::string time;
    std::string side;
    int quantity;
    double price;
};

class Portfolio
{
public:
    Portfolio(double starting_cash);

    void executeBuy(double price, int qty, const std::string &timestamp);
    void executeSell(double price, int qty, const std::string &timestamp);
    void markToMarket(double last_price);
    void report() const;

private:
    double cash;
    int position; // positive = long, negative = short
    double average_entry_price;
    double pnl_realized;
    double pnl_unrealized;

    std::vector<Trade> trade_log;
};
