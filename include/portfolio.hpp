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
    double calculateSharpeRatio() const;
    void updateDailyReturns(const std::string &timestamp);
    double getDailyReturn() const;
    int getPosition() const { return position; }
    double getEquity() const { return cash + pnl_realized + pnl_unrealized; }

private:
    double cash;
    int position; // positive = long, negative = short
    double average_entry_price;
    double pnl_realized;
    double pnl_unrealized;
    std::vector<Trade> trade_log;
    std::vector<double> daily_returns; // Store daily returns for Sharpe ratio
    double previous_day_equity;
    std::string current_date;
};
