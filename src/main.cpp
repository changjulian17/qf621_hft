#include "parser.hpp"
#include "orderbook.hpp"
#include "strategy.hpp"
#include "portfolio.hpp"
#include <iostream>

int main()
{
    auto quotes = parse_csv("/home/omegashenr01n/Desktop/qf621_hft/3_stock_trading_hrs.csv");
    Strategy strategy;
    Portfolio pf(100000); // starting cash: $100,000

    for (const auto &q : quotes)
    {
        OrderBook ob{q.bid_price, q.bid_size, q.ask_price, q.ask_size};
        Signal signal = strategy.generate_signal(ob);
        strategy.update_position(signal);

        if (signal == Signal::BUY)
        {
            pf.executeBuy(q.ask_price, 1, q.datetime);
            pf.markToMarket(q.ask_price);
        }
        else if (signal == Signal::SELL)
        {
            pf.executeSell(q.bid_price, 1, q.datetime);
            pf.markToMarket(q.bid_price);
        }

        // Mark portfolio to mid-price

        // std::cout << q.datetime << " | OBI: " << ob.compute_obi()
        //           << " | Signal: " << (signal == Signal::BUY ? "BUY" : signal == Signal::SELL ? "SELL"
        //                                                                                       : "HOLD")
        //           << std::endl;
    }

    std::cout << "\nFinal Portfolio Report:\n";
    pf.report();

    return 0;
}
