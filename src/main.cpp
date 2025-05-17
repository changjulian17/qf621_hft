#include "parser.hpp"
#include "orderbook.hpp"
#include "strategy.hpp"
#include "portfolio.hpp"
#include "config.hpp"
#include <iostream>

int main()
{
    auto quotes = parse_csv("/home/omegashenr01n/Desktop/qf621_hft/3_stock_trading_hrs.csv");
    Strategy strategy;
    Portfolio pf(100000); // starting cash: $100,000

    for (const auto &q : quotes)
    {
        OrderBook ob{q.bid_price, q.bid_size, q.ask_price, q.ask_size};
        Signal signal = strategy.generate_signal(ob, q.datetime);

        // Apply transaction costs to prices
        double effective_ask = q.ask_price * (1.0 + config::TRANSACTION_COST);
        double effective_bid = q.bid_price * (1.0 - config::TRANSACTION_COST);

        if (signal == Signal::BUY)
        {
            pf.executeBuy(effective_ask, config::MIN_ORDER_SIZE, q.datetime);
            strategy.update_position(signal);
            strategy.update_cash(-effective_ask * config::MIN_ORDER_SIZE);
        }
        else if (signal == Signal::SELL)
        {
            pf.executeSell(effective_bid, config::MIN_ORDER_SIZE, q.datetime);
            strategy.update_position(signal);
            strategy.update_cash(effective_bid * config::MIN_ORDER_SIZE);
        }

        // Mark to market at mid price to avoid bid-ask bounce in P&L calculation
        double mid_price = (q.ask_price + q.bid_price) / 2.0;
        pf.markToMarket(mid_price);
        strategy.update_position_value(mid_price);

        // Mark portfolio to mid-price

        // std::cout << q.datetime << " | OBI: " << ob.compute_obi()
        //           << " | Signal: " << (signal == Signal::BUY ? "BUY" : signal == Signal::SELL ? "SELL"
        //                                                                                       : "HOLD")
        //           << std::endl;
    }

    std::cout << "\nFinal Portfolio Report:\n";
    pf.report();
    std::cout << "Strategy Total Value: $" << strategy.get_total_value() << "\n";

    return 0;
}
