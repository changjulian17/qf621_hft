#include "parser.hpp"
#include <fstream>
#include <sstream>

std::vector<Quote> parse_csv(const std::string &filepath)
{
    std::vector<Quote> quotes;
    std::ifstream file(filepath);
    std::string line;

    // Check if file exists
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::string token;
        Quote q;

        std::getline(ss, token, ','); // DATE
        q.datetime = token;
        std::getline(ss, token, ','); // TIME_M
        q.datetime += " " + token;

        std::getline(ss, token, ','); // EX
        std::getline(ss, token, ','); // BID
        q.bid_price = std::stod(token);
        std::getline(ss, token, ','); // BIDSIZ
        q.bid_size = std::stoi(token);
        std::getline(ss, token, ','); // ASK
        q.ask_price = std::stod(token);
        std::getline(ss, token, ','); // ASKSIZ
        q.ask_size = std::stoi(token);

        // Skip QU_COND, QU_SEQNUM, NATBBO_IND, QU_CANCEL
        for (int i = 0; i < 4; ++i)
            std::getline(ss, token, ',');

        std::getline(ss, token, ','); // QU_SOURCE
        std::getline(ss, token, ','); // SYM_ROOT
        if (token != "AAPL")
            continue; // Filter only AAPL

        // Skip SYM_SUFFIX if present

        quotes.push_back(q);
    }

    return quotes;
}
