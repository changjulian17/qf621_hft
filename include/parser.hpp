// parser.hpp
#pragma once
#include <string>
#include <vector>

struct Quote
{
    std::string datetime;
    double bid_price;
    int bid_size;
    double ask_price;
    int ask_size;
};

std::vector<Quote> parse_csv(const std::string &filepath);
