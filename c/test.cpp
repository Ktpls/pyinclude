#include "util.h"

void test_pipe()
{
    pipe<int>::range(10)
        .sorted([](auto a, auto b)
                { return a > b; })
        .groupby<int>([](auto i)
                      { return i % 3; })
        .foreach ([](auto i)
                  {
                std::cout << i.first << std::endl;
                pipe<int>::of(i.second).foreach([](auto j)
                                                {std::cout << "\t" << j << std::endl;}); })
        .flatmap<int>([](auto i)
                      { return pipe<int>(i.second); })
        .foreach ([](auto i)
                  { std::cout << i << std::endl; });
}