#include <mnv/mnv.hpp>

#include <iostream>

int main(int, char *[])
{
    auto sum = mnv::add(1, 1);
    std::cout << sum << std::endl;
    return 0;
}