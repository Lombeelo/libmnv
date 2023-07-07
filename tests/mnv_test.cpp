#include <mnv/mnv.hpp>

#include <gtest/gtest.h>

TEST(add_test, add_1_2)
{
    EXPECT_EQ(mnv::add(1, 2), 3);
}