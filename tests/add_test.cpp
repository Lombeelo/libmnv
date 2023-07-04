#include <mnv/mnv.hpp>

#include <gtest/gtest.h>

TEST(add_test, add_1_1)
{
    EXPECT_EQ(mnv::add(1, 1), 2);
}