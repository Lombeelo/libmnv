#include <mnv/mnv.hpp>

#include <array>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

TEST(linear_algera_test, vector_dot_product_works)
{
    int product = mnv::internal::vectorDotProduct<int, 3>({1, 2, 3}, {4, 5, 6});
    ASSERT_EQ(product, 32);
}
TEST(linear_algera_test, sum_of_squares_until_works)
{
    int product = mnv::internal::sumOfSquaresUntil<int, 3>({4, 5, 6}, 2);
    ASSERT_EQ(product, 41);
}
TEST(linear_algera_test, sum_of_products_until_works)
{
    int product = mnv::internal::sumOfProductsUntil<int, 3>({1, 2, 3}, {4, 5, 6}, 2);
    ASSERT_EQ(product, 14);
}

TEST(linear_algera_test, add_vectors_works)
{
    std::array<int, 3> summ =
        mnv::internal::addVectors<int, 3>({1, 2, 3}, {4, 5, 6});
    ASSERT_THAT(summ, testing::ElementsAre(5, 7, 9));
}

TEST(linear_algera_test, calculate_mean_vector_works)
{
    std::array<float, 3> mean =
        mnv::calculateMeanVector<float, 3>(
            std::vector<std::array<float, 3>>{{1, 2, 3}, {4, 5, 6}});
    ASSERT_THAT(mean, testing::ElementsAre(2.5f, 3.5f, 4.5f));
}

TEST(linear_algera_test, is_odd_works)
{
    EXPECT_TRUE(mnv::internal::isOdd(1));
    EXPECT_FALSE(mnv::internal::isOdd(2));
}

const mnv::MatrixSq<double, 6> test_matrix =
    {{{5, 4, 3, 2, 4, 2},
      {4, 7, 4, 2, 1, 4},
      {3, 4, 3, 1, 1, 1},
      {2, 2, 1, 3, 1, 2},
      {4, 1, 1, 1, 6, 2},
      {2, 4, 1, 2, 2, 6}}};

const mnv::MatrixSq<double, 6> test_matrix_decomposed =
    {{{2.236, 0, 0, 0, 0, 0},
      {1.789, 1.949, 0, 0, 0, 0},
      {1.342, 0.821, 0.725, 0, 0, 0},
      {0.894, 0.205, -0.508, 1.378, 0, 0},
      {1.789, -1.129, -0.653, -0.508, 0.918, 0},
      {0.894, 1.231, -1.669, 0.073, 0.803, 0.5}}};

TEST(linear_algera_test, is_matrix_symmetric_works)
{
    EXPECT_TRUE(mnv::internal::isMatrixSymmetric(test_matrix));
    mnv::MatrixSq<double, 6> other = test_matrix;
    other[1][3] = 1;
    EXPECT_FALSE(mnv::internal::isMatrixSymmetric(other));
}

TEST(linear_algera_test, choletsky_decomposition_works)
{
    auto decomposed = mnv::internal::doCholetskyDecomposition(test_matrix);
    for (size_t i = 0; i < decomposed.size(); i++)
    {
        for (size_t j = 0; j < decomposed.size(); j++)
        {
            EXPECT_NEAR(decomposed[i][j], test_matrix_decomposed[i][j], 0.001) << "i and j were " << i << " " << j << std::endl;
        }
    }
}