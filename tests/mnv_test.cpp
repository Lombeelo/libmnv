#include <mnv/mnv.hpp>

#include <array>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <variant>
#include <vector>

TEST(linear_algebra_test, vector_dot_product_works)
{
    int product = mnv::internal::vectorDotProduct<int, 3>({1, 2, 3}, {4, 5, 6});
    ASSERT_EQ(product, 32);
}
TEST(linear_algebra_test, sum_of_squares_until_works)
{
    int product = mnv::internal::sumOfSquaresUntil<int, 3>({4, 5, 6}, 2);
    ASSERT_EQ(product, 41);
}
TEST(linear_algebra_test, sum_of_products_until_works)
{
    int product = mnv::internal::sumOfProductsUntil<int, 3>({1, 2, 3}, {4, 5, 6}, 2);
    ASSERT_EQ(product, 14);
}

TEST(linear_algebra_test, add_vectors_works)
{
    mnv::valueVector<int, 3> summ =
        mnv::internal::addVectors<int, 3>({1, 2, 3}, {4, 5, 6});
    ASSERT_THAT(summ, testing::ElementsAre(5, 7, 9));
}

TEST(linear_algebra_test, calculate_mean_vector_works)
{
    mnv::valueVector<float, 3> mean =
        mnv::calculateMeanVector<float, 3>(
            std::vector<mnv::valueVector<float, 3>>{{1, 2, 3}, {4, 5, 6}});
    ASSERT_THAT(mean, testing::ElementsAre(2.5f, 3.5f, 4.5f));
}

TEST(linear_algebra_test, is_odd_works)
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

TEST(linear_algebra_test, is_matrix_symmetric_works)
{
    EXPECT_TRUE(mnv::internal::isMatrixSymmetric(test_matrix));
    mnv::MatrixSq<double, 6> other = test_matrix;
    other[1][3] = 1;
    EXPECT_FALSE(mnv::internal::isMatrixSymmetric(other));
}

TEST(linear_algebra_test, choletsky_decomposition_works)
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

TEST(linear_algebra_test, minor_calculation_works)
{
    const std::array<double, 6> testing_matrix_minors =
        {5, 19, 10, 19, 16, 4};
    for (unsigned int i = 0; i < testing_matrix_minors.size(); i++)
    {
        double minor = mnv::internal::calculateMinor(test_matrix, i + 1);
        EXPECT_EQ(minor, testing_matrix_minors[i]);
    }
}

TEST(linear_algebra_test, matrix_definition_works)
{
    const mnv::MatrixSq<double, 3> pos_def{{{2, -1, 2},
                                            {-1, 1, -3},
                                            {2, -3, 11}}};
    const mnv::MatrixSq<double, 3> neg_def{{{-2, 1, 0},
                                            {1, -2, 0},
                                            {0, 0, -2}}};
    const mnv::MatrixSq<double, 3> undef{{{-2, 2, 0},
                                          {2, -2, 0},
                                          {0, 0, -8}}};
    auto def = mnv::internal::defineMatrix(pos_def);
    EXPECT_EQ(def,
              mnv::internal::MatrixDefinition::PositiveDefinite);

    def = mnv::internal::defineMatrix(neg_def);
    EXPECT_EQ(def,
              mnv::internal::MatrixDefinition::NegativeDefinite);

    def = mnv::internal::defineMatrix(undef);
    EXPECT_EQ(def,
              mnv::internal::MatrixDefinition::Undefinite);
}

TEST(linear_algebra_test, multiply_matrix_by_vector_works)
{
    const mnv::MatrixSq<double, 3> matrix{{{-2, 2, 0},
                                           {2, -2, 0},
                                           {0, 0, -8}}};
    const mnv::valueVector<double, 3> vector{{1, 2, 3}};
    const mnv::valueVector<double, 3> result =
        mnv::internal::multiplyMatrixByVector(matrix, vector);
    ASSERT_THAT(result, testing::ElementsAre(2, -2, -24));
}

TEST(statistic_calculations_test, calculate_cov_matrix_works)
{
    const std::vector<mnv::valueVector<double, 3>> stats = {
        {75, 10.5, 45},
        {65, 12.8, 65},
        {22, 7.3, 74},
        {15, 2.1, 76},
        {18, 9.2, 56}};

    const mnv::MatrixSq<double, 3> test_cov{{{655.6, 68.62, -189.6},
                                             {68.62, 13.0616, -25.716},
                                             {-189.6, -25.716, 133.36}}};
    const auto cov_calculated = mnv::calculateCovarianceMatrix(stats);
    for (size_t i = 0; i < cov_calculated.size(); i++)
    {
        for (size_t j = 0; j < cov_calculated.size(); j++)
        {
            EXPECT_NEAR(cov_calculated[i][j], test_cov[i][j], 0.001) << "i and j were " << i << " " << j << std::endl;
        }
    }
}

TEST(mnv_generator_test, build_works)
{
    const mnv::MatrixSq<double, 3> pos_def{{{2, -1, 2},
                                            {-1, 1, -3},
                                            {2, -3, 11}}};
    const mnv::MatrixSq<double, 3> neg_def{{{-2, 1, 0},
                                            {1, -2, 0},
                                            {0, 0, -2}}};
    const mnv::MatrixSq<double, 3> undef{{{-2, 2, 0},
                                          {2, -2, 0},
                                          {0, 0, -8}}};
    const mnv::MatrixSq<double, 3> assymetric{{{-2, 2, 1},
                                               {2, -2, 0},
                                               {0, 0, -8}}};
    const mnv::valueVector<double, 3> mean{{1, 1, 1}};

    auto gen_failed = mnv::MNVGenerator<double, 3>::build(neg_def, mean, 0);
    auto error = std::get<mnv::MNVGeneratorBuildError>(gen_failed);
    EXPECT_EQ(error.type, mnv::MNVGeneratorBuildError::type::CovarianceMatrixIsNotPositiveDefinite);

    gen_failed = mnv::MNVGenerator<double, 3>::build(undef, mean, 0);
    error = std::get<mnv::MNVGeneratorBuildError>(gen_failed);
    EXPECT_EQ(error.type, mnv::MNVGeneratorBuildError::type::CovarianceMatrixIsNotPositiveDefinite);

    gen_failed = mnv::MNVGenerator<double, 3>::build(assymetric, mean, 0);
    error = std::get<mnv::MNVGeneratorBuildError>(gen_failed);
    EXPECT_EQ(error.type, mnv::MNVGeneratorBuildError::type::CovarianceMatrixIsNotSymmetric);

    auto gen = mnv::MNVGenerator<double, 3>::build(pos_def, mean, 0);
    auto gen_ptr = std::get_if<mnv::MNVGenerator<double, 3>>(&gen);
    EXPECT_NE(gen_ptr, nullptr);
}

TEST(mnv_generator_test, covariance_is_right)
{
    const mnv::valueVector<double, 6> mean{{0, 2, 4, 8, 16, 32}};

    auto gen_packed = mnv::MNVGenerator<double, 6>::build(test_matrix, mean, 0);
    if (std::holds_alternative<mnv::MNVGeneratorBuildError>(gen_packed))
    {
        FAIL();
    }

    auto gen = std::get<mnv::MNVGenerator<double, 6>>(gen_packed);

    std::vector<mnv::valueVector<double, 6>> values{};
    size_t amountOfValues = 10000;
    values.reserve(amountOfValues);
    for (size_t i = 0; i < amountOfValues; i++)
    {
        values.push_back(gen.nextValue());
    }

    auto cov = mnv::calculateCovarianceMatrix(values);

    for (size_t i = 0; i < cov.size(); i++)
    {
        for (size_t j = 0; j < cov.size(); j++)
        {
            EXPECT_NEAR(cov[i][j], test_matrix[i][j], 0.1) << "i and j were " << i << " " << j << std::endl;
        }
    }

    auto mean_calcualated = mnv::calculateMeanVector(values);
    for (size_t j = 0; j < mean_calcualated.size(); j++)
    {
        EXPECT_NEAR(mean_calcualated[j], mean[j], 0.1);
    }
}