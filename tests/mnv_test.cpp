#include <mnv/mnv.hpp>

#include <array>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <variant>
#include <vector>

TEST(linearAlgebraTest, vectorDotProductWorks)
{
    int product = mnv::internal::vectorDotProduct<int, 3>({1, 2, 3}, {4, 5, 6});
    ASSERT_EQ(product, 32);
}
TEST(linearAlgebraTest, sumOfSquaresUntilWorks)
{
    int product = mnv::internal::sumOfSquaresUntil<int, 3>({4, 5, 6}, 2);
    ASSERT_EQ(product, 41);
}
TEST(linearAlgebraTest, sumOfProductsUntilWorks)
{
    int product = mnv::internal::sumOfProductsUntil<int, 3>({1, 2, 3}, {4, 5, 6}, 2);
    ASSERT_EQ(product, 14);
}

TEST(linearAlgebraTest, addVectorsWorks)
{
    mnv::valueVector<int, 3> summ =
        mnv::internal::addVectors<int, 3>({1, 2, 3}, {4, 5, 6});
    ASSERT_THAT(summ, testing::ElementsAre(5, 7, 9));
}

TEST(linearAlgebraTest, calculateMeanVectorWorks)
{
    mnv::valueVector<float, 3> mean =
        mnv::calculateMeanVector<float, 3>(
            std::vector<mnv::valueVector<float, 3>>{{1, 2, 3}, {4, 5, 6}});
    ASSERT_THAT(mean, testing::ElementsAre(2.5f, 3.5f, 4.5f));
}

TEST(linearAlgebraTest, isOddWorks)
{
    EXPECT_TRUE(mnv::internal::isOdd(1));
    EXPECT_FALSE(mnv::internal::isOdd(2));
}

const mnv::MatrixSq<double, 6> testMatrix =
    {{{5, 4, 3, 2, 4, 2},
      {4, 7, 4, 2, 1, 4},
      {3, 4, 3, 1, 1, 1},
      {2, 2, 1, 3, 1, 2},
      {4, 1, 1, 1, 6, 2},
      {2, 4, 1, 2, 2, 6}}};

const mnv::MatrixSq<double, 6> testMatrixDecomposed =
    {{{2.236, 0, 0, 0, 0, 0},
      {1.789, 1.949, 0, 0, 0, 0},
      {1.342, 0.821, 0.725, 0, 0, 0},
      {0.894, 0.205, -0.508, 1.378, 0, 0},
      {1.789, -1.129, -0.653, -0.508, 0.918, 0},
      {0.894, 1.231, -1.669, 0.073, 0.803, 0.5}}};

TEST(linearAlgebraTest, isMatrixSymmetricWorks)
{
    EXPECT_TRUE(mnv::internal::isMatrixSymmetric(testMatrix));
    mnv::MatrixSq<double, 6> other = testMatrix;
    other[1][3] = 1;
    EXPECT_FALSE(mnv::internal::isMatrixSymmetric(other));
}

TEST(linearAlgebraTest, choletskyDecompositionWorks)
{

    auto decomposed = mnv::internal::doCholetskyDecomposition(testMatrix);
    for (size_t i = 0; i < decomposed.size(); i++)
    {
        for (size_t j = 0; j < decomposed.size(); j++)
        {
            EXPECT_NEAR(decomposed[i][j], testMatrixDecomposed[i][j], 0.001) << "i and j were " << i << " " << j << std::endl;
        }
    }
}

TEST(linearAlgebraTest, minorCalculationWorks)
{
    const std::array<double, 6> testingMatrixMinors =
        {5, 19, 10, 19, 16, 4};
    for (unsigned int i = 0; i < testingMatrixMinors.size(); i++)
    {
        double minor = mnv::internal::calculateMinor(testMatrix, i + 1);
        EXPECT_EQ(minor, testingMatrixMinors[i]);
    }
}

TEST(linearAlgebraTest, matrixDefinitionWorks)
{
    const mnv::MatrixSq<double, 3> posDef{{{2, -1, 2},
                                           {-1, 1, -3},
                                           {2, -3, 11}}};
    const mnv::MatrixSq<double, 3> negDef{{{-2, 1, 0},
                                           {1, -2, 0},
                                           {0, 0, -2}}};
    const mnv::MatrixSq<double, 3> undef{{{-2, 2, 0},
                                          {2, -2, 0},
                                          {0, 0, -8}}};
    auto def = mnv::internal::defineMatrix(posDef);
    EXPECT_EQ(def,
              mnv::internal::MatrixDefinition::PositiveDefinite);

    def = mnv::internal::defineMatrix(negDef);
    EXPECT_EQ(def,
              mnv::internal::MatrixDefinition::NegativeDefinite);

    def = mnv::internal::defineMatrix(undef);
    EXPECT_EQ(def,
              mnv::internal::MatrixDefinition::Undefinite);
}

TEST(linearAlgebraTest, multiplyMatrixByVectorWorks)
{
    const mnv::MatrixSq<double, 3> matrix{{{-2, 2, 0},
                                           {2, -2, 0},
                                           {0, 0, -8}}};
    const mnv::valueVector<double, 3> vector{{1, 2, 3}};
    const mnv::valueVector<double, 3> result =
        mnv::internal::multiplyMatrixByVector(matrix, vector);
    ASSERT_THAT(result, testing::ElementsAre(2, -2, -24));
}

TEST(statisticCalculationsTest, calculateCovMatrixWorks)
{
    const std::vector<mnv::valueVector<double, 3>> stats = {
        {75, 10.5, 45},
        {65, 12.8, 65},
        {22, 7.3, 74},
        {15, 2.1, 76},
        {18, 9.2, 56}};

    const mnv::MatrixSq<double, 3> testCov{{{655.6, 68.62, -189.6},
                                            {68.62, 13.0616, -25.716},
                                            {-189.6, -25.716, 133.36}}};
    const auto covCalculated = mnv::calculateCovarianceMatrix(stats);
    for (size_t i = 0; i < covCalculated.size(); i++)
    {
        for (size_t j = 0; j < covCalculated.size(); j++)
        {
            EXPECT_NEAR(covCalculated[i][j], testCov[i][j], 0.001) << "i and j were " << i << " " << j << std::endl;
        }
    }
}

TEST(mnvGeneratorTest, buildWorks)
{
    const mnv::MatrixSq<double, 3> posDef{{{2, -1, 2},
                                           {-1, 1, -3},
                                           {2, -3, 11}}};
    const mnv::MatrixSq<double, 3> negDef{{{-2, 1, 0},
                                           {1, -2, 0},
                                           {0, 0, -2}}};
    const mnv::MatrixSq<double, 3> undef{{{-2, 2, 0},
                                          {2, -2, 0},
                                          {0, 0, -8}}};
    const mnv::MatrixSq<double, 3> assymetric{{{-2, 2, 1},
                                               {2, -2, 0},
                                               {0, 0, -8}}};
    const mnv::valueVector<double, 3> mean{{1, 1, 1}};

    auto genFailed = mnv::MNVGenerator<double, 3>::build(negDef, mean, 0);
    auto error = std::get<mnv::MNVGeneratorBuildError>(genFailed);
    EXPECT_EQ(error.type, mnv::MNVGeneratorBuildError::type::CovarianceMatrixIsNotPositiveDefinite);

    genFailed = mnv::MNVGenerator<double, 3>::build(undef, mean, 0);
    error = std::get<mnv::MNVGeneratorBuildError>(genFailed);
    EXPECT_EQ(error.type, mnv::MNVGeneratorBuildError::type::CovarianceMatrixIsNotPositiveDefinite);

    genFailed = mnv::MNVGenerator<double, 3>::build(assymetric, mean, 0);
    error = std::get<mnv::MNVGeneratorBuildError>(genFailed);
    EXPECT_EQ(error.type, mnv::MNVGeneratorBuildError::type::CovarianceMatrixIsNotSymmetric);

    auto gen = mnv::MNVGenerator<double, 3>::build(posDef, mean, 0);
    auto genPtr = std::get_if<mnv::MNVGenerator<double, 3>>(&gen);
    EXPECT_NE(genPtr, nullptr);
}

TEST(mnvGeneratorTest, build2Works)
{
    const std::vector<mnv::valueVector<double, 3>> stats = {
        {75, 10.5, 45},
        {65, 12.8, 65},
        {22, 7.3, 74},
        {15, 2.1, 76},
        {18, 9.2, 56}};
    const std::vector<mnv::valueVector<double, 3>> statsNotEnoughInfo = {
        {75, 10.5, 45},
        {65, 12.8, 65}};

    auto genFailed = mnv::MNVGenerator<double, 3>::build(statsNotEnoughInfo, 0);
    auto error = std::get<mnv::MNVGeneratorBuildError>(genFailed);
    EXPECT_EQ(error.type, mnv::MNVGeneratorBuildError::type::CovarianceMatrixIsNotPositiveDefinite);

    auto gen = mnv::MNVGenerator<double, 3>::build(stats, 0);
    auto genPtr = std::get_if<mnv::MNVGenerator<double, 3>>(&gen);
    EXPECT_NE(genPtr, nullptr);
}

TEST(mnvGeneratorTest, covarianceIsRight)
{
    const mnv::valueVector<double, 6> mean{{0, 2, 4, 8, 16, 32}};

    auto genPacked = mnv::MNVGenerator<double, 6>::build(testMatrix, mean, 0);
    if (std::holds_alternative<mnv::MNVGeneratorBuildError>(genPacked))
    {
        FAIL();
    }

    auto gen = std::get<mnv::MNVGenerator<double, 6>>(genPacked);

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
            EXPECT_NEAR(cov[i][j], testMatrix[i][j], 0.1) << "i and j were " << i << " " << j << std::endl;
        }
    }

    auto meanCalcualated = mnv::calculateMeanVector(values);
    for (size_t j = 0; j < meanCalcualated.size(); j++)
    {
        EXPECT_NEAR(meanCalcualated[j], mean[j], 0.1);
    }
}