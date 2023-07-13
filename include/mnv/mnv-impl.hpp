#ifndef MNV_IMPL_HPP
#define MNV_IMPL_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <variant>
#include <vector>

namespace mnv
{
    template <typename T, size_t Dim>
    using valueVector = std::array<T, Dim>;

    template <typename T, size_t Dim>
    using MatrixSq = valueVector<valueVector<T, Dim>, Dim>;

    namespace internal
    {
        template <typename T, size_t Dim>
        T vectorDotProduct(valueVector<T, Dim> const &first, valueVector<T, Dim> const &second)
        {
            T result{};

            for (size_t i = 0; i < first.size(); i++)
            {
                result += first[i] * second[i];
            }

            return result;
        }

        template <typename T, size_t Dim>
        valueVector<T, Dim> addVectors(valueVector<T, Dim> const &first, valueVector<T, Dim> const &second)
        {
            valueVector<T, Dim> result{};

            for (size_t i = 0; i < first.size(); i++)
            {
                result[i] = first[i] + second[i];
            }

            return result;
        }

        inline bool isOdd(size_t number)
        {
            return static_cast<bool>(number & 1);
        }

        template <typename T, size_t Dim>
        T calculateMinor(MatrixSq<T, Dim> const &matrix, unsigned int minor_order)
        {
            if (minor_order == 1)
            {
                return matrix[0][0];
            }

            if (minor_order == 2)
            {
                return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1];
            }

            T sum = 0;

            // разложение Лапласа, рекурсивное
            size_t column = 0;
            for (size_t row = 0; row < minor_order; row++)
            {
                MatrixSq<T, Dim> minorMatrix;
                for (size_t i = 0; i < minor_order - 1; i++)
                {
                    for (size_t j = 0; j < minor_order - 1; j++)
                    {
                        minorMatrix[i][j] = matrix[i + (i >= row)][j + (column + 1)];
                    }
                }
                T otherMinor = matrix[row][column] * calculateMinor(minorMatrix, minor_order - 1);
                sum = isOdd(row)
                          ? sum - otherMinor
                          : sum + otherMinor;
            }

            return sum;
        }

        enum class MatrixDefinition
        {
            ZeroDefinite,
            PositiveDefinite,
            NegativeDefinite,
            Undefinite
        };

        template <typename T, size_t Dim>
        MatrixDefinition defineMatrix(MatrixSq<T, Dim> const &matrix)
        {
            MatrixDefinition result = MatrixDefinition::ZeroDefinite;
            bool lastNegative = false;
            for (size_t i = 1; i < matrix.size() + 1; i++)
            {
                T minor = calculateMinor(matrix, (unsigned int)i);
                if (minor > 0)
                {
                    switch (result)
                    {
                    case MatrixDefinition::ZeroDefinite:
                        result = MatrixDefinition::PositiveDefinite;
                        break;

                    case MatrixDefinition::NegativeDefinite:
                        if (lastNegative)
                        {
                            lastNegative = false;
                            break;
                        }

                        result = MatrixDefinition::Undefinite;
                        break;

                    default:
                        break;
                    }
                }

                if (minor < 0)
                {
                    switch (result)
                    {
                    case MatrixDefinition::ZeroDefinite:
                        result = MatrixDefinition::NegativeDefinite;
                        lastNegative = true;
                        break;

                    case MatrixDefinition::NegativeDefinite:
                        if (!lastNegative)
                        {
                            lastNegative = true;
                            break;
                        }

                        result = MatrixDefinition::Undefinite;
                        break;

                    case MatrixDefinition::PositiveDefinite:
                        result = MatrixDefinition::Undefinite;
                        break;

                    default:
                        break;
                    }
                }

                if (minor == 0)
                {
                    return MatrixDefinition::Undefinite;
                }
            }

            return result;
        }

        template <typename T, size_t Dim>
        T sumOfProductsUntil(valueVector<T, Dim> const &vectorA, valueVector<T, Dim> const &vectorB, size_t idx)
        {
            T sum = 0;

            for (size_t i = 0; i < idx; i++)
            {
                sum += vectorA[i] * vectorB[i];
            }

            return sum;
        }

        template <typename T, size_t Dim>
        T sumOfSquaresUntil(valueVector<T, Dim> const &vector, size_t idx)
        {
            return sumOfProductsUntil(vector, vector, idx);
        }

        template <typename MatrixType>
        inline bool isMatrixSymmetric(MatrixType const &matrix)
        {
            for (size_t i = 0; i < matrix.size() - 1; i++)
            {
                for (size_t j = i + 1; j < matrix.size(); j++)
                {
                    if (matrix[i][j] != matrix[j][i])
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        template <typename T, size_t Dim>
        MatrixSq<T, Dim> doCholetskyDecomposition(MatrixSq<T, Dim> const &matrix)
        {
            MatrixSq<T, Dim> result{};
            // 3. Decomposition itself
            result[0][0] = std::sqrt(matrix[0][0]);

            for (size_t i = 1; i < matrix.size(); i++)
            {
                result[i][0] = matrix[i][0] / result[0][0];
            }

            for (size_t j = 1; j < matrix.size(); j++)
            {
                for (size_t i = j; i < matrix.size(); i++)
                {

                    if (i == j)
                    {
                        result[i][j] = std::sqrt(matrix[i][i] - sumOfSquaresUntil(result[i], j));
                    }
                    else
                    {
                        result[i][j] = (matrix[i][j] - sumOfProductsUntil(result[i], result[j], j)) / result[j][j];
                    }
                }
            }

            return result;
        }

        template <typename T, size_t Dim>
        valueVector<T, Dim> multiplyMatrixByVector(MatrixSq<T, Dim> const &matrix, valueVector<T, Dim> const &vector)
        {
            valueVector<T, Dim> result{};
            for (size_t i = 0; i < result.size(); i++)
            {
                result[i] = vectorDotProduct(matrix[i], vector);
            }

            return result;
        }
    } // namespace internal

    template <typename T, size_t Dim>
    valueVector<T, Dim> MNVGenerator<T, Dim>::nextValue()
    {
        valueVector<T, Dim> randomStandardNormalVector{};

        for (size_t i = 0; i < randomStandardNormalVector.size(); i++)
        {
            randomStandardNormalVector[i] = distribution(m_generator);
        }

        valueVector<T, Dim> multipliedVector =
            internal::multiplyMatrixByVector(m_decomposedCovariance, randomStandardNormalVector);
        return internal::addVectors(multipliedVector, m_mean);
    }

    template <typename T, size_t Dim>
    std::variant<MNVGenerator<T, Dim>, MNVGeneratorBuildError>
    MNVGenerator<T, Dim>::build(
        MatrixSq<T, Dim> const &covariance,
        valueVector<T, Dim> const &mean,
        size_t seed)
    {
        using ::mnv::internal::MatrixDefinition;

        // 1. Check for symmetric matrix

        if (!internal::isMatrixSymmetric(covariance))
        {
            // error: ABSOLUTELY wrong matrix
            return MNVGeneratorBuildError{
                MNVGeneratorBuildError::type::CovarianceMatrixIsNotSymmetric,
                ERRMSG("The covariance matrix provided is not symmetric. It's totally unsuitable to use here. Please provide a valid covariance matrix.\n")};
        }

        // 2. Check for positive-definite matrix

        MatrixDefinition def = internal::defineMatrix(covariance);
        switch (def)
        {
        case MatrixDefinition::NegativeDefinite: // error: how tf you did that (wrong matrix)?
        case MatrixDefinition::Undefinite:       // error: how tf you did that (wrong matrix)?
            return MNVGeneratorBuildError{
                MNVGeneratorBuildError::type::CovarianceMatrixIsNotPositiveDefinite,
                ERRMSG("The covariance matrix provided is not positive-definite. It could be the wrong matrix or there's not enough values provided to construct the positive-definite one\n")};

        case MatrixDefinition::PositiveDefinite: // ok
            break;
        default:
            break;
        }

        return MNVGenerator<T, Dim>(
            internal::doCholetskyDecomposition(covariance), mean, seed);
    }

    template <typename T, size_t Dim>
    std::variant<MNVGenerator<T, Dim>, MNVGeneratorBuildError>
    MNVGenerator<T, Dim>::build(
        std::vector<valueVector<T, Dim>> const &statisticVectors,
        size_t seed)
    {
        return build(calculateCovarianceMatrix(statisticVectors),
                     calculateMeanVector(statisticVectors),
                     seed);
    }

    template <typename T, size_t Dim>
    void MNVGenerator<T, Dim>::seed(size_t seed)
    {
        m_generator.seed(seed);
        return;
    }

    // private constructor is used to force MNVGenerator::build()
    template <typename T, size_t Dim>
    MNVGenerator<T, Dim>::MNVGenerator(MatrixSq<T, Dim> decomposedCovariance, valueVector<T, Dim> mean, size_t seed)
        : m_decomposedCovariance(decomposedCovariance), m_mean(mean), m_seed(seed)
    {
        this->m_generator.seed(seed);
    }

    template <typename T, size_t Dim>
    valueVector<T, Dim> calculateMeanVector(std::vector<valueVector<T, Dim>> const &inputVectors)
    {
        valueVector<T, Dim> result{};

        for (auto &&vec : inputVectors)
        {
            result = internal::addVectors(result, vec);
        }

        for (size_t i = 0; i < result.size(); i++)
        {
            result[i] /= static_cast<T>(inputVectors.size());
        }

        return result;
    }

    template <typename T, size_t Dim>
    MatrixSq<T, Dim> calculateCovarianceMatrix(std::vector<valueVector<T, Dim>> const &inputVectors)
    {
        MatrixSq<T, Dim> result{};

        valueVector<T, Dim> mean = calculateMeanVector(inputVectors);

        for (size_t i = 0; i < result.size(); i++)
        {
            for (size_t j = 0; j < result.size(); j++)
            {
                for (size_t k = 0; k < inputVectors.size(); k++)
                {
                    result[i][j] += (inputVectors[k][i] - mean[i]) * (inputVectors[k][j] - mean[j]);
                }

                result[i][j] /= static_cast<T>(inputVectors.size());
            }
        }

        return result;
    }
} // namespace mnv

#endif // MNV_IMPL_HPP