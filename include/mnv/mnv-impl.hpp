#ifndef MNV_IMPL_HPP
#define MNV_IMPL_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <variant>
#include <vector>

namespace mnv
{
    template <typename T, size_t Dim>
    using valueVector = std::array<T, Dim>;

    template <typename T, size_t LenRows, size_t LenCols>
    using Matrix = valueVector<valueVector<T, LenCols>, LenRows>;

    template <typename T, size_t Dim>
    using MatrixSq = Matrix<T, Dim, Dim>;

    namespace internal
    {
        template <typename T, size_t Dim>
        T vectorDotProduct(valueVector<T, Dim> first, valueVector<T, Dim> second)
        {
            T result{};

            for (size_t i = 0; i < first.size(); i++)
            {
                result += first[i] * second[i];
            }

            return result;
        }

        template <typename T, size_t Dim>
        valueVector<T, Dim> addVectors(valueVector<T, Dim> first, valueVector<T, Dim> second)
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
        T calculateMinor(MatrixSq<T, Dim> matrix, int minor_order)
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
                        minorMatrix[i][j] = matrix[i + (column + 1)][j + (j >= row)];
                    }
                }
                T otherMinor = calculateMinor(minorMatrix, minor_order - 1);
                sum = isOdd(row)
                          ? sum + otherMinor
                          : sum - otherMinor;
            }

            return sum;
        }

        enum class MatrixDefinition
        {
            ZeroDefinite,
            PositiveDefinite,
            PositiveSemidefinite,
            NegativeDefinite,
            NegativeSemidefinite,
            Undefinite
        };

        template <typename T, size_t Dim>
        MatrixDefinition defineMatrix(MatrixSq<T, Dim> matrix)
        {
            MatrixDefinition result = MatrixDefinition::ZeroDefinite;
            bool firstZero = false;
            for (size_t i = 1; i < matrix.size() + 1; i++)
            {
                T minor = calculateMinor(matrix, i);
                if (minor > 0)
                {
                    switch (result)
                    {
                    case MatrixDefinition::ZeroDefinite:
                        result = firstZero
                                     ? MatrixDefinition::PositiveSemidefinite
                                     : MatrixDefinition::PositiveDefinite;
                        break;

                    case MatrixDefinition::NegativeDefinite:
                    case MatrixDefinition::NegativeSemidefinite:
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
                        result = firstZero
                                     ? MatrixDefinition::NegativeSemidefinite
                                     : MatrixDefinition::NegativeDefinite;
                        break;

                    case MatrixDefinition::PositiveDefinite:
                    case MatrixDefinition::PositiveSemidefinite:
                        result = MatrixDefinition::Undefinite;
                        break;

                    default:
                        break;
                    }
                }

                if (minor == 0)
                {
                    switch (result)
                    {
                    case MatrixDefinition::PositiveDefinite:
                        result = MatrixDefinition::PositiveSemidefinite;
                        break;
                    case MatrixDefinition::NegativeDefinite:
                        result = MatrixDefinition::NegativeSemidefinite;
                        break;

                    case MatrixDefinition::ZeroDefinite:
                        firstZero = true;

                    default:
                        break;
                    }
                }
            }

            return result;
        }

        template <typename T, size_t Dim>
        T sumOfProductsUntil(valueVector<T, Dim> vectorA, valueVector<T, Dim> vectorB, size_t idx)
        {
            T sum = 0;

            for (size_t i = 0; i < idx; i++)
            {
                sum += vectorA[i] * vectorB[i];
            }

            return sum;
        }

        template <typename T, size_t Dim>
        T sumOfSquaresUntil(valueVector<T, Dim> vector, size_t idx)
        {
            return sumOfProductsUntil(vector, vector, idx);
        }

        template <typename MatrixType>
        inline bool isMatrixSymmetric(MatrixType matrix)
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
        MatrixSq<T, Dim> doCholetskyDecomposition(MatrixSq<T, Dim> matrix)
        {
            MatrixSq<T, Dim> result{};
            // 3. Decomposition itself
            for (size_t j = 0; j < matrix.size(); j++)
            {
                for (size_t i = j; i < matrix.size(); i++)
                {
                    if ((i == 0) && (j == 0))
                    {
                        result[i][j] = std::sqrt(matrix[i][i]);
                        continue;
                    }

                    if (j == 0)
                    {
                        result[i][j] = matrix[i][j] / result[0][0];
                        continue;
                    }

                    if (i == j)
                    {
                        result[i][j] = std::sqrt(matrix[i][i] - sumOfSquaresUntil(result[i], j));
                        continue;
                    }
                    result[i][j] = (matrix[i][j] - sumOfProductsUntil(result[i], result[j], j)) / result[j][j];
                }
            }

            return result;
        }

        template <typename T, size_t Dim>
        valueVector<T, Dim> multiplyMatrixByVector(MatrixSq<T, Dim> matrix, valueVector<T, Dim> vector)
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
            internal::multiplyMatrixByVector(randomStandardNormalVector, m_decomposedCovariance);
        return internal::addVectors(multipliedVector, m_mean);
    }

    enum class MNVGeneratorBuildError : size_t;

    template <typename T, size_t Dim>
    std::variant<MNVGenerator<T, Dim>, MNVGeneratorBuildError> MNVGenerator<T, Dim>::build(
        MatrixSq<T, Dim> covariance,
        valueVector<T, Dim> mean,
        size_t seed)
    {
        using ::mnv::internal::MatrixDefinition;

        // 1. Check for symmetric matrix

        if (!internal::isMatrixSymmetric(covariance))
        {
            // error: ABSOLUTELY wrong matrix
            return MNVGeneratorBuildError::CovarianceMatrixIsNotSymmetric;
        }

        // 2. Check for positive-definite matrix

        MatrixDefinition def = internal::defineMatrix(covariance);
        switch (def)
        {
        case MatrixDefinition::NegativeDefinite:     // error: how tf you did that (wrong matrix)?
        case MatrixDefinition::NegativeSemidefinite: // error: how tf you did that (wrong matrix)?
        case MatrixDefinition::Undefinite:           // error: how tf you did that (wrong matrix)?
        case MatrixDefinition::PositiveSemidefinite: // error: not enough values
            return MNVGeneratorBuildError::CovarianceMatrixIsNotPositiveDefinite;

        case MatrixDefinition::ZeroDefinite:
        case MatrixDefinition::PositiveDefinite: // ok
            break;
        }

        return MNVGenerator(internal::doCholetskyDecomposition(covariance), mean, seed);
    }

    // private constructor is used to force MNVGenerator::build()
    template <typename T, size_t Dim>
    MNVGenerator<T, Dim>::MNVGenerator(MatrixSq<T, Dim> decomposedCovariance, valueVector<T, Dim> mean, size_t seed)
        : m_decomposedCovariance(decomposedCovariance), m_mean(mean), m_seed(seed)
    {
        this->m_generator.seed(seed);
    }

    template <typename T, size_t Dim>
    valueVector<T, Dim> calculateMeanVector(std::vector<valueVector<T, Dim>> const &input_vectors)
    {
        valueVector<T, Dim> result{};

        for (auto &&vec : input_vectors)
        {
            result = internal::addVectors(result, vec);
        }

        for (size_t i = 0; i < result.size(); i++)
        {
            result[i] /= static_cast<T>(input_vectors.size());
        }

        return result;
    }

    template <typename T, size_t Dim>
    MatrixSq<T, Dim> calculateCovarianceMatrix(std::vector<valueVector<T, Dim>> const &input_vectors)
    {
        MatrixSq<T, Dim> result{};

        valueVector<T, Dim> mean = calculateMeanVector(input_vectors);

        for (size_t i = 0; i < result.size(); i++)
        {
            for (int j = 0; j < result.size(); j++)
            {
                for (int k = 0; k < input_vectors.size(); k++)
                {
                    result[i][j] += (input_vectors[k][i] - mean[i]) * (input_vectors[k][j] - mean[j]);
                }

                result[i][j] /= input_vectors.size();
            }
        }

        return result;
    }
} // namespace mnv

#endif // MNV_IMPL_HPP