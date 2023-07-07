#ifndef MNV_HPP
#define MNV_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <mnv/export_static.h>
#include <random>
#include <vector>

namespace mnv
{
    template <typename T, size_t Len>
    using valueVector = std::array<T, Len>;

    template <typename T, size_t LenRows, size_t LenCols>
    using Matrix = valueVector<valueVector<T, LenCols>, LenRows>;

    template <typename T, size_t Len>
    using MatrixSq = Matrix<T, Len, Len>;

    namespace internal
    {
        template <typename T, size_t Len>
        valueVector<T, Len> multiplyVectors(valueVector<T, Len> first, valueVector<T, Len> second)
        {
            T result;

            for (int i = 0; i < first.size(); i++)
            {
                result += first[i] * second[i];
            }

            return result;
        }

        template <typename T, size_t Len>
        valueVector<T, Len> addVectors(valueVector<T, Len> first, valueVector<T, Len> second)
        {
            valueVector<T, Len> result;

            for (int i = 0; i < first.size(); i++)
            {
                result[i] = first[i] + second[i];
            }

            return result;
        }

        template <typename T>
        T calculateMean(std::vector<T> &input_values)
        {
            T result;

            for (auto &&value : input_values)
            {
                result += value;
            }

            return result / input_values.size();
        }

        inline bool isOdd(size_t number)
        {
            return static_cast<bool>(number & 1);
        }

        template <typename T, size_t Len>
        T calculateMinor(MatrixSq<T, Len> matrix, int minor_order)
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
                MatrixSq<T, Len> minorMatrix;
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

        template <typename T, size_t Len>
        MatrixDefinition defineMatrix(MatrixSq<T, Len> matrix)
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

        template <typename T, size_t Len>
        T sumOfSquaresUntil(valueVector<T, Len> vector, size_t idx)
        {
            T sum = 0;

            for (size_t i = 0; i < idx; i++)
            {
                sum += vector[i] * vector[i];
            }

            return sum;
        }

        template <typename T, size_t Len>
        T sumOfProductsUntil(valueVector<T, Len> vectorA, valueVector<T, Len> vectorB, size_t idx)
        {
            T sum = 0;

            for (size_t i = 0; i < idx; i++)
            {
                sum += vectorA[i] * vectorB[i];
            }

            return sum;
        }

        template <typename MatrixType>
        inline bool isMatrixSymmetic(MatrixType matrix)
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

        template <typename T, size_t Len>
        MatrixSq<T, Len> doCholetskyDecomposition(MatrixSq<T, Len> matrix)
        {
            MatrixSq<T, Len> result;

            // 1. Check for symmetric matrix

            if (!isMatrixSymmetic(matrix))
            {
                // error: ABSOLUTELY wrong matrix
                return result;
            }

            // 2. Check for positive-definite matrix

            MatrixDefinition def = defineMatrix(matrix);
            switch (def)
            {
            case MatrixDefinition::NegativeDefinite:     // error: how tf you did that (wrong matrix)?
            case MatrixDefinition::NegativeSemidefinite: // error: how tf you did that (wrong matrix)?
            case MatrixDefinition::Undefinite:           // error: how tf you did that (wrong matrix)?
            case MatrixDefinition::PositiveSemidefinite: // error: not enough values
            case MatrixDefinition::ZeroDefinite:         // error: empty matrix
                return result;

            case MatrixDefinition::PositiveDefinite: // ok
                break;
            }

            // 3. Decomposition itself
            for (size_t j = 0; j < matrix.size(); j++)
            {
                for (size_t i = 0; i < matrix.size(); i++)
                {
                    if ((i == 1) && (j == 1))
                    {
                        matrix[i][j] = std::sqrt<T>(matrix[i][i]);
                        continue;
                    }

                    if (j == 1)
                    {
                        matrix[i][j] = matrix[i][j] / result[0][0];
                        continue;
                    }

                    if (i == j)
                    {
                        matrix[i][j] = std::sqrt<T>(matrix[i][i] - sumOfSquaresUntil(result[i], i));
                        continue;
                    }

                    matrix[i][j] = (matrix[i][j] - sumOfProductsUntil(result[i], result[j], i)) / result[i][i];
                }
            }

            return result;
        }
    }

    template <typename T, size_t Len>
    valueVector<T, Len> generateMNV(MatrixSq<T, Len> covariance, valueVector<T, Len> meanVector, T seed)
    {
        static std::mt19937 generator{};
        generator.seed(seed);
        std::normal_distribution<T> distribution{0, 1};

        valueVector<T, Len> randomStandardNormalVector{};
        for (size_t i = 0; i < randomStandardNormalVector.size(); i++)
        {
            randomStandardNormalVector[i] = distribution(generator);
        }

        MatrixSq<T, Len> decomposedMatrix = internal::doCholetskyDecomposition(covariance);
        valueVector<T, Len> multipliedVector = internal::multiplyVectorByMatrix(randomStandardNormalVector, decomposedMatrix);
        return internal::addVectors(multipliedVector, meanVector);
    }

    template <typename T, size_t Len>
    valueVector<T, Len> calculateMeanVector(std::vector<valueVector<T, Len>> &input_vectors)
    {
        valueVector<T, Len> result;

        for (auto &&vec : input_vectors)
        {
            result = internal::addVectors(result, vec);
        }

        for (size_t i = 0; i < result.size(); i++)
        {
            result[i] /= input_vectors.size()
        }

        return result;
    }

    template <typename T, size_t Len>
    MatrixSq<T, Len> calculateCovarianceMatrix(std::vector<valueVector<T, Len>> &input_vectors)
    {
        MatrixSq<T, Len> result;

        valueVector<T, Len> mean = calculateMeanVector(input_vectors);

        for (int i = 0; i < result.size(); i++)
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

    int add(int a, int b);
} // namespace mnv

#endif // MNV HPP