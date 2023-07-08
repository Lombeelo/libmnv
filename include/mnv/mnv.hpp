#ifndef MNV_HPP
#define MNV_HPP

#include <array>
#include <mnv/export_static.h>
#include <vector>

#include <mnv/mnv-impl.hpp>
/**
 * @file mnv.hpp Main interface file of the mnv library
 * @author Dmitry Parfenyuk (cekunda.rf@gmail.com)
 * @brief A header-only library to generate random multivariate normal values
 * @version 0.1
 * @date 2023-07-08
 *
 * @copyright Copyright (c) 2023
 *
 */

namespace mnv
{
    /**
     * @brief Basic vector, can be abstracted away
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Len Length of array, can be abstracted
     */
    template <typename T, size_t Len>
    using valueVector = std::array<T, Len>;

    /**
     * @brief Basic matrix, used only for definitions
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam LenRows Rows count
     * @tparam LenCols Columns count
     */
    template <typename T, size_t LenRows, size_t LenCols>
    using Matrix = valueVector<valueVector<T, LenCols>, LenRows>;

    /**
     * @brief Square matrix, array of arrays, size is statically defined
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Len Matrix size
     */
    template <typename T, size_t Len>
    using MatrixSq = Matrix<T, Len, Len>;

    /**
     * @brief Function to statistically calculate covariance matrix using statistic data
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Len Matrix size
     * @param input_vectors Input vectors to calculate covariance matrix
     * @return MatrixSq<T, Len> The covariance matrix
     */
    template <typename T, size_t Len>
    MatrixSq<T, Len> calculateCovarianceMatrix(std::vector<valueVector<T, Len>> &input_vectors);

    /**
     * @brief Function to statistically calculate mean vector using statistic data
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Len Vectors' size
     * @param input_vectors Input vectors to calculate mean vector
     * @return valueVector<T, Len> The mean vector
     */
    template <typename T, size_t Len>
    valueVector<T, Len> calculateMeanVector(std::vector<valueVector<T, Len>> &input_vectors);

    /**
     * @brief The main generator function for the multivariate normal random values (unfinished)
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Len Generated vectors' size
     * @param covariance Covariance matrix of the distribution. Must be symmetrical and positive-definite
     * @param meanVector Mean vector
     * @param seed Generator seed. Internal rng state is always reset
     * @return valueVector<T, Len> Generated multivariate normal value
     */
    template <typename T, size_t Len>
    valueVector<T, Len> generateMNV(MatrixSq<T, Len> covariance, valueVector<T, Len> meanVector, T seed);

    int add(int a, int b);
} // namespace mnv

#endif // MNV_HPP