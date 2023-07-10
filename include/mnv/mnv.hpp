#ifndef MNV_HPP
#define MNV_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mnv/export_static.h>
#include <random>
#include <string_view>
#include <variant>
#include <vector>

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
     * @tparam Dim Length of array, can be abstracted
     */
    template <typename T, size_t Dim>
    using valueVector = std::array<T, Dim>;

    /**
     * @brief Square matrix, array of arrays, size is statically defined
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Dim Matrix size
     */
    template <typename T, size_t Dim>
    using MatrixSq = valueVector<valueVector<T, Dim>, Dim>;

    struct MNVGeneratorBuildError
    {
        enum class type
        {
            CovarianceMatrixIsNotPositiveDefinite,
            CovarianceMatrixIsNotSymmetric,
        } type;
        std::string_view message;
    };

    template <typename T, size_t Dim>
    class MNVGenerator
    {
    public:
        valueVector<T, Dim> nextValue();

        static std::variant<MNVGenerator<T, Dim>, MNVGeneratorBuildError>
        build(
            MatrixSq<T, Dim> const &covariance,
            valueVector<T, Dim> const &mean,
            size_t seed);

    private:
        // private constructor is used to force MNVGenerator::build()
        MNVGenerator(MatrixSq<T, Dim> decomposedCovariance, valueVector<T, Dim> mean, size_t seed);

        // distribution params
        MatrixSq<T, Dim> m_decomposedCovariance{};
        valueVector<T, Dim> m_mean{};

        // rng params
        size_t m_seed{0};
        std::mt19937 m_generator{};
        std::normal_distribution<T> distribution{0, 1};
    };

    /**
     * @brief Function to statistically calculate covariance matrix using statistic data
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Dim Matrix size
     * @param input_vectors Input vectors to calculate covariance matrix
     * @return MatrixSq<T, Dim> The covariance matrix
     */
    template <typename T, size_t Dim>
    MatrixSq<T, Dim> calculateCovarianceMatrix(std::vector<valueVector<T, Dim>> const &input_vectors);

    /**
     * @brief Function to statistically calculate mean vector using statistic data
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Dim Vectors' size
     * @param input_vectors Input vectors to calculate mean vector
     * @return valueVector<T, Dim> The mean vector
     */
    template <typename T, size_t Dim>
    valueVector<T, Dim> calculateMeanVector(std::vector<valueVector<T, Dim>> const &input_vectors);

} // namespace mnv

#include <mnv/mnv-impl.hpp>

#endif // MNV_HPP